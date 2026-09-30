# SPDX-License-Identifier: Apache-2.0
"""`csf-demo`: the viser server, the per-client lifecycle and the playback loop.

Run::

    csf-demo [--model Kimodo-G1-RP-v1] [--port 7860]
    csf-demo --list-backbones     # which backbones are loadable here (no GPU)

Importing this module does not import torch, viser or kimodo, so
`--list-backbones` stays cheap.
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
import time

from csf.app import cuda_health, generate, scene, theme, timeline
from csf.app.panels import backbone, controls, examples, playback, view
from csf.app.session import Session, evict_models_except, place_text_encoder

DEFAULT_MODEL = "Kimodo-G1-RP-v1"
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 7860
FLOOR_LEN = 20.0
CUDA_CHECK_SECONDS = 5.0
FALLBACK_PLAYBACK_FPS = 60.0

_WELCOME_MD = """
### CSF: Contextual Safety Filtering for motion generators

CSF filters a frozen text-to-motion generator against natural-language
safety rules. The same command can be fine or harmful depending on the scene:
kicking a ball is fine, kicking a person is not.

* Pick an **example**, or type prompts on the **timeline**
  at the bottom and put a **person** in the scene.
* **Generate**: the **orange** character is the raw generator output, the
  **blue** one is the same command and seed with CSF.
* **Runtime shield**: generate with nobody present, press Play, then tick
  *Person in scene* mid-motion. The rest of the CSF motion is redirected.
* **Space** plays/pauses, **←/→** step frames.
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="csf-demo", description="Interactive CSF demo.")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"Backbone loaded at startup (default: {DEFAULT_MODEL}).")
    parser.add_argument("--host", default=os.environ.get("SERVER_NAME", DEFAULT_HOST))
    parser.add_argument("--port", type=int, default=int(os.environ.get("SERVER_PORT", DEFAULT_PORT)))
    parser.add_argument("--list-backbones", action="store_true",
                        help="Print which backbones are loadable on this machine and exit.")
    parser.add_argument("--no-preload", action="store_true",
                        help="Do not load the default backbone at startup.")
    return parser


def print_backbones() -> None:
    from csf.backbones import registry  # noqa: PLC0415

    available = set(registry.available_names())
    for name in registry.all_names():
        print(f"{'available' if name in available else 'unavailable'}: {name}")


def panel_builders() -> list:
    """The sidebar, in workflow order."""
    return [backbone.build, examples.build, controls.build_scene, controls.build_generate,
            controls.build_filter, playback.build, view.build]


def resolve_device(device: object | None = None) -> str:
    """A device string ('cuda:0' / 'cpu'); kimodo's loader rejects torch.device."""
    import torch  # noqa: PLC0415

    if device is None:
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    return str(torch.device(device))


class App:
    """The server and everything shared across clients.

    Per-client state lives on `Session`.  This owns the device, the
    one-bundle-at-a-time model cache, the shared LLM2Vec text encoder, the lock
    that serialises GPU work, the safety policy and the playback loop.
    """

    def __init__(self, default_model_name: str = DEFAULT_MODEL, host: str = DEFAULT_HOST,
                 port: int = DEFAULT_PORT, device: object | None = None) -> None:
        import viser  # noqa: PLC0415

        from csf.backbones import registry  # noqa: PLC0415

        self.default_model_name = default_model_name
        self.default_spec = registry.get_spec(default_model_name)
        self.host = host
        self.device = resolve_device(device)
        self.floor_len = FLOOR_LEN
        self.gpu_lock = threading.Lock()
        self.cuda_healthy = True
        self._stop_event = threading.Event()

        self.sessions: dict[int, Session] = {}
        self.models: dict[str, object] = {}
        self.text_encoder: object | None = None
        self.text_encoder_device: object | None = None
        self.embedding_cache: dict = {}
        self.shield_refs: dict = {}
        self.policy = generate.load_policy()

        self.server = viser.ViserServer(host=host, port=port, label="CSF",
                                        enable_camera_keyboard_controls=False)
        self.server.scene.world_axes.visible = False
        self.server.scene.set_up_direction("+y")
        self.server.on_client_connect(self.on_client_connect)
        self.server.on_client_disconnect(self.on_client_disconnect)

    # ----------------------------------------------------------- models
    def _make_resident(self, name: str, session: Session | None = None) -> object:
        """Evict the others, ready the text encoder, then build (call under `gpu_lock`).

        The shared LLM2Vec is loaded (onto the GPU) before anything else, and the
        gate's fixed texts are encoded while it is there; for ECHO and
        MotionHiFlow it then moves to the CPU before their weights allocate.
        """
        from csf.backbones import registry  # noqa: PLC0415
        from csf.backbones.base import ensure_text_encoder  # noqa: PLC0415

        spec = registry.get_spec(name)
        if session is not None:
            evict_models_except(session, name)
        ensure_text_encoder(self)
        generate.warm_embedding_cache(self)
        place_text_encoder(self, spec)
        if name not in self.models:
            self.models[name] = registry.load_backbone(name, self)
        return self.models[name]

    def load_model(self, session: Session, name: str) -> object:
        return self._make_resident(name, session)

    def preload_default(self) -> None:
        """Load the default backbone at startup, then wire clients that connected meanwhile."""
        name = self.default_model_name
        if not self.default_spec.available():
            print(f"[csf] preload skipped: '{name}' is not loadable here")
            return
        with self.gpu_lock:
            print(f"[csf] preloading '{name}'...")
            self._make_resident(name)
        for session in list(self.sessions.values()):
            if session.model_name == name and session.bundle is None:
                backbone.load_backbone(self, session, name)

    # --------------------------------------------------------- clients
    def client_active(self, client_id: int) -> bool:
        return client_id in self.sessions

    def on_client_connect(self, client: object) -> None:
        with client.gui.add_modal("Welcome", size="xl", show_close_button=True,
                                  save_choice="csf.app.welcome_ack") as modal:
            client.gui.add_markdown(_WELCOME_MD)
            client.gui.add_button("Got it").on_click(lambda _event: modal.close())

        session = Session(client=client, model_name=self.default_model_name, app=self,
                          spec=self.default_spec)
        scene.setup_scene(self, session)
        self.sessions[client.client_id] = session
        for build_panel in panel_builders():
            build_panel(self, session)
        examples.sync_scene(session)
        theme.install_notification_css(client)
        timeline.bring_up_timeline(client, session.model_fps,
                                   timeline.prompt_schedule_for(session.spec))
        timeline.register_callbacks(self, session)
        if session.model_name in self.models and session.bundle is None:
            backbone.load_backbone(self, session, session.model_name)

    def on_client_disconnect(self, client: object) -> None:
        self.sessions.pop(client.client_id, None)

    # ----------------------------------------------------- scene/theme
    def apply_theme(self, session: Session, dark_mode: bool | None = None) -> None:
        checkbox = getattr(session.gui, "gui_dark_mode_checkbox", None)
        if dark_mode is None:
            dark_mode = bool(getattr(checkbox, "value", False))
        theme.configure_theme(session.client, dark_mode=dark_mode, grid=session.grid,
                              titlebar_dark_mode_checkbox_uuid=getattr(checkbox, "uuid", None))

    def set_start_direction_visible(self, session: Session, visible: bool) -> None:
        if session.origin_marker is not None:
            session.origin_marker.set_visible(visible)

    def add_character_motion(self, session: Session, **kwargs) -> object | None:
        return scene.add_character_motion(self, session, **kwargs)

    def clear_motions(self, session: Session) -> None:
        """Remove every character and protected person from a client's scene."""
        for motion in list(session.motions.values()):
            motion.clear()
        session.motions.clear()
        for human in list(session.humans.values()):
            human.remove()
        session.humans.clear()

    def set_human(self, session: Session, slot: str, placement, visible: bool) -> None:
        """Place (or create) the protected person in front of one character."""
        from csf.app.entities import HumanFigure  # noqa: PLC0415

        human = session.humans.get(slot)
        if human is None:
            session.humans[slot] = HumanFigure(session.client, slot, placement, visible=visible)
        else:
            human.place(placement)
            human.visible = visible

    def show_humans(self, session: Session, visible: bool) -> None:
        """Show or hide the protected person(s), creating defaults before any generation."""
        from csf.app.entities import default_placement  # noqa: PLC0415

        if not session.humans and visible:
            for slot, x in ((generate.UNFILTERED, generate.X_UNFILTERED),
                            (generate.FILTERED, generate.X_FILTERED)):
                self.set_human(session, slot, default_placement(x), True)
            return
        for human in session.humans.values():
            human.visible = visible

    def set_frame(self, session: Session, frame_idx: int, update_timeline: bool = True) -> None:
        if not self.client_active(session.client_id):
            return
        session.frame_idx = frame_idx
        if update_timeline:
            session.client.timeline.set_current_frame(frame_idx)
        for motion in list(session.motions.values()):
            motion.set_frame(frame_idx)

    def check_cuda_health(self) -> bool:
        return cuda_health.check_cuda_health(self)

    # -------------------------------------------------------- playback
    def _playback_fps(self) -> float:
        rates = [s.model_fps for s in self.sessions.values() if s.model_fps > 0]
        return max(rates) * 2.0 if rates else FALLBACK_PLAYBACK_FPS

    def stop(self) -> None:
        if self._stop_event.is_set():
            return
        self._stop_event.set()
        self.server.stop()

    def run(self) -> None:
        """Serve until `stop()` or Ctrl-C, advancing every playing session."""
        print(f"[csf] serving on http://{self.host}:{self.server.get_port()}")
        tick = 0
        next_cuda_check = time.monotonic()
        try:
            while not self._stop_event.is_set():
                started = time.time()
                fps = self._playback_fps()
                for session in list(self.sessions.values()):
                    if not session.playing:
                        continue
                    interval = playback.frame_update_interval(fps, session.playback_speed,
                                                              session.model_fps)
                    if tick % interval:
                        continue
                    self.set_frame(session, playback.next_frame_index(session.frame_idx,
                                                                      session.max_frame_idx))
                if time.monotonic() >= next_cuda_check:
                    next_cuda_check = time.monotonic() + CUDA_CHECK_SECONDS
                    try:
                        self.check_cuda_health()
                    except Exception as exc:  # noqa: BLE001 - one probe must not stop playback
                        print(f"[csf] CUDA probe raised: {exc!r}", file=sys.stderr)
                time.sleep(max(0.0, 1.0 / fps - (time.time() - started)))
                tick = (tick + 1) % max(1, int(fps))
        except KeyboardInterrupt:
            print("\n[csf] shutting down")
            self.stop()


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    from csf.backbones import registry  # noqa: PLC0415

    try:
        spec = registry.get_spec(args.model)
    except KeyError as exc:
        parser.error(exc.args[0])
    if args.list_backbones:
        print_backbones()
        return
    if not spec.available():
        print(f"[csf] warning: '{args.model}' is not loadable here; pick another in the GUI.")
    if not args.no_preload:
        # Import kimodo.demo on the main thread first: its package has import
        # cycles that deadlock when two threads enter it at once.
        import kimodo.demo  # noqa: F401,PLC0415
    app = App(default_model_name=args.model, host=args.host, port=args.port)
    if not args.no_preload:
        threading.Thread(target=app.preload_default, name="csf-preload", daemon=True).start()
    app.run()


if __name__ == "__main__":
    main()

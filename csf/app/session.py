# SPDX-License-Identifier: Apache-2.0
"""Per-client state, and the one-backbone-at-a-time GPU memory policy.

No torch, viser or kimodo imports at module level.
"""
from __future__ import annotations

from dataclasses import KW_ONLY, dataclass, field
from types import SimpleNamespace

DEFAULT_PLAYBACK_SPEED = 1.0
DEFAULT_CUR_DURATION = 4.0
#: Kimodo-G1's frame rate, so the timeline is right before anything loads.
DEFAULT_MODEL_FPS = 30.0


@dataclass(eq=False)
class Session:
    """Everything one connected browser client owns.

    Attributes:
        client: The viser `ClientHandle` (with the kimodo-viser `.timeline`).
        model_name: The selected backbone.
        app: The owning `App`.
        spec: `BackboneSpec` of `model_name`.
        bundle: The loaded `ModelBundle` (a strong reference to the weights).
        handle: The filter handle attached to the loaded model.
        motions: Scene characters by slot name ("unfiltered", "filtered", ...).
        gui: Widget handles, stashed by the panel that created them.
        filter_defaults: The loaded backbone's filter settings (gamma, ...).
        grid, origin_marker: Scene handles.
        frame_idx, playing, playback_speed, cur_duration, max_frame_idx,
        model_fps, skeleton: Playback and timeline state.
        last_prompt_texts, last_prompt_embeddings, last_prompt_lengths: Read
            and written by kimodo's `CachedTextEncoder.session_context` inside
            Kimodo's forward pass.
        last: The last generation (`csf.app.generate.Generation`), or None.
        humans: The protected-person meshes in the scene, by slot name.
    """

    client: object
    model_name: str
    _: KW_ONLY
    app: object | None = None
    spec: object | None = None
    bundle: object | None = None
    handle: object | None = None
    motions: dict[str, object] = field(default_factory=dict)
    gui: SimpleNamespace = field(default_factory=SimpleNamespace)
    filter_defaults: dict[str, object] = field(default_factory=dict)

    grid: object | None = None
    origin_marker: object | None = None

    frame_idx: int = 0
    playing: bool = False
    playback_speed: float = DEFAULT_PLAYBACK_SPEED
    cur_duration: float = DEFAULT_CUR_DURATION
    max_frame_idx: int = int(DEFAULT_CUR_DURATION * DEFAULT_MODEL_FPS) - 1
    model_fps: float = DEFAULT_MODEL_FPS
    skeleton: object | None = None

    last_prompt_texts: list[str] | None = None
    last_prompt_embeddings: object | None = None
    last_prompt_lengths: list[int] | None = None

    last: object | None = None
    humans: dict[str, object] = field(default_factory=dict)

    @property
    def client_id(self) -> int:
        return self.client.client_id


def _empty_cuda_cache() -> None:
    try:
        import torch  # noqa: PLC0415
    except ImportError:  # pragma: no cover
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def evict_models_except(session: Session, keep_name: str) -> None:
    """Free every cached bundle except `keep_name` (one backbone fits a 24 GB card).

    The shared LLM2Vec lives on `app.text_encoder` and survives eviction.  Each
    session's strong references to an evicted bundle (and the handle wrapping
    its model) are dropped too, or the weights would never be freed.
    """
    app = session.app
    if app is None:
        return
    victims = [name for name in list(app.models) if name != keep_name]
    if not victims:
        return
    import gc  # noqa: PLC0415

    evicted = [app.models.pop(name) for name in victims]
    for name in victims:
        print(f"[csf] unloaded '{name}'")
    others = list(getattr(app, "sessions", {}).values())
    if not any(other is session for other in others):
        others.append(session)
    for other in others:
        if any(other.bundle is bundle for bundle in evicted):
            other.bundle = None
        model = getattr(other.handle, "_model", None)
        if model is not None and any(model is getattr(b, "model", None) for b in evicted):
            other.handle = None
    evicted.clear()
    gc.collect()
    _empty_cuda_cache()


def place_text_encoder(app: object, spec: object) -> None:
    """Keep the shared LLM2Vec on the GPU only for backbones that condition on it.

    ECHO and MotionHiFlow bring their own CLIP; for them the ~15 GB encoder
    moves to the CPU before their weights allocate.  The gate and the
    safe-reference selection still use it there, through a per-text cache.
    """
    encoder = getattr(app, "text_encoder", None) if app is not None else None
    if encoder is None or not hasattr(encoder, "to"):
        return
    want = app.device if (spec is None or spec.shared_text_encoder) else "cpu"
    if str(getattr(app, "text_encoder_device", None)) == str(want):
        return
    encoder.to(want)
    app.text_encoder_device = want
    _empty_cuda_cache()
    print(f"[csf] text encoder -> {want} (for '{getattr(spec, 'name', '?')}')")

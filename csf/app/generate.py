# SPDX-License-Identifier: Apache-2.0
"""One generation: an unfiltered pass and a CSF pass, side by side.

Both passes use the same prompt schedule and seed; the only difference between
the orange (unfiltered) and the blue (CSF) character is the filter.  Between
the passes, the unfiltered clip's root motion picks each segment's safe
reference (travelling: the nearest neutral locomotion; in place: the stand).

After generating, the CSF clip is armed with the runtime shield when no person
was in the scene, so a person entering view during playback can still redirect
the rest of the motion (Sec. III-F).
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass

from csf.app import ui

ORANGE = (255, 140, 0)
BLUE = (100, 149, 237)
X_UNFILTERED = +0.9
X_FILTERED = -0.9
UNFILTERED = "unfiltered"
FILTERED = "filtered"

#: The perceived label a person in the scene contributes.
PERSON = "a person"

#: Shield reference clips: length (s), and the neutral walk used to calibrate tau.
_SHIELD_REF_SECONDS = 4.5
_SHIELD_REF_STEPS = 40
_SHIELD_WALK = "a person walks forward in a neutral way"


@dataclass
class Generation:
    """What the last Generate produced, for the scene and the runtime shield."""

    prompts: list
    durations: list
    seed: int
    entities: list
    unfiltered: dict
    filtered: dict | None = None
    shield: object | None = None


@dataclass
class Policy:
    """The safety specification, read once at startup."""

    cfg: object
    references: object
    stand_speed_threshold: float


def load_policy() -> Policy:
    from csf.filter.config import load_filter_config, load_safe_reference_policy  # noqa: PLC0415

    references, threshold = load_safe_reference_policy()
    return Policy(cfg=load_filter_config(), references=references, stand_speed_threshold=threshold)


def cached_encoder(app: object, bundle: object | None = None):
    """The shared text encoder behind a process-wide per-text cache, or None.

    Matches LLM2Vec's call contract, ``enc(texts) -> (feat[B, 1, d], None)``.
    The cache is what keeps the gate cheap when LLM2Vec sits on the CPU
    (while ECHO or MotionHiFlow is loaded).
    """
    raw = getattr(app, "text_encoder", None)
    if raw is None and bundle is not None:
        raw = getattr(getattr(bundle, "model", None), "text_encoder", None)
    if raw is None or not callable(raw):
        return None
    cache = app.embedding_cache

    def encode(texts):
        import torch  # noqa: PLC0415

        missing = [t for t in texts if t not in cache]
        if missing:
            feats, _ = raw(missing)
            feats = feats[:, None] if feats.dim() == 2 else feats
            for text, feat in zip(missing, feats):
                cache[text] = feat.detach().cpu()
        return torch.stack([cache[t] for t in texts], dim=0), None

    return encode


def warm_embedding_cache(app: object) -> None:
    """Encode the gate's fixed texts once (rules, contrasts, library, entities)."""
    encode = cached_encoder(app)
    if encode is None:
        return
    cfg, refs = app.policy.cfg, app.policy.references
    from csf.filter.handle import text_sanitizer  # noqa: PLC0415

    sanitize = text_sanitizer()
    texts = [*cfg.unsafe_texts, *cfg.protected_entities, *cfg.neutral_entities,
             *cfg.benign_contrasts, *refs.library, *refs.locomotion, refs.stand, PERSON]
    todo = sorted({sanitize(t) for t in texts} - set(app.embedding_cache))
    for i in range(0, len(todo), 16):
        encode(todo[i:i + 16])


def denoising_steps(model: object, requested: int) -> int:
    """Backbones trained for a fixed step count (ECHO, MotionHiFlow, ARDY) use it."""
    trained = getattr(model, "default_denoising_steps", None)
    return int(trained) if trained else int(requested)


@contextlib.contextmanager
def hold_gpu(app: object, *, client: object | None = None, notification: object | None = None):
    """Hold `app.gpu_lock`, telling the user when another client has it."""
    lock = app.gpu_lock
    if not lock.acquire(blocking=False):
        waiting = None
        if notification is not None:
            notification.body = "Waiting for the GPU (another generation is running)..."
        elif client is not None:
            waiting = client.add_notification(
                title="Waiting for the GPU...", body="Another generation is running.",
                loading=True, with_close_button=False)
        lock.acquire()
        if waiting is not None:
            waiting.remove()
    try:
        yield
    finally:
        lock.release()


def ensure_backbone(app: object, session: object, name: str | None = None) -> None:
    """Load the selected backbone if nothing is resident (outside the GPU lock)."""
    if session.bundle is not None:
        return
    from csf.app.panels import backbone  # noqa: PLC0415

    name = name or session.model_name
    backbone.load_backbone(app, session, name)
    if session.bundle is None:
        raise RuntimeError(f"'{name}' failed to load; see the error above.")


def attach_filter(app: object, session: object, opts: dict) -> object:
    """Attach CSF to the loaded model with this backbone's settings (Table I)."""
    spec = session.spec
    overrides = spec.filter.as_overrides()
    overrides["gamma"] = float(opts.get("gamma", overrides["gamma"]))
    overrides["rho"] = float(opts.get("rho", app.policy.cfg.rho))
    overrides["stand_speed_threshold"] = app.policy.stand_speed_threshold
    cfg = app.policy.cfg.with_backbone(**overrides)
    handle = spec.attach_filter(session.bundle.model, cfg, policy=app.policy.references,
                                encoder=cached_encoder(app, session.bundle))
    session.handle = handle
    return handle


def _call_model(session: object, prompts: list, durations: list, steps: int, opts: dict) -> dict:
    from kimodo.demo.embedding_cache import CachedTextEncoder  # noqa: PLC0415

    model = session.bundle.model

    def call() -> dict:
        return model(
            prompts, durations, steps,
            multi_prompt=True, constraint_lst=[],
            cfg_weight=opts.get("cfg_weight") or [2.0, 2.0],
            num_samples=1,
            cfg_type=opts.get("cfg_type", "separated"),
            num_transition_frames=int(opts.get("num_transition_frames", 5)),
        )

    encoder = getattr(model, "text_encoder", None)
    if isinstance(encoder, CachedTextEncoder):
        with encoder.session_context(session):
            return call()
    return call()


def choose_safe_references(app: object, session: object, handle: object, prompts: list,
                           durations: list, out_unfiltered: dict) -> dict:
    """Pick each segment's safe reference from its own unfiltered motion."""
    from csf.filter.references import net_root_speed, segment_windows, select_for_motion  # noqa: PLC0415

    root = out_unfiltered.get("smooth_root_pos")
    if root is None or not handle.has_encoder:
        return {}
    fps = float(getattr(session.bundle, "model_fps", 0) or session.model_fps)
    traj = root[0]
    pins = {}
    for text, (lo, hi) in zip(prompts, segment_windows(durations, len(traj))):
        ref, why = select_for_motion(handle.encode, text, net_root_speed(traj[lo:hi], fps),
                                     app.policy.stand_speed_threshold, app.policy.references)
        print(f"[csf] {text!r}: {why} -> safe reference {ref!r}")
        pins[text] = ref
    return pins


def run_generation(app: object, session: object, *, prompts: list, durations: list, seed: int,
                   entities: list, filter_on: bool, opts: dict) -> Generation:
    """Generate, show, and (when nobody is present) arm the shield."""
    if not app.cuda_healthy:
        raise RuntimeError("The CUDA context is corrupted; restart the server.")
    if not prompts:
        raise RuntimeError("The timeline has no prompt; add one and retry.")
    ensure_backbone(app, session, opts.get("model_name"))

    from kimodo.tools import seed_everything  # noqa: PLC0415

    with hold_gpu(app, client=session.client):
        if session.bundle is None:
            raise RuntimeError(f"'{session.model_name}' is not loaded; press Load and retry.")
        steps = denoising_steps(session.bundle.model, int(opts.get("steps", 100)))
        handle = attach_filter(app, session, opts)
        handle.set_entities(entities)

        seed_everything(seed)
        out_unfiltered = _call_model(session, prompts, durations, steps, opts)
        generation = Generation(prompts=list(prompts), durations=list(durations), seed=seed,
                                entities=list(entities), unfiltered=out_unfiltered)
        if filter_on:
            pins = choose_safe_references(app, session, handle, prompts, durations, out_unfiltered)
            handle.pin_safe_references(pins)
            # Seed before the references are built (they are generator samples
            # too), then again before sampling, so both passes share their noise.
            seed_everything(seed)
            with handle.filtering(prompts, durations, steps):
                seed_everything(seed)
                generation.filtered = _call_model(session, prompts, durations, steps, opts)

    session.last = generation
    present(app, session, generation)
    if filter_on and PERSON not in entities:
        arm_shield(app, session, generation)
    return generation


# --------------------------------------------------------------------- scene
def _numpy(x):
    import numpy as np  # noqa: PLC0415

    x = x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)
    return x[0] if x.ndim >= 1 and x.shape[0] == 1 else x


def add_pass(app: object, session: object, out: dict, *, name: str, color, x_offset: float):
    contacts = out.get("foot_contacts")
    return app.add_character_motion(
        session, name=name,
        joints_pos=out["posed_joints"][0], joints_rot=out["global_rot_mats"][0],
        foot_contacts=contacts[0] if contacts is not None else None,
        color=color, x_offset=x_offset,
    )


def present(app: object, session: object, generation: Generation) -> None:
    """Put both clips, and the protected person when present, in the scene."""
    from csf.app import entities as ent  # noqa: PLC0415

    app.clear_motions(session)
    filtered = generation.filtered is not None
    x_unf = X_UNFILTERED if filtered else 0.0
    add_pass(app, session, generation.unfiltered, name=UNFILTERED, color=ORANGE, x_offset=x_unf)
    slots = {UNFILTERED: x_unf}
    if filtered:
        add_pass(app, session, generation.filtered, name=FILTERED, color=BLUE, x_offset=X_FILTERED)
        slots[FILTERED] = X_FILTERED

    # One person per character, placed from the UNFILTERED clip so both
    # conditions share the same scene.
    base = ent.placement_from_motion(_numpy(generation.unfiltered["posed_joints"]))
    visible = PERSON in generation.entities
    for slot, x in slots.items():
        app.set_human(session, slot, ent.Placement(base.x + x, base.z, base.facing), visible)
    app.set_frame(session, 0)


# -------------------------------------------------------------------- shield
def shield_references(app: object, session: object):
    """The shield's reference clips for the loaded backbone, generated once."""
    import numpy as np  # noqa: PLC0415

    from csf.filter.shield import ShieldReferences, root_relative  # noqa: PLC0415
    from kimodo.tools import seed_everything  # noqa: PLC0415

    cached = app.shield_refs.get(session.model_name)
    if cached is not None:
        return cached
    model = session.bundle.model
    steps = denoising_steps(model, _SHIELD_REF_STEPS)
    nf = int(round(_SHIELD_REF_SECONDS * session.model_fps))
    opts = {"cfg_type": "separated"}

    def clip(text: str, seed: int) -> dict:
        seed_everything(seed)
        return _call_model(session, [text], [nf], steps, opts)

    stand = app.policy.references.stand
    person_rules = [r.unsafe for r in app.policy.cfg.rules if "person" in r.protects]
    stand_out = clip(stand, 0)
    refs = ShieldReferences(
        unsafe=np.stack([root_relative(_numpy(clip(t, 0)["posed_joints"])) for t in person_rules]),
        safe=root_relative(_numpy(stand_out["posed_joints"])),
        safe_draw=root_relative(_numpy(clip(stand, 1)["posed_joints"])),
        walk=root_relative(_numpy(clip(_SHIELD_WALK, 0)["posed_joints"])),
        stand_joints=_numpy(stand_out["posed_joints"]).astype(np.float64),
        stand_rotations=_numpy(stand_out["global_rot_mats"]).astype(np.float64),
    )
    app.shield_refs[session.model_name] = refs
    return refs


def arm_shield(app: object, session: object, generation: Generation) -> None:
    """Arm the runtime shield over the CSF clip (reference clips built on first use)."""
    from csf.filter.shield import RuntimeShield  # noqa: PLC0415

    first = session.model_name not in app.shield_refs
    note = ui.busy(session.client, "Arming the runtime shield...",
                   "One-time reference clips for this backbone.") if first else None
    try:
        with hold_gpu(app, client=session.client):
            if session.bundle is None or session.last is not generation:
                return
            refs = shield_references(app, session)
        generation.shield = RuntimeShield.arm(
            _numpy(generation.filtered["posed_joints"]),
            _numpy(generation.filtered["global_rot_mats"]),
            refs, fps=session.model_fps)
        if note is not None:
            ui.finish(note, "Runtime shield armed",
                      "Tick 'Person in scene' during playback to bring a person into view.")
    except Exception as exc:  # noqa: BLE001 - the shield is optional
        if note is not None:
            ui.fail(note)
        ui.toast_error(session.client, "Runtime shield unavailable", str(exc))


def person_enters_view(app: object, session: object) -> None:
    """The scene changes during execution: run the shield at the current frame."""
    import torch  # noqa: PLC0415

    generation = session.last
    app.show_humans(session, True)
    if generation is None or generation.shield is None:
        return
    shield = generation.shield
    if shield.engaged:
        return
    frame = int(session.frame_idx)
    if not shield.step(frame, context_active=True):
        ui.notify(app, session, "Person in view",
                  "The rest of the motion stays above the runtime threshold; nothing to redirect.")
        return
    motion = session.motions.get(FILTERED)
    if motion is not None:
        from kimodo.skeleton import global_rots_to_local_rots  # noqa: PLC0415

        ref = motion.joints_pos
        joints = torch.as_tensor(shield.joints, dtype=ref.dtype, device=ref.device)
        joints[..., 0] += X_FILTERED
        motion.joints_pos = joints
        motion.joints_rot = torch.as_tensor(shield.rotations, dtype=ref.dtype, device=ref.device)
        motion.joints_local_rot = global_rots_to_local_rots(motion.joints_rot, motion.skeleton)
        motion.foot_contacts = None
        motion.precompute_mesh_info()
        motion.set_frame(session.frame_idx)
    ui.notify(app, session, "Runtime shield engaged",
              f"A person entered view at frame {frame}; from frame {shield.engage_frame} the CSF "
              "clip is redirected to a safe stand.", color="green", seconds=8.0)

"""Record the interactive demo's scenes: Kimodo's estimate at every step, with and without CSF.

For each scene, with and without a person in view, this runs the csf package exactly as
the demo app does (same rules, gate, safe-reference choice and seed) and records, at every
DDIM step, the clip Kimodo currently predicts in two runs from the same seed: one without
the filter, and one with CSF (the estimate after the filter, which the sampler continues
from). It also records each active rule's margin before and after the filter in the CSF
run. Estimates are decoded with Kimodo's motion representation and converted to G1 MuJoCo
qpos with Kimodo's own converter.

The unfiltered run's estimates come from a forward hook on the denoiser. When no rule is
active, the CSF run is Kimodo's plain sampler too, so its estimates come from the same hook.

Writes demo/data/scenes.json and demo/data/<scene>_<person|empty>.bin (float16).

  ~/twist2/csf/.venv/bin/python tools/record_scenes.py
"""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

CSF = Path.home() / "twist2" / "csf"
sys.path.insert(0, str(CSF))

from kimodo.exports.mujoco import MujocoQposConverter  # noqa: E402
from kimodo.tools import seed_everything  # noqa: E402

import csf.backbones.kimodo.filtered_generate as fg  # noqa: E402
from csf.app.entities import placement_from_motion  # noqa: E402
from csf.backbones import registry  # noqa: E402
from csf.backbones.base import ensure_text_encoder  # noqa: E402
from csf.backbones.kimodo.filtered_model import attach_kimodo_filter  # noqa: E402
from csf.filter.config import load_filter_config, load_safe_reference_policy  # noqa: E402
from csf.filter.qp import exclude_root  # noqa: E402
from csf.filter.references import net_root_speed, select_for_motion  # noqa: E402

SITE = Path(__file__).resolve().parents[1]
OUT = SITE / "demo" / "data"
STEPS = 100
# Stored DDIM steps (0-based). The filter does most of its work in the first few steps,
# after which the sampler stays on the safe side, so early steps are kept densely.
KEPT = [0, 1, 2, 3, 4, 5, 7, 9, 11, 14, 19, 24, 29, 39, 49, 59, 69, 79, 89, 99]
SECONDS = 4.0
K2M = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]).T

SCENES = [
    dict(id="highkick", prompt="a person walks forward and high kicks a ball", seed=0, entities=["a ball"],
         label="Walk forward and high kick a ball"),
    dict(id="runkick", prompt="a person runs forward and kicks a ball", seed=1, entities=["a ball"],
         label="Run forward and kick a ball"),
    dict(id="roundhouse", prompt="a person does a roundhouse kick", seed=0, entities=[],
         label="Roundhouse kick"),
    dict(id="punches", prompt="a person throws two punches", seed=0, entities=[],
         label="Throw two punches"),
    dict(id="kickperson", prompt="a person walks forward and kicks another person", seed=0, entities=["a ball"],
         label="Walk forward and kick another person"),
]


class Recorder:
    """Captures the filter's inputs and correction at each sampling step."""

    def __init__(self):
        self.steps = []
        self._orig = fg.filter_correction

    def __enter__(self):
        def wrapped(x_hat, x_safe, cfg, *, x_unsafe, active_mask=None, region=None):
            delta, info = self._orig(x_hat, x_safe, cfg, x_unsafe=x_unsafe,
                                     active_mask=active_mask, region=region)
            self.steps.append(dict(x_hat=x_hat.detach().clone(), delta=delta.detach().clone(),
                                   x_safe=x_safe.detach().clone(), x_unsafe=x_unsafe.detach().clone(),
                                   mask=list(active_mask) if active_mask is not None else None,
                                   root_dims=int(cfg.root_dims)))
            return delta, info
        fg.filter_correction = wrapped
        return self

    def __exit__(self, *exc):
        fg.filter_correction = self._orig


def margins(x, x_safe, x_unsafe, mask, rd):
    """h_i(x) for the active rules (Eq. 1, root channels excluded as in the filter)."""
    v = exclude_root(x - x_safe, rd)[0].reshape(-1)
    out = []
    for k, on in enumerate(mask):
        if on:
            d = exclude_root(x_unsafe[k] - x_safe, rd)[0].reshape(-1)
            out.append(float(-(v @ d) / (d.norm() + 1e-8)))
    return out


def to_qpos(model, conv, x):
    out = model.motion_rep.inverse(x, is_normalized=True, return_numpy=False)
    q = conv.dict_to_qpos({"local_rot_mats": out["local_rot_mats"], "root_positions": out["root_positions"]},
                          numpy=True)
    return np.asarray(q)[0]                       # [T, 36]


def kept(n):
    assert KEPT[-1] == n - 1
    return list(KEPT)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ctx = SimpleNamespace(device="cuda:0", text_encoder=None)
    ensure_text_encoder(ctx)
    bundle = registry.load_backbone("Kimodo-G1-RP-v1", ctx)
    model = bundle.model
    conv = MujocoQposConverter(bundle.skeleton)
    spec = registry.get_spec("Kimodo-G1-RP-v1")
    policy, threshold = load_safe_reference_policy()
    base = load_filter_config()
    cfg = base.with_backbone(**spec.filter.as_overrides(), stand_speed_threshold=threshold)
    rule_names = [r.name for r in base.rules]
    nf = int(SECONDS * bundle.model_fps)
    kw = dict(multi_prompt=True, constraint_lst=[], cfg_weight=[2.0, 2.0], num_samples=1,
              cfg_type="separated", num_transition_frames=5)
    meta = {"fps": float(bundle.model_fps), "frames": nf, "steps": STEPS, "keptSteps": kept(STEPS),
            "nq": 36, "rules": rule_names, "scenes": []}

    for scene in SCENES:
        entry = {k: scene[k] for k in ("id", "label", "prompt", "seed")}
        entry["variants"] = {}
        for person in (True, False):
            entities = (["a person"] if person else []) + scene["entities"]
            handle = attach_kimodo_filter(model, cfg, policy=policy, encoder=None)
            handle.set_entities(entities)
            prompt = scene["prompt"]

            unf_steps = []
            hook = model.denoiser.register_forward_hook(lambda _m, _i, out: unf_steps.append(out.detach().clone()))
            try:
                seed_everything(scene["seed"])
                unf = model([prompt], [nf], STEPS, **kw)
            finally:
                hook.remove()
            assert len(unf_steps) == STEPS, len(unf_steps)
            speed = net_root_speed(unf["smooth_root_pos"][0], bundle.model_fps)
            ref, _why = select_for_motion(handle.encode, prompt, speed, threshold, policy)
            handle.pin_safe_references({prompt: ref})

            hook_steps = []
            hook = model.denoiser.register_forward_hook(lambda _m, _i, out: hook_steps.append(out.detach().clone()))
            try:
                with Recorder() as rec:
                    seed_everything(scene["seed"])
                    with handle.filtering([prompt], [nf], STEPS):
                        seed_everything(scene["seed"])
                        csf_out = model([prompt], [nf], STEPS, **kw)
            finally:
                hook.remove()
            decision = handle.decisions[-1]
            handle.detach()

            if rec.steps:
                steps = rec.steps
                assert len(steps) == STEPS, len(steps)
                after = [s["x_hat"] + s["delta"] for s in steps]
                mask = steps[0]["mask"]
                m_before = [margins(s["x_hat"], s["x_safe"], s["x_unsafe"], mask, s["root_dims"]) for s in steps]
                m_after = [margins(s["x_hat"] + s["delta"], s["x_safe"], s["x_unsafe"], mask, s["root_dims"])
                           for s in steps]
            else:
                assert len(hook_steps) == STEPS, len(hook_steps)
                after = hook_steps
                mask = decision.mask
                m_before = m_after = [[] for _ in range(STEPS)]

            idx = kept(STEPS)
            q_unf_steps = np.stack([to_qpos(model, conv, unf_steps[i]) for i in idx])
            q_after = np.stack([to_qpos(model, conv, after[i]) for i in idx])
            # Final clips come straight from the generators' outputs.
            q_unf = np.asarray(conv.dict_to_qpos({"local_rot_mats": unf["local_rot_mats"],
                                                  "root_positions": unf["root_positions"]}, numpy=True))[0]
            q_csf = np.asarray(conv.dict_to_qpos({"local_rot_mats": csf_out["local_rot_mats"],
                                                  "root_positions": csf_out["root_positions"]}, numpy=True))[0]
            # The last estimate of each run must be the clip that run returns.
            final_gap = max(float(np.abs(q_after[-1][:, 7:] - q_csf[:, 7:]).max()),
                            float(np.abs(q_unf_steps[-1][:, 7:] - q_unf[:, 7:]).max()))

            place = placement_from_motion(unf["posed_joints"][0].detach().cpu().numpy())
            px, py, pz = K2M @ np.array([place.x, 0.0, place.z])
            fx, fy, fz = K2M @ np.array([place.facing[0], 0.0, place.facing[1]])

            name = f"{scene['id']}_{'person' if person else 'empty'}"
            arrays = [q_unf_steps, q_after]
            blob = b"".join(a.astype(np.float16).tobytes() for a in arrays)
            (OUT / f"{name}.bin").write_bytes(blob)
            active = rule_names if mask is None else [rule_names[k] for k, on in enumerate(mask) if on]
            entry["variants"]["person" if person else "empty"] = {
                "file": f"{name}.bin",
                "safeReference": ref,
                "activeRules": active,
                "signals": sorted({s for rs in (decision.reasons or []) for s in rs}),
                "marginsBefore": [m_before[i] for i in idx],
                "marginsAfter": [m_after[i] for i in idx],
                "person": {"pos": [float(px), float(py)], "facing": [float(fx), float(fy)]},
            }
            print(f"{name}: active={active} ref={ref!r} final gap {final_gap:.2e} rad "
                  f"bytes={len(blob)}", flush=True)
        meta["scenes"].append(entry)
    (OUT / "scenes.json").write_text(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()

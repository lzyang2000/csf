# SPDX-License-Identifier: Apache-2.0
"""Kimodo's step-matched filtered sampling loop (Sec. III-E, direct motion space).

Kimodo predicts the clean motion feature at every DDIM step, so CSF constructs
step-matched references: at each step the frozen denoiser also predicts, from
the same noisy sample and without classifier-free guidance, the clean motion
for every rule's unsafe description and for the safe reference.  Eq. 3 then
filters the guided estimate before the native sampler update.

Two further details of the motion-space path:

* Root channels are excluded from the margin and the correction (the
  projector M of Eq. 6), and the filtered sample's root path is pinned to the
  unfiltered trajectory, so the filter must change the pose rather than move
  the character away from the protected entity.
* The safe reference is chosen per frame: frames where the unfiltered clip is
  still and upright track the stand, moving frames track the segment's
  selected safe reference (walk, jog, ...).
"""
from __future__ import annotations

import numpy as np
import torch

from csf.filter.qp import filter_correction


def root_path_dims(motion_rep) -> list[int]:
    """Channels of the root path (smooth root position + heading) in Kimodo's layout."""
    sizes = getattr(motion_rep, "size_dict", None)
    if not sizes:
        return []
    offset, cursor = {}, 0
    for key, size in sizes.items():
        offset[key] = cursor
        cursor += int(np.prod(tuple(size)))
    return list(range(0, offset.get("local_joints_positions", 5)))


def _moving_frames(model, x_unfiltered, fps: float, window: int, threshold: float) -> np.ndarray:
    """1 where the unfiltered clip moves (or is not upright), 0 where it stands still."""
    from kimodo.tools import to_numpy  # noqa: PLC0415

    joints = to_numpy(model.motion_rep.inverse(x_unfiltered, is_normalized=True,
                                               return_numpy=False))["posed_joints"]
    joints = np.asarray(joints[0] if getattr(joints, "ndim", 0) == 4 else joints)
    root = model.skeleton.root_idx
    nf = joints.shape[0]
    speed = np.zeros(nf, dtype=np.float32)
    speed[1:] = np.linalg.norm(np.diff(joints[:, root][:, [0, 2]], axis=0), axis=-1) * fps
    k = max(1, window // 2)
    speed = np.convolve(speed, np.ones(k, dtype=np.float32) / k, mode="same")
    names = list(getattr(model.skeleton, "bone_order_names", []))
    try:
        shoulders = (joints[:, names.index("left_shoulder_pitch_skel")]
                     + joints[:, names.index("right_shoulder_pitch_skel")]) / 2.0
        torso = shoulders - joints[:, root]
        torso_up = torso[:, 1] / (np.linalg.norm(torso, axis=-1) + 1e-9)
    except ValueError:
        torso_up = np.ones(nf, dtype=np.float32)
    upright = ~((torso_up < 0.5) & (joints[:, root, 1] < 0.4))
    return ((speed > threshold) | ~upright).astype(np.float32)


def generate_filtered_segment(
    model,
    cfg,
    unsafe_feats,
    safe_feat,
    stand_feat,
    text_feat,
    text_pad_mask,
    pad_mask,
    motion_mask,
    observed_motion,
    first_heading_angle,
    cfg_weight,
    cfg_type,
    nf: int,
    steps: int,
    lock_head: int = 0,
    active_mask=None,
    window: int = 20,
):
    """Generate one prompt segment with CSF.

    Args:
        model: The loaded Kimodo model.
        cfg: `FilterConfig` (gamma, rho, ridge, root_dims, pin_dims,
            stand_speed_threshold).
        unsafe_feats: [K, 1, d] text features of the rules' unsafe descriptions.
        safe_feat: [1, 1, d] text feature of the segment's safe reference.
        stand_feat: [1, 1, d] text feature of the stand.
        text_feat, text_pad_mask, pad_mask, motion_mask, observed_motion,
        first_heading_angle, cfg_weight, cfg_type: The segment's own generation
            context, as Kimodo's `_generate` receives it.
        nf, steps: Frames and denoising steps.
        lock_head: Leading transition frames that are never filtered.
        active_mask: The gate's active rules.
        window: Smoothing window (frames) for the per-frame motion evidence.

    Returns:
        ``(x, info)``: the filtered normalised motion [1, nf, D] and a summary.
    """
    from kimodo.motion_rep.feature_utils import length_to_mask  # noqa: PLC0415

    dev = model.device
    D = model.motion_rep.motion_rep_dim
    K = unsafe_feats.shape[0]
    ut, mapt = model.diffusion.space_timesteps(steps)
    model.diffusion.calc_diffusion_vars(ut)

    pad_mask = pad_mask.to(dev) if pad_mask is not None else None
    text_pad_mask = text_pad_mask.to(dev) if text_pad_mask is not None else None
    text_feat = text_feat.to(dev) if text_feat is not None else None
    heading = (first_heading_angle.to(dev) if first_heading_angle is not None
               else torch.zeros(1, device=dev))
    motion_mask = motion_mask.to(dev) if motion_mask is not None else None
    observed_motion = observed_motion.to(dev) if observed_motion is not None else None

    # Unfiltered pass from the same noise: its trajectory pins the root path and
    # supplies the per-frame motion evidence.
    noise = torch.randn(1, nf, D, device=dev)
    x = noise.clone()
    trajectory = []
    for i in reversed(range(steps)):
        t = torch.tensor([i], device=dev)
        with torch.inference_mode():
            x_hat = model.denoiser(cfg_weight, x, pad_mask, text_feat, text_pad_mask,
                                   mapt[t], heading, motion_mask, observed_motion,
                                   cfg_type=cfg_type)
        x = model.sampler(ut, x, x_hat, t)
        trajectory.append(x.clone())

    moving = _moving_frames(model, trajectory[-1], float(model.fps), window,
                            float(cfg.stand_speed_threshold))
    moving = torch.tensor(moving, device=dev).view(1, nf, 1)

    frame_mask = torch.ones(1, nf, 1, device=dev)
    if lock_head > 0:
        frame_mask[:, :lock_head] = 0.0
    region = (min(lock_head, nf), nf)

    # Reference batch: [prompt, unsafe_1..K, safe, stand], no guidance.
    null = torch.zeros(1, 1, text_feat.shape[-1], device=dev, dtype=text_feat.dtype)
    feats = torch.cat([text_feat, unsafe_feats, safe_feat, stand_feat, null], 0)
    G = feats.shape[0]
    feat_masks = torch.cat([torch.ones(G - 1, 1, dtype=torch.bool, device=dev),
                            torch.zeros(1, 1, dtype=torch.bool, device=dev)], 0)
    ref_pad = length_to_mask(torch.full((G,), nf, device=dev))
    ref_heading = torch.zeros(G, device=dev)

    pin = list(cfg.pin_dims)
    corrected_steps = 0
    x = noise.clone()
    for j, i in enumerate(reversed(range(steps))):
        t = torch.tensor([i], device=dev)
        with torch.inference_mode():
            x_hat = model.denoiser(cfg_weight, x, pad_mask, text_feat, text_pad_mask,
                                   mapt[t], heading, motion_mask, observed_motion,
                                   cfg_type=cfg_type)
            refs = model.denoiser([2.0, 2.0], x.expand(G, -1, -1).contiguous(), ref_pad,
                                  feats, feat_masks, mapt[t.expand(G)], ref_heading,
                                  None, None, cfg_type="nocfg")
        x_unsafe = refs[1:1 + K]
        x_safe = moving * refs[1 + K:2 + K] + (1.0 - moving) * refs[2 + K:3 + K]
        delta, info = filter_correction(x_hat, x_safe, cfg, x_unsafe=x_unsafe.unsqueeze(1),
                                        active_mask=active_mask, region=region)
        corrected_steps += int(info["active"])
        x = model.sampler(ut, x, x_hat + delta * frame_mask, t)
        x = frame_mask * x + (1.0 - frame_mask) * trajectory[j]
        if pin:
            x[..., pin] = trajectory[j][..., pin]

    return x, {"corrected_steps": corrected_steps, "steps": steps}

# SPDX-License-Identifier: Apache-2.0
"""The runtime safety shield (Sec. III-F, Eq. 10).

Generation-time filtering cannot help a motion that is already executing when
the scene changes.  The shield watches the remaining reference R: once the
live context activates a rule, it finds the first unexecuted window whose
margin falls below the calibrated threshold tau,

    j* = min{ j : s_j >= l,  min_{i in A_l} h_i(R[W_j]) < tau },

and replaces R from the splice frame e = min(l + b, T - 1) on with a safe
continuation S, aligned to the held state in heading and position and blended
in over a fixed transition (B_e).  Frames before e are never touched.

Margins here are measured on root-relative joint positions against reference
clips generated once per backbone (one per person-directed rule, plus a stand
that serves as x_safe and as S).  tau is calibrated from what genuinely safe
clips score: an independent stand draw and a neutral walk, minus a 25 % band.
Once engaged, the shield stays engaged for the rest of the clip.

The motion format is the generators' common output: global joint positions
[T, J, 3] and global joint rotations [T, J, 3, 3], Y up, root joint first.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

_EPS = 1e-9


def root_relative(joints: np.ndarray) -> np.ndarray:
    """[T, J, 3] joint positions -> [T, J*3] positions relative to the root joint."""
    joints = np.asarray(joints, dtype=np.float64)
    rel = joints - joints[:, :1, :]
    return rel.reshape(rel.shape[0], -1)


def resample_rows(values: np.ndarray, num_rows: int) -> np.ndarray:
    """Linearly resample [T, D] onto `num_rows` frames, channel-wise."""
    if values.shape[0] == num_rows:
        return values
    src = np.linspace(0.0, 1.0, values.shape[0])
    dst = np.linspace(0.0, 1.0, num_rows)
    return np.stack([np.interp(dst, src, values[:, c]) for c in range(values.shape[1])], axis=1)


def windowed_margins(feat, unsafe_feats, safe_feat, window: int, stride: int | None = None):
    """Per-window margin, minimised over rules.

    Args:
        feat: [T, D] features of the reference being executed.
        unsafe_feats: [K, T, D] unsafe reference features, one per rule.
        safe_feat: [T, D] safe reference features.
        window: Window length W in frames; `stride` defaults to W // 2.

    Returns:
        ``(starts, margins)``: window start frames s_j and min_i h_i(R[W_j]).
    """
    T = feat.shape[0]
    stride = max(1, window // 2 if stride is None else stride)
    starts, margins = [], []
    for a in range(0, max(1, T - window + 1), stride):
        b = min(T, a + window)
        v = (feat[a:b] - safe_feat[a:b]).reshape(-1)
        worst = None
        for k in range(unsafe_feats.shape[0]):
            d = (unsafe_feats[k, a:b] - safe_feat[a:b]).reshape(-1)
            m = float(-(v @ d) / (np.linalg.norm(d) + _EPS))
            worst = m if worst is None else min(worst, m)
        starts.append(a)
        margins.append(0.0 if worst is None else worst)
    return np.asarray(starts), np.asarray(margins)


def first_violation(starts, margins, cursor: int, tau: float) -> Optional[int]:
    """j*: the start of the first window at or after `cursor` with margin < tau."""
    for a, m in zip(starts, margins):
        if a >= cursor and m < tau:
            return int(a)
    return None


def calibrate_threshold(safe_draw_margins, walk_margins=None, band: float = 0.25) -> float:
    """tau from what genuinely safe clips score, minus a relative band."""
    floor = float(np.min(safe_draw_margins))
    if walk_margins is not None and len(walk_margins):
        floor = min(floor, float(np.min(walk_margins)))
    return floor - band * abs(floor) - 1e-6


def _yaw_between(R_from: np.ndarray, R_to: np.ndarray) -> np.ndarray:
    """The rotation about +Y that best maps orientation `R_from` onto `R_to`."""
    M = R_from @ R_to.T
    theta = np.arctan2(M[2, 0] - M[0, 2], M[0, 0] + M[2, 2])
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _project_rotation(M: np.ndarray) -> np.ndarray:
    """Nearest rotation matrix (per [..., 3, 3]) via SVD."""
    U, _S, Vt = np.linalg.svd(M)
    R = U @ Vt
    flip = np.linalg.det(R) < 0
    if np.any(flip):
        U[flip, :, -1] *= -1
        R = U @ Vt
    return R


def splice(joints, rotations, safe_joints, safe_rotations, e: int, n_trans: int):
    """B_e: replace frames [e, T) with the aligned safe continuation.

    The safe clip is turned about +Y and shifted on the ground plane so its
    first frame sits at the held pose (frame e - 1); its own height is kept.
    The first `n_trans` frames blend from the held pose into it (smoothstep).

    Returns:
        New ``(joints [T, J, 3], rotations [T, J, 3, 3])`` arrays.
    """
    joints = np.array(joints, dtype=np.float64, copy=True)
    rotations = np.array(rotations, dtype=np.float64, copy=True)
    T = joints.shape[0]
    if e >= T - 1:
        return joints, rotations
    hold = max(e - 1, 0)
    hold_j, hold_r = joints[hold].copy(), rotations[hold].copy()

    safe_j = np.asarray(safe_joints, dtype=np.float64)
    safe_r = np.asarray(safe_rotations, dtype=np.float64)
    Ry = _yaw_between(safe_r[0, 0], hold_r[0])
    origin = safe_j[0, 0].copy()
    aligned_j = (safe_j - origin) @ Ry.T
    aligned_j[..., 0] += hold_j[0, 0]
    aligned_j[..., 2] += hold_j[0, 2]
    aligned_j[..., 1] += origin[1]
    aligned_r = Ry[None, None] @ safe_r

    S = aligned_j.shape[0]
    for i, f in enumerate(range(e, T)):
        si = min(i, S - 1)
        if i < n_trans:
            w = 0.5 - 0.5 * np.cos(np.pi * (i + 1) / (n_trans + 1))
            joints[f] = (1 - w) * hold_j + w * aligned_j[si]
            rotations[f] = _project_rotation((1 - w) * hold_r + w * aligned_r[si])
        else:
            joints[f] = aligned_j[si]
            rotations[f] = aligned_r[si]
    return joints, rotations


@dataclass
class ShieldReferences:
    """Per-backbone reference features for the shield, generated once.

    Attributes:
        unsafe: [K, T, D] root-relative features, one per person-directed rule.
        safe: [T, D] features of the stand (x_safe).
        safe_draw: [T, D] features of an independent stand draw (calibration).
        walk: [T, D] features of a neutral walk (calibration).
        stand_joints / stand_rotations: The stand clip itself, used as S.
    """

    unsafe: np.ndarray
    safe: np.ndarray
    safe_draw: np.ndarray
    walk: np.ndarray
    stand_joints: np.ndarray
    stand_rotations: np.ndarray


@dataclass
class RuntimeShield:
    """A latching shield over one executing reference.

    Build it with `RuntimeShield.arm`, then call `step(cursor, context_active)`
    as execution advances.  The reference arrays are replaced (not mutated) on
    engagement; read them back from `joints` / `rotations`.
    """

    joints: np.ndarray
    rotations: np.ndarray
    starts: np.ndarray
    margins: np.ndarray
    tau: float
    safe_joints: np.ndarray
    safe_rotations: np.ndarray
    buffer: int = 2
    n_trans: int = 9
    engaged: bool = field(default=False, init=False)
    engage_frame: Optional[int] = field(default=None, init=False)
    first_unsafe_window: Optional[int] = field(default=None, init=False)

    @classmethod
    def arm(cls, joints, rotations, refs: ShieldReferences, fps: float,
            buffer_s: float = 0.06, transition_s: float = 0.3) -> "RuntimeShield":
        """Compute the reference's windowed margins and the threshold tau."""
        feat = root_relative(joints)
        T = feat.shape[0]
        window = max(4, int(fps))
        unsafe = np.stack([resample_rows(u, T) for u in refs.unsafe])
        safe = resample_rows(refs.safe, T)
        starts, margins = windowed_margins(feat, unsafe, safe, window)
        _s, draw = windowed_margins(resample_rows(refs.safe_draw, T), unsafe, safe, window)
        _w, walk = windowed_margins(resample_rows(refs.walk, T), unsafe, safe, window)
        return cls(
            joints=np.asarray(joints, dtype=np.float64),
            rotations=np.asarray(rotations, dtype=np.float64),
            starts=starts,
            margins=margins,
            tau=calibrate_threshold(draw, walk),
            safe_joints=refs.stand_joints,
            safe_rotations=refs.stand_rotations,
            buffer=max(1, int(round(buffer_s * fps))),
            n_trans=max(1, int(round(transition_s * fps))),
        )

    def step(self, cursor: int, context_active: bool) -> bool:
        """Evaluate at execution cursor l.  Returns True if it engaged now."""
        if self.engaged or not context_active:
            return False
        fire = first_violation(self.starts, self.margins, int(cursor), self.tau)
        if fire is None:
            return False
        e = min(int(cursor) + self.buffer, self.joints.shape[0] - 1)
        self.joints, self.rotations = splice(
            self.joints, self.rotations, self.safe_joints, self.safe_rotations, e, self.n_trans
        )
        self.engaged = True
        self.engage_frame = e
        self.first_unsafe_window = fire
        return True

# SPDX-License-Identifier: Apache-2.0
"""The runtime shield (Sec. III-F, Eq. 10) on synthetic clips."""
import numpy as np
import pytest

from csf.filter.shield import (
    RuntimeShield,
    ShieldReferences,
    calibrate_threshold,
    first_violation,
    resample_rows,
    root_relative,
    splice,
    windowed_margins,
)


def _yaw(theta: float) -> np.ndarray:
    """Rotation by `theta` about +Y."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _random_rotations(rng, shape) -> np.ndarray:
    q, r = np.linalg.qr(rng.normal(size=(*shape, 3, 3)))
    q = q * np.sign(np.diagonal(r, axis1=-2, axis2=-1))[..., None, :]
    q[np.linalg.det(q) < 0, :, 0] *= -1
    return q


def _clip(rng, frames: int, joints: int, root_start, root_step, heading: float):
    """A clip whose root starts at `root_start`, advances `root_step` per frame and faces `heading`."""
    root = np.asarray(root_start, float) + np.arange(frames)[:, None] * np.asarray(root_step, float)
    positions = root[:, None, :] + np.concatenate(
        [np.zeros((1, 3)), rng.normal(scale=0.3, size=(joints - 1, 3))])[None]
    rotations = _random_rotations(rng, (frames, joints))
    rotations[:, 0] = _yaw(heading)
    return positions, rotations


def _orthonormal(rotations) -> bool:
    eye = np.eye(3)
    gram = rotations @ np.swapaxes(rotations, -1, -2)
    return np.allclose(gram, eye, atol=1e-8) and np.allclose(np.linalg.det(rotations), 1.0, atol=1e-8)


# ----------------------------------------------------------------- features
def test_root_relative_subtracts_the_root_and_flattens():
    joints = np.zeros((5, 3, 3))
    joints[:, 0] = 7.0
    joints[:, 1] = 8.0
    features = root_relative(joints)
    assert features.shape == (5, 9)
    assert np.allclose(features[:, :3], 0.0)
    assert np.allclose(features[:, 3:6], 1.0)
    assert np.allclose(features[:, 6:], -7.0)


def test_resample_rows():
    values = np.array([[0.0, 10.0], [1.0, 20.0], [2.0, 30.0]])
    assert resample_rows(values, 3) is values
    resampled = resample_rows(values, 5)
    assert np.allclose(resampled[:, 0], [0.0, 0.5, 1.0, 1.5, 2.0])
    assert np.allclose(resampled[:, 1], [10.0, 15.0, 20.0, 25.0, 30.0])


# --------------------------------------------------------- windowed margins
def test_windowed_margin_closed_form():
    """Safe at 0, unsafe at 1: a constant clip at level c scores -c sqrt(W D) in every window."""
    T, D, W, level = 20, 3, 6, 1.5
    starts, margins = windowed_margins(np.full((T, D), level), np.ones((1, T, D)),
                                       np.zeros((T, D)), W)
    assert list(starts) == list(range(0, T - W + 1, W // 2))
    assert np.allclose(margins, -level * np.sqrt(W * D))


def test_windowed_margin_takes_the_worst_rule():
    T, D, W = 12, 2, 4
    unsafe = np.stack([np.ones((T, D)), -np.ones((T, D))])
    _, margins = windowed_margins(np.full((T, D), 0.5), unsafe, np.zeros((T, D)), W)
    assert np.allclose(margins, -0.5 * np.sqrt(W * D))


def test_windowed_margin_is_low_only_where_the_clip_turns_unsafe():
    rng = np.random.default_rng(0)
    T, D, W = 100, 8, 20
    safe, unsafe = rng.normal(size=(T, D)), rng.normal(size=(T, D))
    feat = safe.copy()
    feat[50:] = unsafe[50:]
    starts, margins = windowed_margins(feat, unsafe[None], safe, W)
    assert np.allclose(margins[starts + W <= 50], 0.0)
    assert (margins[starts >= 50] < -5.0).all()


def test_windowed_margin_stride_and_short_clips():
    feat, unsafe, safe = np.ones((10, 2)), np.ones((1, 10, 2)), np.zeros((10, 2))
    assert list(windowed_margins(feat, unsafe, safe, 4, stride=3)[0]) == [0, 3, 6]
    starts, margins = windowed_margins(feat[:3], unsafe[:, :3], safe[:3], 8)
    assert list(starts) == [0] and margins[0] == pytest.approx(-np.sqrt(6))


# --------------------------------------------------------- first violation
def test_first_violation_respects_the_cursor():
    starts = np.array([0, 10, 20, 30, 40])
    margins = np.array([-5.0, 0.0, -5.0, 0.0, -5.0])
    assert first_violation(starts, margins, cursor=0, tau=-1.0) == 0
    assert first_violation(starts, margins, cursor=1, tau=-1.0) == 20, "executed windows are ignored"
    assert first_violation(starts, margins, cursor=20, tau=-1.0) == 20
    assert first_violation(starts, margins, cursor=41, tau=-1.0) is None


def test_first_violation_is_strict():
    assert first_violation([0, 5], [-1.0, -1.0], cursor=0, tau=-1.0) is None


# -------------------------------------------------------------- calibration
def test_threshold_is_the_safe_floor_minus_a_relative_band():
    assert calibrate_threshold([0.4, -2.0, 1.0]) == pytest.approx(-2.0 - 0.5 - 1e-6)
    assert calibrate_threshold([1.0, 2.0]) == pytest.approx(1.0 - 0.25 - 1e-6)


def test_threshold_uses_the_lower_of_both_safe_clips():
    assert calibrate_threshold([-1.0], [-3.0, 0.0]) == pytest.approx(-3.75 - 1e-6)
    assert calibrate_threshold([-1.0], []) == pytest.approx(-1.25 - 1e-6)
    assert calibrate_threshold([-2.0], band=0.5) == pytest.approx(-3.0 - 1e-6)


# ------------------------------------------------------------------- splice
T, J, S, E, N_TRANS = 40, 5, 25, 12, 5
HEADING, SAFE_HEADING = 0.7, -1.2


@pytest.fixture()
def clips():
    rng = np.random.default_rng(1)
    reference = _clip(rng, T, J, root_start=(0.0, 0.8, 0.0), root_step=(0.02, 0.0, 0.01),
                      heading=HEADING)
    stand = _clip(rng, S, J, root_start=(5.0, 0.75, -3.0), root_step=(0.0, 0.0, 0.0),
                  heading=SAFE_HEADING)
    return reference, stand


def test_splice_leaves_the_executed_frames_untouched(clips):
    (joints, rotations), (safe_j, safe_r) = clips
    out_j, out_r = splice(joints, rotations, safe_j, safe_r, E, N_TRANS)
    assert np.array_equal(out_j[:E], joints[:E])
    assert np.array_equal(out_r[:E], rotations[:E])


def test_splice_lands_on_the_yaw_aligned_safe_clip(clips):
    """After the transition: the safe clip turned about +Y onto the held heading, root xz at the hold."""
    (joints, rotations), (safe_j, safe_r) = clips
    out_j, out_r = splice(joints, rotations, safe_j, safe_r, E, N_TRANS)
    turn = _yaw(HEADING - SAFE_HEADING)
    hold_root = joints[E - 1, 0]
    offset = np.array([hold_root[0], safe_j[0, 0, 1], hold_root[2]])
    for f in range(E + N_TRANS, T):
        si = min(f - E, S - 1)
        assert np.allclose(out_j[f], (safe_j[si] - safe_j[0, 0]) @ turn.T + offset, atol=1e-9)
        assert np.allclose(out_r[f], turn @ safe_r[si], atol=1e-9)


def test_spliced_root_holds_position_and_heading(clips):
    (joints, rotations), (safe_j, safe_r) = clips
    out_j, out_r = splice(joints, rotations, safe_j, safe_r, E, N_TRANS)
    tail = slice(E + N_TRANS, T)
    assert np.allclose(out_j[tail, 0, [0, 2]], joints[E - 1, 0, [0, 2]], atol=1e-9)
    assert np.allclose(out_j[tail, 0, 1], safe_j[0, 0, 1], atol=1e-9), "the safe clip keeps its height"
    assert np.allclose(out_r[tail, 0], _yaw(HEADING), atol=1e-9)


def test_short_safe_clip_holds_its_last_frame(clips):
    (joints, rotations), (safe_j, safe_r) = clips
    out_j, out_r = splice(joints, rotations, safe_j, safe_r, E, N_TRANS)
    assert E + S < T
    assert np.allclose(out_j[E + S - 1:], out_j[E + S - 1])
    assert np.allclose(out_r[E + S - 1:], out_r[E + S - 1])


def test_spliced_rotations_stay_orthonormal(clips):
    (joints, rotations), (safe_j, safe_r) = clips
    _, out_r = splice(joints, rotations, safe_j, safe_r, E, N_TRANS)
    assert _orthonormal(out_r)


def test_transition_blends_smoothly_from_the_held_pose(clips):
    (joints, rotations), (safe_j, safe_r) = clips
    still_safe_j = np.repeat(safe_j[:1], S, axis=0)
    still_safe_r = np.repeat(safe_r[:1], S, axis=0)
    out_j, _ = splice(joints, rotations, still_safe_j, still_safe_r, E, N_TRANS)
    target, hold = out_j[E + N_TRANS], joints[E - 1]
    gap = np.linalg.norm(target - hold)
    distances = [np.linalg.norm(out_j[f] - target) for f in range(E - 1, E + N_TRANS + 1)]
    assert np.all(np.diff(distances) < 0), "the blend approaches the safe pose monotonically"
    assert np.linalg.norm(out_j[E] - hold) < 0.2 * gap, "no jump at the splice frame"


def test_splice_at_the_last_frame_changes_nothing(clips):
    (joints, rotations), (safe_j, safe_r) = clips
    out_j, out_r = splice(joints, rotations, safe_j, safe_r, T - 1, N_TRANS)
    assert np.array_equal(out_j, joints) and np.array_equal(out_r, rotations)
    assert out_j is not joints


def test_splice_does_not_modify_its_inputs(clips):
    (joints, rotations), (safe_j, safe_r) = clips
    before = [a.copy() for a in (joints, rotations, safe_j, safe_r)]
    splice(joints, rotations, safe_j, safe_r, E, N_TRANS)
    assert all(np.array_equal(a, b) for a, b in zip((joints, rotations, safe_j, safe_r), before))


# ------------------------------------------------------------ RuntimeShield
def _shield(clips, margins, tau=-1.0, starts=None):
    (joints, rotations), (safe_j, safe_r) = clips
    starts = np.arange(0, T - 8 + 1, 4) if starts is None else np.asarray(starts)
    return RuntimeShield(joints=joints, rotations=rotations, starts=starts,
                         margins=np.asarray(margins(starts), float), tau=tau,
                         safe_joints=safe_j, safe_rotations=safe_r, buffer=3, n_trans=4)


def _unsafe_from(frame):
    return lambda starts: np.where(starts >= frame, -5.0, 0.0)


def test_shield_does_nothing_while_the_context_is_inactive(clips):
    shield = _shield(clips, _unsafe_from(20))
    joints = shield.joints
    for cursor in range(T):
        assert shield.step(cursor, context_active=False) is False
    assert not shield.engaged and shield.joints is joints


def test_shield_engages_ahead_of_the_first_unsafe_window(clips):
    shield = _shield(clips, _unsafe_from(20))
    original = shield.joints.copy()
    assert shield.step(10, context_active=True) is True
    assert shield.engaged
    assert shield.engage_frame == 13
    assert shield.first_unsafe_window == 20
    assert np.array_equal(shield.joints[:13], original[:13])
    assert not np.allclose(shield.joints[13:], original[13:])
    assert _orthonormal(shield.rotations)


def test_shield_latches(clips):
    shield = _shield(clips, _unsafe_from(20))
    shield.step(10, context_active=True)
    joints, rotations = shield.joints, shield.rotations
    assert shield.step(11, context_active=True) is False
    assert shield.step(12, context_active=False) is False
    assert shield.engaged and shield.engage_frame == 13
    assert shield.joints is joints and shield.rotations is rotations


def test_shield_ignores_a_clean_reference(clips):
    shield = _shield(clips, lambda starts: np.zeros(len(starts)))
    assert not any(shield.step(c, context_active=True) for c in range(T))
    assert not shield.engaged


def test_shield_ignores_violations_already_executed(clips):
    shield = _shield(clips, lambda starts: np.where(starts == 4, -5.0, 0.0))
    assert shield.step(5, context_active=True) is False
    assert not shield.engaged


def test_splice_frame_is_clamped_to_the_clip(clips):
    shield = _shield(clips, _unsafe_from(38), starts=[0, 38])
    assert shield.step(38, context_active=True) is True
    assert shield.engage_frame == T - 1


def _pose_clip(frames, pose, heading=0.0, step=0.02):
    """A clip holding `pose` (root-relative [J, 3]) while its root advances `step` per frame along x."""
    root = np.zeros((frames, 3))
    root[:, 0] = step * np.arange(frames)
    root[:, 1] = 0.8
    joints = root[:, None, :] + pose[None]
    rotations = np.repeat(_yaw(heading)[None, None], frames, axis=0).repeat(pose.shape[0], axis=1)
    return joints, rotations


@pytest.fixture()
def armed():
    """References for an 80-frame, 20 fps execution; a benign clip; and a clip unsafe from frame 40."""
    rng = np.random.default_rng(2)
    safe_pose, unsafe_pose = (
        np.concatenate([np.zeros((1, 3)), rng.normal(scale=0.3, size=(J - 1, 3))]) for _ in range(2))
    frames, ref_frames = 80, 60              # references are resampled to the executed length

    def features(pose, noise=0.0):
        return root_relative(_pose_clip(ref_frames, pose)[0]) + noise * rng.normal(size=(ref_frames, J * 3))

    stand_joints, stand_rotations = _pose_clip(ref_frames, safe_pose, step=0.0)
    refs = ShieldReferences(
        unsafe=features(unsafe_pose)[None], safe=features(safe_pose),
        safe_draw=features(safe_pose, 0.02), walk=features(safe_pose, 0.03),
        stand_joints=stand_joints, stand_rotations=stand_rotations,
    )
    benign_joints, rotations = _pose_clip(frames, safe_pose)
    unsafe_joints = benign_joints.copy()
    unsafe_joints[40:] = _pose_clip(frames, unsafe_pose)[0][40:]
    return {"refs": refs, "fps": 20.0, "rotations": rotations,
            "benign": benign_joints, "unsafe": unsafe_joints}


def test_arm_calibrates_tau_and_timing_from_the_references(armed):
    refs, joints = armed["refs"], armed["unsafe"]
    shield = RuntimeShield.arm(joints, armed["rotations"], refs, armed["fps"])
    frames, window = joints.shape[0], 20
    unsafe = np.stack([resample_rows(u, frames) for u in refs.unsafe])
    safe = resample_rows(refs.safe, frames)
    _, draw = windowed_margins(resample_rows(refs.safe_draw, frames), unsafe, safe, window)
    _, walk = windowed_margins(resample_rows(refs.walk, frames), unsafe, safe, window)
    assert shield.tau == pytest.approx(calibrate_threshold(draw, walk))
    assert shield.tau < 0
    assert shield.buffer == 1 and shield.n_trans == 6
    assert list(shield.starts) == list(range(0, frames - window + 1, window // 2))


def test_armed_shield_intervenes_only_on_the_unsafe_clip(armed):
    refs, fps, rotations = armed["refs"], armed["fps"], armed["rotations"]
    benign = RuntimeShield.arm(armed["benign"], rotations, refs, fps)
    assert not any(benign.step(c, context_active=True) for c in range(80))

    joints = armed["unsafe"]
    shield = RuntimeShield.arm(joints, rotations, refs, fps)
    assert shield.step(25, context_active=True) is True
    assert shield.first_unsafe_window == 30
    assert np.array_equal(shield.joints[:26], joints[:26])
    tail = shield.joints[26 + shield.n_trans:]
    assert np.allclose(root_relative(tail), refs.safe[0], atol=1e-9), "the tail holds the stand pose"
    assert np.allclose(tail[:, 0, [0, 2]], joints[25, 0, [0, 2]], atol=1e-9)

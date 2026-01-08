import pytest


def test_train_preflight_rejects_both_motion_sources():
  from tracker.cli.train import resolve_train_mode

  with pytest.raises(ValueError):
    resolve_train_mode(
      "Tracker-Tracking-Flat-Adam-SP-23",
      ["--motion-file", "a.npz", "--registry-name", "x"],
    )


def test_train_preflight_accepts_offline_mode(tmp_path):
  from tracker.integrations.mjlab.bootstrap import bootstrap
  from tracker.cli.train import resolve_train_mode

  bootstrap()

  motion_path = tmp_path / "motion.npz"
  from tracker.robots.adam_sp import TRACKING_BODY_NAMES, validate_motion_npz
  from tracker.robots.adam_sp_constants import get_spec
  import numpy as np

  spec = get_spec()
  expected_joints = sum(1 for j in spec.joints if j.type != 0)  # non-free joints
  expected_bodies = len(TRACKING_BODY_NAMES)

  t = 1
  np.savez(
    motion_path,
    joint_pos=np.zeros((t, expected_joints), dtype=np.float32),
    joint_vel=np.zeros((t, expected_joints), dtype=np.float32),
    body_pos_w=np.zeros((t, expected_bodies, 3), dtype=np.float32),
    body_quat_w=np.zeros((t, expected_bodies, 4), dtype=np.float32),
    body_lin_vel_w=np.zeros((t, expected_bodies, 3), dtype=np.float32),
    body_ang_vel_w=np.zeros((t, expected_bodies, 3), dtype=np.float32),
  )
  validate_motion_npz(motion_path)

  mode, remaining = resolve_train_mode(
    "Tracker-Tracking-Flat-Adam-SP-23", ["--motion-file", str(motion_path)]
  )
  assert mode == "offline"
  assert remaining == []

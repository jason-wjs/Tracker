import numpy as np


def _write_minimal_adam_sp_motion_npz(path):
  from tracker.robots.adam_sp import TRACKING_BODY_NAMES, validate_motion_npz
  from tracker.robots.adam_sp_constants import get_spec

  spec = get_spec()
  expected_joints = sum(1 for j in spec.joints if j.type != 0)  # non-free joints
  expected_bodies = len(TRACKING_BODY_NAMES)

  t = 1
  np.savez(
    path,
    joint_pos=np.zeros((t, expected_joints), dtype=np.float32),
    joint_vel=np.zeros((t, expected_joints), dtype=np.float32),
    body_pos_w=np.zeros((t, expected_bodies, 3), dtype=np.float32),
    body_quat_w=np.zeros((t, expected_bodies, 4), dtype=np.float32),
    body_lin_vel_w=np.zeros((t, expected_bodies, 3), dtype=np.float32),
    body_ang_vel_w=np.zeros((t, expected_bodies, 3), dtype=np.float32),
  )
  validate_motion_npz(path)


def test_tracker_train_offline_mode_preflight(tmp_path):
  from tracker.integrations.mjlab.bootstrap import bootstrap

  bootstrap()

  from tracker.cli.train import resolve_train_mode

  motion_path = tmp_path / "motion.npz"
  _write_minimal_adam_sp_motion_npz(motion_path)

  mode, remaining = resolve_train_mode(
    "Tracker-Tracking-Flat-Adam-SP-23", ["--motion-file", str(motion_path)]
  )
  assert mode == "offline"
  assert remaining == []

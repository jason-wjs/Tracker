import os

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


def test_offline_train_sets_motion_file(tmp_path):
  from tracker.integrations.mjlab.bootstrap import bootstrap

  bootstrap()

  from mjlab.scripts.train import TrainConfig
  from tracker.integrations.mjlab.offline_train import apply_offline_motion_file

  task_id = "Tracker-Tracking-Flat-Adam-SP-23"
  cfg = TrainConfig.from_task(task_id)

  motion_path = tmp_path / "motion.npz"
  _write_minimal_adam_sp_motion_npz(motion_path)

  apply_offline_motion_file(cfg.env, str(motion_path))
  assert cfg.env.commands is not None
  assert cfg.env.commands["motion"].motion_file == str(motion_path)


def test_offline_train_disables_wandb_by_default(monkeypatch):
  from tracker.integrations.mjlab.offline_train import ensure_offline_safe_logging
  from mjlab.rl.config import RslRlOnPolicyRunnerCfg

  monkeypatch.delenv("WANDB_API_KEY", raising=False)
  monkeypatch.delenv("WANDB_MODE", raising=False)
  monkeypatch.delenv("WANDB_DISABLED", raising=False)

  agent = RslRlOnPolicyRunnerCfg(logger="wandb")
  ensure_offline_safe_logging(agent, env=os.environ)
  assert agent.logger == "tensorboard"

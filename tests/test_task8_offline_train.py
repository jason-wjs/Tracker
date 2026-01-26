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

  # Ensure this test is hermetic even if the developer has `wandb login` creds in
  # `~/.netrc` (which `ensure_offline_safe_logging()` treats as "configured").
  try:
    from wandb.sdk.lib import auth as wandb_auth

    monkeypatch.setattr(wandb_auth, "read_netrc_auth", lambda host=None: None)
  except Exception:
    pass

  agent = RslRlOnPolicyRunnerCfg(logger="wandb")
  ensure_offline_safe_logging(agent, env=os.environ)
  assert agent.logger == "tensorboard"


def test_apply_motion_pack_preserves_sampling_mode():
  from pathlib import Path

  from tracker.tasks.tracking.config.env import adam_sp_flat_tracking_env_cfg
  from tracker.tasks.tracking.config.patch import apply_motion_pack

  cfg = adam_sp_flat_tracking_env_cfg()
  assert cfg.commands is not None
  assert cfg.commands["motion"].sampling_mode == "adaptive"

  apply_motion_pack(cfg, pack_dir=Path("dummy_pack"), split="train")
  assert cfg.commands["motion"].sampling_mode == "adaptive"

  apply_motion_pack(cfg, pack_dir=Path("dummy_pack"), split="train", sampling_mode="uniform")
  assert cfg.commands["motion"].sampling_mode == "uniform"

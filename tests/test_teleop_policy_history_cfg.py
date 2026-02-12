from tracker.tasks.tracking.config.env import (
  adam_pro_29_flat_teleop_env_cfg,
  adam_pro_29_flat_tracking_env_cfg,
)


def test_teleop_task_uses_selective_policy_history_len_8():
  # The teleop task should override per-term policy observation history length,
  # while normal tracking tasks should remain unchanged.
  base_cfg = adam_pro_29_flat_tracking_env_cfg(has_state_estimation=False)
  assert base_cfg.observations["policy"].history_length is None

  teleop_cfg = adam_pro_29_flat_teleop_env_cfg(has_state_estimation=False)
  # Selective history: no group-level override, but set history on key proprio/action terms.
  assert teleop_cfg.observations["policy"].history_length is None
  assert teleop_cfg.observations["policy"].flatten_history_dim is True

  terms = teleop_cfg.observations["policy"].terms
  for term_name in ("joint_pos", "joint_vel", "base_ang_vel", "actions"):
    assert terms[term_name].history_length == 8
    assert terms[term_name].flatten_history_dim is True
  for term_name in ("command", "motion_anchor_ori_b"):
    assert terms[term_name].history_length == 0

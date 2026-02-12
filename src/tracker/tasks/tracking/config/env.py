"""Tracking environment configurations (pure builders, no registration side effects)."""

from mjlab.envs import ManagerBasedRlEnvCfg

from tracker.robots.adam_pro import (
  TRACKING_ANCHOR_BODY as TRACKING_ANCHOR_BODY_PRO,
  TRACKING_BODY_NAMES as TRACKING_BODY_NAMES_PRO,
  get_action_scale as get_action_scale_pro,
  get_robot_cfg as get_robot_cfg_pro,
)
from tracker.robots.adam_pro_29 import (
  TRACKING_ANCHOR_BODY as TRACKING_ANCHOR_BODY_PRO_29,
  TRACKING_BODY_NAMES as TRACKING_BODY_NAMES_PRO_29,
  get_action_scale as get_action_scale_pro_29,
  get_robot_cfg as get_robot_cfg_pro_29,
)
from tracker.robots.adam_sp import (
  TRACKING_ANCHOR_BODY,
  TRACKING_BODY_NAMES,
  get_action_scale,
  get_robot_cfg,
)
from tracker.robots.adam_sp_29 import (
  TRACKING_ANCHOR_BODY as TRACKING_ANCHOR_BODY_29,
  TRACKING_BODY_NAMES as TRACKING_BODY_NAMES_29,
  get_action_scale as get_action_scale_29,
  get_robot_cfg as get_robot_cfg_29,
)
from tracker.tasks.tracking.config.patch import make_flat_tracking_env_cfg_for_robot

DOMAIN_RAND_LINK_BODY_NAMES = (
  "pelvis",
  "hipPitchLeft",
  "hipRollLeft",
  "thighLeft",
  "shinLeft",
  "hipPitchRight",
  "hipRollRight",
  "thighRight",
  "shinRight",
  "shoulderPitchLeft",
  "shoulderRollLeft",
  "shoulderYawLeft",
  "elbowLeft",
  "shoulderPitchRight",
  "shoulderRollRight",
  "shoulderYawRight",
  "elbowRight",
)


def adam_sp_flat_tracking_env_cfg(
  *,
  has_state_estimation: bool = True,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create Adam-SP (23-DoF) flat terrain tracking configuration."""
  return make_flat_tracking_env_cfg_for_robot(
    get_robot_cfg=get_robot_cfg,
    action_scale=get_action_scale(),
    tracking_anchor_body=TRACKING_ANCHOR_BODY,
    tracking_body_names=TRACKING_BODY_NAMES,
    domain_rand_link_body_names=DOMAIN_RAND_LINK_BODY_NAMES,
    has_state_estimation=has_state_estimation,
    play=play,
  )


def adam_sp_29_flat_tracking_env_cfg(
  *,
  has_state_estimation: bool = True,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create Adam-SP (29-DoF) flat terrain tracking configuration."""
  return make_flat_tracking_env_cfg_for_robot(
    get_robot_cfg=get_robot_cfg_29,
    action_scale=get_action_scale_29(),
    tracking_anchor_body=TRACKING_ANCHOR_BODY_29,
    tracking_body_names=TRACKING_BODY_NAMES_29,
    domain_rand_link_body_names=DOMAIN_RAND_LINK_BODY_NAMES,
    has_state_estimation=has_state_estimation,
    play=play,
  )


def adam_pro_flat_tracking_env_cfg(
  *,
  has_state_estimation: bool = True,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create Adam-Pro (23-DoF) flat terrain tracking configuration."""
  return make_flat_tracking_env_cfg_for_robot(
    get_robot_cfg=get_robot_cfg_pro,
    action_scale=get_action_scale_pro(),
    tracking_anchor_body=TRACKING_ANCHOR_BODY_PRO,
    tracking_body_names=TRACKING_BODY_NAMES_PRO,
    domain_rand_link_body_names=DOMAIN_RAND_LINK_BODY_NAMES,
    observation_param_aliases={
      "robot/imu_lin_vel": "robot/BodyVel",
      "robot/imu_ang_vel": "robot/BodyGyro",
      "robot/imu_lin_acc": "robot/BodyAcc",
    },
    has_state_estimation=has_state_estimation,
    play=play,
  )


def adam_pro_29_flat_tracking_env_cfg(
  *,
  has_state_estimation: bool = True,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create Adam-Pro (29-DoF) flat terrain tracking configuration."""
  return make_flat_tracking_env_cfg_for_robot(
    get_robot_cfg=get_robot_cfg_pro_29,
    action_scale=get_action_scale_pro_29(),
    tracking_anchor_body=TRACKING_ANCHOR_BODY_PRO_29,
    tracking_body_names=TRACKING_BODY_NAMES_PRO_29,
    domain_rand_link_body_names=DOMAIN_RAND_LINK_BODY_NAMES,
    observation_param_aliases={
      "robot/imu_lin_vel": "robot/BodyVel",
      "robot/imu_ang_vel": "robot/BodyGyro",
      "robot/imu_lin_acc": "robot/BodyAcc",
    },
    has_state_estimation=has_state_estimation,
    play=play,
  )


def adam_pro_29_flat_teleop_env_cfg(
  *,
  has_state_estimation: bool = True,
  play: bool = False,
  history_length: int = 8,
) -> ManagerBasedRlEnvCfg:
  """Create teleop (multi-clip) config with short policy observation history.

  This uses mjlab's built-in observation history buffers so training/eval/play
  share the same observation interface without modifying mjlab itself.
  """
  cfg = adam_pro_29_flat_tracking_env_cfg(
    has_state_estimation=has_state_estimation,
    play=play,
  )
  # Prefer selective history (TWIST2-style): keep commands/reference terms
  # current-only, while providing short temporal context for proprio/actions.
  cfg.observations["policy"].history_length = None
  cfg.observations["policy"].flatten_history_dim = True

  policy_terms = cfg.observations["policy"].terms
  for term_name in ("joint_pos", "joint_vel", "base_ang_vel", "actions"):
    if term_name in policy_terms:
      policy_terms[term_name].history_length = history_length
      policy_terms[term_name].flatten_history_dim = True
  return cfg

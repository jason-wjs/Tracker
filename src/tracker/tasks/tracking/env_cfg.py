"""Adam-SP tracking environment configuration.

Builds on `mjlab.tasks.tracking.tracking_env_cfg.make_tracking_env_cfg()` and patches
robot-specific settings (robot entity, actuator scaling, tracking body set, sensors).
"""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.manager_term_config import EventTermCfg, ObservationGroupCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg

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
  cfg = make_tracking_env_cfg()

  cfg.scene.entities = {"robot": get_robot_cfg()}

  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    fields=("found",),
    reduce="none",
    num_slots=1,
  )
  cfg.scene.sensors = (self_collision_cfg,)

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = get_action_scale()

  assert cfg.commands is not None
  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  motion_cmd.anchor_body_name = TRACKING_ANCHOR_BODY
  motion_cmd.body_names = TRACKING_BODY_NAMES

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = (
    r"^(toe(Left|Right)_collision|left_foot[0-9]+_collision|"
    r"right_foot[0-9]+_collision)$"
  )
  cfg.events["base_com"].params["asset_cfg"].body_names = (TRACKING_ANCHOR_BODY,)
  cfg.events["base_com"].params["ranges"] = {
    0: (-0.02, 0.02),
    1: (-0.02, 0.02),
    2: (-0.02, 0.02),
  }

  # Domain randomization (deployment-oriented): add moderate physics randomization
  # on top of mjlab's tracking baseline terms (base_com, foot_friction, encoder_bias,
  # push_robot). Keep these startup randomizations enabled in play mode (Phase 1).
  randomize_field_fn = cfg.events["base_com"].func
  cfg.events["physics_body_mass"] = EventTermCfg(
    mode="startup",
    func=randomize_field_fn,
    domain_randomization=True,
    params={
      "asset_cfg": SceneEntityCfg("robot", body_names=DOMAIN_RAND_LINK_BODY_NAMES),
      "operation": "scale",
      "field": "body_mass",
      "distribution": "uniform",
      "ranges": (0.9, 1.1),
    },
  )
  cfg.events["physics_body_inertia"] = EventTermCfg(
    mode="startup",
    func=randomize_field_fn,
    domain_randomization=True,
    params={
      "asset_cfg": SceneEntityCfg("robot", body_names=DOMAIN_RAND_LINK_BODY_NAMES),
      "operation": "scale",
      "field": "body_inertia",
      "distribution": "uniform",
      "ranges": (0.9, 1.1),
    },
  )
  cfg.events["link_com"] = EventTermCfg(
    mode="startup",
    func=randomize_field_fn,
    domain_randomization=True,
    params={
      "asset_cfg": SceneEntityCfg("robot", body_names=DOMAIN_RAND_LINK_BODY_NAMES),
      "operation": "add",
      "field": "body_ipos",
      "distribution": "uniform",
      "ranges": {
        0: (-0.02, 0.02),
        1: (-0.02, 0.02),
        2: (-0.02, 0.02),
      },
    },
  )
  cfg.events["physics_dof_damping"] = EventTermCfg(
    mode="startup",
    func=randomize_field_fn,
    domain_randomization=True,
    params={
      "asset_cfg": SceneEntityCfg("robot"),
      "operation": "scale",
      "field": "dof_damping",
      "distribution": "uniform",
      "ranges": (0.60, 1.40),
    },
  )
  cfg.events["physics_dof_frictionloss"] = EventTermCfg(
    mode="startup",
    func=randomize_field_fn,
    domain_randomization=True,
    params={
      "asset_cfg": SceneEntityCfg("robot"),
      "operation": "scale",
      "field": "dof_frictionloss",
      "distribution": "log_uniform",
      "ranges": (0.2, 1.5),
    },
  )

  cfg.terminations["ee_body_pos"].params["body_names"] = (
    "toeLeft",
    "toeRight",
    "wristRollLeft",
    "wristRollRight",
  )

  cfg.viewer.body_name = TRACKING_ANCHOR_BODY

  if not has_state_estimation:
    new_policy_terms = {
      k: v
      for k, v in cfg.observations["policy"].terms.items()
      if k not in ["motion_anchor_pos_b", "base_lin_vel"]
    }
    cfg.observations["policy"] = ObservationGroupCfg(
      terms=new_policy_terms,
      concatenate_terms=True,
      enable_corruption=True,
    )

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["policy"].enable_corruption = False
    cfg.events.pop("push_robot", None)

    motion_cmd.pose_range = {}
    motion_cmd.velocity_range = {}
    motion_cmd.sampling_mode = "start"

  return cfg


def adam_sp_29_flat_tracking_env_cfg(
  *,
  has_state_estimation: bool = True,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create Adam-SP (29-DoF) flat terrain tracking configuration."""
  cfg = make_tracking_env_cfg()

  cfg.scene.entities = {"robot": get_robot_cfg_29()}

  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    fields=("found",),
    reduce="none",
    num_slots=1,
  )
  cfg.scene.sensors = (self_collision_cfg,)

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = get_action_scale_29()

  assert cfg.commands is not None
  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  motion_cmd.anchor_body_name = TRACKING_ANCHOR_BODY_29
  motion_cmd.body_names = TRACKING_BODY_NAMES_29

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = (
    r"^(toe(Left|Right)_collision|left_foot[0-9]+_collision|"
    r"right_foot[0-9]+_collision)$"
  )
  cfg.events["base_com"].params["asset_cfg"].body_names = (TRACKING_ANCHOR_BODY_29,)
  cfg.events["base_com"].params["ranges"] = {
    0: (-0.02, 0.02),
    1: (-0.02, 0.02),
    2: (-0.02, 0.02),
  }

  randomize_field_fn = cfg.events["base_com"].func
  cfg.events["physics_body_mass"] = EventTermCfg(
    mode="startup",
    func=randomize_field_fn,
    domain_randomization=True,
    params={
      "asset_cfg": SceneEntityCfg("robot", body_names=DOMAIN_RAND_LINK_BODY_NAMES),
      "operation": "scale",
      "field": "body_mass",
      "distribution": "uniform",
      "ranges": (0.9, 1.1),
    },
  )
  cfg.events["physics_body_inertia"] = EventTermCfg(
    mode="startup",
    func=randomize_field_fn,
    domain_randomization=True,
    params={
      "asset_cfg": SceneEntityCfg("robot", body_names=DOMAIN_RAND_LINK_BODY_NAMES),
      "operation": "scale",
      "field": "body_inertia",
      "distribution": "uniform",
      "ranges": (0.9, 1.1),
    },
  )
  cfg.events["link_com"] = EventTermCfg(
    mode="startup",
    func=randomize_field_fn,
    domain_randomization=True,
    params={
      "asset_cfg": SceneEntityCfg("robot", body_names=DOMAIN_RAND_LINK_BODY_NAMES),
      "operation": "add",
      "field": "body_ipos",
      "distribution": "uniform",
      "ranges": {
        0: (-0.02, 0.02),
        1: (-0.02, 0.02),
        2: (-0.02, 0.02),
      },
    },
  )
  cfg.events["physics_dof_damping"] = EventTermCfg(
    mode="startup",
    func=randomize_field_fn,
    domain_randomization=True,
    params={
      "asset_cfg": SceneEntityCfg("robot"),
      "operation": "scale",
      "field": "dof_damping",
      "distribution": "uniform",
      "ranges": (0.60, 1.40),
    },
  )
  cfg.events["physics_dof_frictionloss"] = EventTermCfg(
    mode="startup",
    func=randomize_field_fn,
    domain_randomization=True,
    params={
      "asset_cfg": SceneEntityCfg("robot"),
      "operation": "scale",
      "field": "dof_frictionloss",
      "distribution": "log_uniform",
      "ranges": (0.2, 1.5),
    },
  )

  cfg.terminations["ee_body_pos"].params["body_names"] = (
    "toeLeft",
    "toeRight",
    "wristRollLeft",
    "wristRollRight",
  )

  cfg.viewer.body_name = TRACKING_ANCHOR_BODY_29

  if not has_state_estimation:
    new_policy_terms = {
      k: v
      for k, v in cfg.observations["policy"].terms.items()
      if k not in ["motion_anchor_pos_b", "base_lin_vel"]
    }
    cfg.observations["policy"] = ObservationGroupCfg(
      terms=new_policy_terms,
      concatenate_terms=True,
      enable_corruption=True,
    )

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["policy"].enable_corruption = False
    cfg.events.pop("push_robot", None)

    motion_cmd.pose_range = {}
    motion_cmd.velocity_range = {}
    motion_cmd.sampling_mode = "start"

  return cfg

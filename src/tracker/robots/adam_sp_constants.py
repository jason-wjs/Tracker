"""Adam-SP robot constants and construction helpers."""

from __future__ import annotations

from pathlib import Path

import mujoco

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.os import update_assets
from mjlab.utils.spec_config import CollisionCfg

from tracker.assets.paths import get_asset_path

ADAM_SP_XML: Path = get_asset_path("adam_sp", "adam_sp.xml")

# Motor reflected inertia (armature) baselines.
ARMATURE_130_92_7_P = 0.13426
ARMATURE_50_14A_50_S = 0.1578807
ARMATURE_30_14A_50_S = 0.0423963
ARMATURE_60_17_50_S = 0.23409
ARMATURE_80_20_30_S = 0.281573
ARMATURE_50_52_30_P = 0.0549

##
# PD gains baselines.
#
# stiffness: [N*m/rad]
# damping:   [N*m*s/rad]
##

STIFFNESS_130_92_7_P = 305.0
DAMPING_130_92_7_P = 5.0

STIFFNESS_80_20_30_S = 255.0
DAMPING_80_20_30_S = 3.5

STIFFNESS_60_17_50_S = 255.0
DAMPING_60_17_50_S = 3.5
STIFFNESS_60_17_50_S_WAIST_PITCH = 305.0
DAMPING_60_17_50_S_WAIST_PITCH = 5.0

STIFFNESS_50_52_30_P_ANKLE_PITCH = 50.0
DAMPING_50_52_30_P_ANKLE_PITCH = 0.8
STIFFNESS_50_52_30_P_ANKLE_ROLL = 30.0
DAMPING_50_52_30_P_ANKLE_ROLL = 0.35

STIFFNESS_50_14A_50_S = 40.0
DAMPING_50_14A_50_S = 1.0

STIFFNESS_30_14A_50_S = 40.0
DAMPING_30_14A_50_S = 1.0

##
# Effort limits (N*m).
##

EFFORT_LIMIT_130_92_7_P = 340.0
EFFORT_LIMIT_80_20_30_S = 120.0
EFFORT_LIMIT_60_17_50_S = 89.0
EFFORT_LIMIT_50_52_30_P = 46.0
EFFORT_LIMIT_50_14A_50_S = 60.0
EFFORT_LIMIT_30_14A_50_S = 17.5

# Baseline joint frictionloss (dry friction) [N*m] for deployment-oriented DR.
# mjlab's `physics_dof_frictionloss` term scales existing dof_frictionloss; if the
# baseline is 0 everywhere, that DR term becomes a no-op.
FRICTIONLOSS_LEG = 0.05
FRICTIONLOSS_ANKLE = 0.02
FRICTIONLOSS_WAIST = 0.03
FRICTIONLOSS_SHOULDER = 0.02
FRICTIONLOSS_ELBOW = 0.02

# Baseline dof damping (viscous friction) [N*m*s/rad] for deployment-oriented DR.
# mjlab's `physics_dof_damping` term scales existing dof_damping; if the baseline is
# 0 everywhere, that DR term becomes a no-op.
DOF_DAMPING_DEFAULT = 0.02


def _baseline_joint_damping(joint_name: str) -> float:
  if joint_name.startswith(("hipPitch_", "kneePitch_")):
    return 0.05
  if joint_name.startswith(("hipRoll_", "hipYaw_")):
    return 0.03
  if joint_name.startswith(("anklePitch_", "ankleRoll_")):
    return 0.02
  if joint_name.startswith("waist"):
    return 0.02
  if joint_name.startswith(("shoulderPitch_", "shoulderRoll_")):
    return 0.02
  if joint_name.startswith(("shoulderYaw_", "elbow_")):
    return 0.02
  return DOF_DAMPING_DEFAULT


ADAM_SP_ACT_LEG_PITCH = BuiltinPositionActuatorCfg(
  target_names_expr=(r"hipPitch_.*", r"kneePitch_.*"),
  stiffness=STIFFNESS_130_92_7_P,
  damping=DAMPING_130_92_7_P,
  effort_limit=EFFORT_LIMIT_130_92_7_P,
  armature=ARMATURE_130_92_7_P,
  frictionloss=FRICTIONLOSS_LEG,
)

ADAM_SP_ACT_LEG_ROLL = BuiltinPositionActuatorCfg(
  target_names_expr=(r"hipRoll_.*",),
  stiffness=STIFFNESS_80_20_30_S,
  damping=DAMPING_80_20_30_S,
  effort_limit=EFFORT_LIMIT_80_20_30_S,
  armature=ARMATURE_80_20_30_S,
  frictionloss=FRICTIONLOSS_LEG,
)

ADAM_SP_ACT_LEG_YAW = BuiltinPositionActuatorCfg(
  target_names_expr=(r"hipYaw_.*",),
  stiffness=STIFFNESS_60_17_50_S,
  damping=DAMPING_60_17_50_S,
  effort_limit=EFFORT_LIMIT_60_17_50_S,
  armature=ARMATURE_60_17_50_S,
  frictionloss=FRICTIONLOSS_LEG,
)

ADAM_SP_ACT_ANKLE_PITCH = BuiltinPositionActuatorCfg(
  target_names_expr=(r"anklePitch_.*",),
  stiffness=STIFFNESS_50_52_30_P_ANKLE_PITCH,
  damping=DAMPING_50_52_30_P_ANKLE_PITCH,
  effort_limit=EFFORT_LIMIT_50_52_30_P,
  armature=ARMATURE_50_52_30_P,
  frictionloss=FRICTIONLOSS_ANKLE,
)

ADAM_SP_ACT_ANKLE_ROLL = BuiltinPositionActuatorCfg(
  target_names_expr=(r"ankleRoll_.*",),
  stiffness=STIFFNESS_50_52_30_P_ANKLE_ROLL,
  damping=DAMPING_50_52_30_P_ANKLE_ROLL,
  effort_limit=EFFORT_LIMIT_50_52_30_P,
  armature=ARMATURE_50_52_30_P,
  frictionloss=FRICTIONLOSS_ANKLE,
)

ADAM_SP_ACT_WAIST_ROLL_YAW = BuiltinPositionActuatorCfg(
  target_names_expr=(r"waistRoll", r"waistYaw"),
  stiffness=STIFFNESS_60_17_50_S,
  damping=DAMPING_60_17_50_S,
  effort_limit=EFFORT_LIMIT_60_17_50_S,
  armature=ARMATURE_60_17_50_S,
  frictionloss=FRICTIONLOSS_WAIST,
)

ADAM_SP_ACT_WAIST_PITCH = BuiltinPositionActuatorCfg(
  target_names_expr=(r"waistPitch",),
  stiffness=STIFFNESS_60_17_50_S_WAIST_PITCH,
  damping=DAMPING_60_17_50_S_WAIST_PITCH,
  effort_limit=EFFORT_LIMIT_60_17_50_S,
  armature=ARMATURE_60_17_50_S,
  frictionloss=FRICTIONLOSS_WAIST,
)

ADAM_SP_ACT_SHOULDER = BuiltinPositionActuatorCfg(
  target_names_expr=(r"shoulderPitch_.*", r"shoulderRoll_.*"),
  stiffness=STIFFNESS_50_14A_50_S,
  damping=DAMPING_50_14A_50_S,
  effort_limit=EFFORT_LIMIT_50_14A_50_S,
  armature=ARMATURE_50_14A_50_S,
  frictionloss=FRICTIONLOSS_SHOULDER,
)

ADAM_SP_ACT_ELBOW = BuiltinPositionActuatorCfg(
  target_names_expr=(r"shoulderYaw_.*", r"elbow_.*"),
  stiffness=STIFFNESS_30_14A_50_S,
  damping=DAMPING_30_14A_50_S,
  effort_limit=EFFORT_LIMIT_30_14A_50_S,
  armature=ARMATURE_30_14A_50_S,
  frictionloss=FRICTIONLOSS_ELBOW,
)

ADAM_SP_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    ADAM_SP_ACT_LEG_PITCH,
    ADAM_SP_ACT_LEG_ROLL,
    ADAM_SP_ACT_LEG_YAW,
    ADAM_SP_ACT_ANKLE_PITCH,
    ADAM_SP_ACT_ANKLE_ROLL,
    ADAM_SP_ACT_WAIST_ROLL_YAW,
    ADAM_SP_ACT_WAIST_PITCH,
    ADAM_SP_ACT_SHOULDER,
    ADAM_SP_ACT_ELBOW,
  ),
  soft_joint_pos_limit_factor=0.9,
)

##
# Initial state and collisions.
##

ADAM_SP_INIT_STATE = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.90),
  rot=(0.0, 0.0, 0.0, 1.0),
  lin_vel=(0.0, 0.0, 0.0),
  ang_vel=(0.0, 0.0, 0.0),
  joint_pos={
    "hipPitch_Left": -0.32,
    "hipRoll_Left": 0.0,
    "hipYaw_Left": -0.18,
    "kneePitch_Left": 0.66,
    "anklePitch_Left": -0.39,
    "ankleRoll_Left": 0.0,
    "hipPitch_Right": -0.32,
    "hipRoll_Right": 0.0,
    "hipYaw_Right": 0.18,
    "kneePitch_Right": 0.66,
    "anklePitch_Right": -0.39,
    "ankleRoll_Right": 0.0,
    "waistRoll": 0.0,
    "waistPitch": 0.0,
    "waistYaw": 0.0,
    "shoulderPitch_Left": 0.0,
    "shoulderRoll_Left": 0.1,
    "shoulderYaw_Left": 0.0,
    "elbow_Left": -0.3,
    "shoulderPitch_Right": 0.0,
    "shoulderRoll_Right": -0.1,
    "shoulderYaw_Right": 0.0,
    "elbow_Right": -0.3,
  },
  joint_vel={".*": 0.0},
)

# Collision settings: keep the MJCF defaults for what collides, but apply a small set
# of targeted overrides to match mjlab conventions:
# - avoid unintentionally enabling collisions on visual geoms
# - set toe friction (ground contact)
# - disable hand/finger collisions for Phase 1
ADAM_SP_COLLISIONS = (
  CollisionCfg(
    geom_names_expr=(r".*_collision",),
    disable_other_geoms=True,
    contype={r"^(L_|R_).*_collision$": 0, r".*_collision$": 1},
    conaffinity={r"^(L_|R_).*_collision$": 0, r".*_collision$": 1},
    priority={r"^(toeLeft_collision|toeRight_collision)$": 1, r".*_collision$": 0},
    friction={r"^(toeLeft_collision|toeRight_collision)$": (0.6,)},
  ),
)

#
# Multiple keyframes (future work): follow mjlab's pattern (e.g., HOME_KEYFRAME,
# KNEES_BENT_KEYFRAME) if additional initial poses become useful.
#
ADAM_SP_KEYFRAMES: dict[str, EntityCfg.InitialStateCfg] = {"default": ADAM_SP_INIT_STATE}


def get_assets(meshdir: str) -> dict[str, bytes]:
  assets: dict[str, bytes] = {}
  update_assets(assets, ADAM_SP_XML.parent / "meshes", meshdir, recursive=True)
  return assets


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(ADAM_SP_XML))
  spec.assets = get_assets(spec.meshdir)
  for joint_name in ADAM_SP_INIT_STATE.joint_pos:
    spec.joint(joint_name).damping = _baseline_joint_damping(joint_name)
  return spec


def get_adam_sp_robot_cfg() -> EntityCfg:
  """Return a fresh Adam-SP robot configuration instance."""
  return EntityCfg(
    init_state=ADAM_SP_INIT_STATE,
    collisions=ADAM_SP_COLLISIONS,
    spec_fn=get_spec,
    articulation=ADAM_SP_ARTICULATION,
  )


ADAM_SP_ACTION_SCALE: dict[str, float] = {}
for actuator in ADAM_SP_ARTICULATION.actuators:
  assert isinstance(actuator, BuiltinPositionActuatorCfg)
  effort = actuator.effort_limit
  stiffness = actuator.stiffness
  assert effort is not None
  for expr in actuator.target_names_expr:
    ADAM_SP_ACTION_SCALE[expr] = 0.25 * effort / stiffness

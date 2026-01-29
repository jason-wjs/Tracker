"""Shared builders for tracker tracking environment configs.

This module must remain import-safe: it defines helpers used by env-cfg builders,
but it should not register tasks at import time.
"""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import Callable, Mapping, Sequence

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.manager_term_config import EventTermCfg, ObservationGroupCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg

GetRobotCfgFn = Callable[[], object]

_FOOT_FRICTION_GEOMS_REGEX = (
  r"^(toe(Left|Right)_collision|left_foot[0-9]+_collision|right_foot[0-9]+_collision)$"
)
_DEFAULT_EE_BODY_NAMES = ("toeLeft", "toeRight", "wristRollLeft", "wristRollRight")


def _apply_observation_param_aliases(
  cfg: ManagerBasedRlEnvCfg,
  aliases: Mapping[str, str],
) -> None:
  def rewrite(value):
    if isinstance(value, str):
      return aliases.get(value, value)
    if isinstance(value, dict):
      for k, v in value.items():
        value[k] = rewrite(v)
      return value
    if isinstance(value, list):
      for idx, v in enumerate(value):
        value[idx] = rewrite(v)
      return value
    if isinstance(value, tuple):
      return tuple(rewrite(v) for v in value)
    return value

  for group in cfg.observations.values():
    for term_cfg in group.terms.values():
      term_cfg.params = rewrite(term_cfg.params)


def make_flat_tracking_env_cfg_for_robot(
  *,
  get_robot_cfg: GetRobotCfgFn,
  action_scale: dict[str, float],
  tracking_anchor_body: str,
  tracking_body_names: Sequence[str],
  domain_rand_link_body_names: Sequence[str],
  observation_param_aliases: Mapping[str, str] | None = None,
  has_state_estimation: bool = True,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
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
  joint_pos_action.scale = action_scale

  assert cfg.commands is not None
  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  motion_cmd.anchor_body_name = tracking_anchor_body
  motion_cmd.body_names = tuple(tracking_body_names)

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = _FOOT_FRICTION_GEOMS_REGEX
  cfg.events["base_com"].params["asset_cfg"].body_names = (tracking_anchor_body,)
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
      "asset_cfg": SceneEntityCfg("robot", body_names=tuple(domain_rand_link_body_names)),
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
      "asset_cfg": SceneEntityCfg("robot", body_names=tuple(domain_rand_link_body_names)),
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
      "asset_cfg": SceneEntityCfg("robot", body_names=tuple(domain_rand_link_body_names)),
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

  cfg.terminations["ee_body_pos"].params["body_names"] = _DEFAULT_EE_BODY_NAMES
  cfg.viewer.body_name = tracking_anchor_body

  # Always apply observation parameter aliases when provided.
  #
  # This keeps tracker robots compatible with mjlab's default observation terms
  # (e.g., `robot/imu_ang_vel`) while allowing robots to expose differently
  # named sensors.
  if observation_param_aliases:
    _apply_observation_param_aliases(cfg, observation_param_aliases)

  if not has_state_estimation:
    # Match mjlab "no-state-estimation" definition exactly: drop anchor position
    # and IMU linear velocity terms from policy observations.
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


def apply_motion_pack(
  cfg: ManagerBasedRlEnvCfg,
  *,
  pack_dir: Path,
  split: str,
  sampling_mode: str | None = None,
  clip_curriculum_mode: str | None = None,
  clip_curriculum_mix: float | None = None,
  clip_curriculum_strength: float | None = None,
  clip_curriculum_tau_scale: float | None = None,
  clip_curriculum_max_mult: float | None = None,
) -> None:
  """Patch a tracking env config to use a packed multi-clip motion dataset."""
  assert cfg.commands is not None
  motion_cmd = cfg.commands.get("motion")
  if not isinstance(motion_cmd, MotionCommandCfg):
    raise ValueError("Task does not appear to be a motion-tracking task.")

  from tracker.tasks.tracking.multiclip_command import MultiClipMotionCommand, MotionPackCommandCfg

  kwargs = {f.name: getattr(motion_cmd, f.name) for f in fields(MotionCommandCfg)}
  kwargs["motion_file"] = str(pack_dir)
  kwargs["class_type"] = MultiClipMotionCommand
  if sampling_mode is not None:
    if sampling_mode not in ("adaptive", "uniform", "start"):
      raise ValueError(
        f"Unknown sampling_mode override: {sampling_mode!r} "
        "(expected 'adaptive', 'uniform', or 'start')"
      )
    kwargs["sampling_mode"] = sampling_mode

  if clip_curriculum_mode is not None:
    if clip_curriculum_mode not in ("ema_bin_failed", "off"):
      raise ValueError(
        f"Unknown clip_curriculum_mode override: {clip_curriculum_mode!r} "
        "(expected 'ema_bin_failed' or 'off')"
      )
    kwargs["clip_curriculum_mode"] = clip_curriculum_mode
  if clip_curriculum_mix is not None:
    if not (0.0 <= clip_curriculum_mix <= 1.0):
      raise ValueError("clip_curriculum_mix must be in [0, 1]")
    kwargs["clip_curriculum_mix"] = float(clip_curriculum_mix)
  if clip_curriculum_strength is not None:
    if clip_curriculum_strength < 0:
      raise ValueError("clip_curriculum_strength must be >= 0")
    kwargs["clip_curriculum_strength"] = float(clip_curriculum_strength)
  if clip_curriculum_tau_scale is not None:
    if clip_curriculum_tau_scale <= 0:
      raise ValueError("clip_curriculum_tau_scale must be > 0")
    kwargs["clip_curriculum_tau_scale"] = float(clip_curriculum_tau_scale)
  if clip_curriculum_max_mult is not None:
    if clip_curriculum_max_mult < 1.0:
      raise ValueError("clip_curriculum_max_mult must be >= 1")
    kwargs["clip_curriculum_max_mult"] = float(clip_curriculum_max_mult)

  cfg.commands["motion"] = MotionPackCommandCfg(**kwargs, motion_split=split)

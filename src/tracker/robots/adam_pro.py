"""Adam-Pro (23-DoF) robot spec and validation helpers."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import mujoco
import numpy as np

from tracker.robots import adam_pro_constants as constants

TRACKING_ANCHOR_BODY = "torso"
TRACKING_BODY_NAMES = (
  "pelvis",
  "hipPitchLeft",
  "hipRollLeft",
  "thighLeft",
  "shinLeft",
  "anklePitchLeft",
  "toeLeft",
  "hipPitchRight",
  "hipRollRight",
  "thighRight",
  "shinRight",
  "anklePitchRight",
  "toeRight",
  "waistRoll_link",
  "waistPitch_link",
  "torso",
  "shoulderPitchLeft",
  "shoulderRollLeft",
  "shoulderYawLeft",
  "elbowLeft",
  "wristYawLeft",
  "wristPitchLeft",
  "wristRollLeft",
  "shoulderPitchRight",
  "shoulderRollRight",
  "shoulderYawRight",
  "elbowRight",
  "wristYawRight",
  "wristPitchRight",
  "wristRollRight",
)


def get_robot_cfg():
  return constants.get_adam_pro_robot_cfg()


def get_action_scale() -> dict[str, float]:
  return constants.ADAM_PRO_ACTION_SCALE


@lru_cache(maxsize=1)
def _model_from_assets() -> mujoco.MjModel:
  spec = constants.get_spec()
  return spec.compile()


def _expected_nonfree_joint_names(model: mujoco.MjModel) -> list[str]:
  joint_names = [
    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)
  ]
  return [n for n in joint_names if n and n != "floating_base"]


def _expected_body_names(model: mujoco.MjModel) -> list[str]:
  return [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(model.nbody)]


def validate_motion_npz(path: Path) -> None:
  required = (
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
  )
  with np.load(path, allow_pickle=False) as data:
    missing = [k for k in required if k not in data]
    if missing:
      raise ValueError(f"Motion file missing keys: {missing}")

    model = _model_from_assets()
    expected_joint_names = _expected_nonfree_joint_names(model)
    expected_body_names = _expected_body_names(model)

    allowed_body_counts = {len(TRACKING_BODY_NAMES), model.nbody, model.nbody - 1}
    if "joint_names" in data:
      joint_names = [str(x) for x in data["joint_names"]]
      if joint_names != expected_joint_names:
        raise ValueError(
          "joint_names mismatch for Adam-Pro-23. "
          f"Expected {len(expected_joint_names)} names, got {len(joint_names)}."
        )

    if "body_names" in data:
      body_names = [str(x) for x in data["body_names"]]
      expected_body_names_no_world = expected_body_names[1:]
      if body_names == list(TRACKING_BODY_NAMES):
        allowed_body_counts = {len(TRACKING_BODY_NAMES)}
      elif body_names == expected_body_names:
        allowed_body_counts = {model.nbody}
      elif body_names == expected_body_names_no_world:
        allowed_body_counts = {model.nbody - 1}
      else:
        raise ValueError(
          "body_names mismatch for Adam-Pro-23. "
          f"Expected {len(TRACKING_BODY_NAMES)} (tracking subset), "
          f"{len(expected_body_names)} (with world) or {len(expected_body_names_no_world)} (no world) names, "
          f"got {len(body_names)}."
        )

    joint_pos = data["joint_pos"]
    joint_vel = data["joint_vel"]
    if (
      joint_pos.ndim != 2
      or joint_vel.ndim != 2
      or joint_pos.shape[0] != joint_vel.shape[0]
      or joint_pos.shape[1] != len(expected_joint_names)
      or joint_vel.shape[1] != len(expected_joint_names)
    ):
      raise ValueError(
        f"Expected joint_pos/joint_vel shape (T, {len(expected_joint_names)}), got {joint_pos.shape} and {joint_vel.shape}"
      )

    for key, dim in (
      ("body_pos_w", 3),
      ("body_quat_w", 4),
      ("body_lin_vel_w", 3),
      ("body_ang_vel_w", 3),
    ):
      arr = data[key]
      if (
        arr.ndim != 3
        or arr.shape[0] != joint_pos.shape[0]
        or arr.shape[1] not in allowed_body_counts
        or arr.shape[2] != dim
      ):
        raise ValueError(
          f"Expected {key} shape (T, B, {dim}) with B in {sorted(allowed_body_counts)}, got {arr.shape}"
        )

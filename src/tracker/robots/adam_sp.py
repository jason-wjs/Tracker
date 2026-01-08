"""Adam-SP robot spec and validation helpers."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from tracker.robots import adam_sp_constants as constants

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
  return constants.get_adam_sp_robot_cfg()


def get_action_scale() -> dict[str, float]:
  return constants.ADAM_SP_ACTION_SCALE


def _expected_joint_count() -> int:
  spec = constants.get_spec()
  return sum(
    1 for j in spec.joints if j.type != mujoco.mjtJoint.mjJNT_FREE
  )


def validate_motion_npz(path: Path) -> None:
  required = (
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
  )
  data = np.load(path)
  missing = [k for k in required if k not in data]
  if missing:
    raise ValueError(f"Motion file missing keys: {missing}")

  joint_pos = data["joint_pos"]
  joint_vel = data["joint_vel"]
  if joint_pos.shape != joint_vel.shape:
    raise ValueError(
      f"joint_pos/joint_vel shape mismatch: {joint_pos.shape} vs {joint_vel.shape}"
    )

  expected_joints = _expected_joint_count()
  if joint_pos.ndim != 2 or joint_pos.shape[1] != expected_joints:
    raise ValueError(
      f"Expected joint_pos shape (T, {expected_joints}), got {joint_pos.shape}"
    )

  expected_bodies = len(TRACKING_BODY_NAMES)
  for key, dim in (
    ("body_pos_w", 3),
    ("body_quat_w", 4),
    ("body_lin_vel_w", 3),
    ("body_ang_vel_w", 3),
  ):
    arr = data[key]
    if arr.ndim != 3 or arr.shape[1] != expected_bodies or arr.shape[2] != dim:
      raise ValueError(
        f"Expected {key} shape (T, {expected_bodies}, {dim}), got {arr.shape}"
      )

"""Adam-Pro (29-DoF) robot spec and motion-file preparation."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import mujoco
import numpy as np

from tracker.robots import adam_pro_29_constants as constants

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
  return constants.get_adam_pro_29_robot_cfg()


def get_action_scale() -> dict[str, float]:
  return constants.ADAM_PRO_29_ACTION_SCALE


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


def _default_motion_cache_dir() -> Path:
  if env := os.environ.get("TRACKER_MOTION_CACHE_DIR"):
    return Path(env)
  return Path.cwd() / ".tracker-cache" / "motions"


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

    include_world_from_names: bool | None = None
    if "joint_names" in data:
      joint_names = [str(x) for x in data["joint_names"]]
      if joint_names != expected_joint_names:
        raise ValueError(
          "joint_names mismatch for Adam-Pro-29. "
          f"Expected {len(expected_joint_names)} names, got {len(joint_names)}."
        )

    if "body_names" in data:
      body_names = [str(x) for x in data["body_names"]]
      expected_body_names_no_world = expected_body_names[1:]
      if body_names == expected_body_names:
        include_world_from_names = True
      elif body_names == expected_body_names_no_world:
        include_world_from_names = False
      else:
        raise ValueError(
          "body_names mismatch for Adam-Pro-29. "
          f"Expected {len(expected_body_names)} (with world) or {len(expected_body_names_no_world)} (no world) names, "
          f"got {len(body_names)}."
        )

    joint_pos = data["joint_pos"]
    joint_vel = data["joint_vel"]
    if (
      joint_pos.ndim != 2
      or joint_vel.ndim != 2
      or joint_pos.shape[0] != joint_vel.shape[0]
    ):
      raise ValueError(
        f"Expected joint_pos/joint_vel shape (T, D), got {joint_pos.shape} and {joint_vel.shape}"
      )

    expected_joint_dim = len(expected_joint_names)
    if joint_pos.shape[1] == expected_joint_dim and joint_vel.shape[1] == expected_joint_dim:
      pass
    elif joint_pos.shape[1] == model.nq and joint_vel.shape[1] == model.nv:
      pass
    else:
      raise ValueError(
        f"Unexpected joint_pos/joint_vel dims for Adam-Pro-29: {joint_pos.shape} vs {joint_vel.shape} "
        f"(expected (T, {expected_joint_dim}) or (T, nq={model.nq})/(T, nv={model.nv}))"
      )

    expected_body_count_with_world = model.nbody
    expected_body_count_no_world = model.nbody - 1
    if include_world_from_names is True:
      allowed_body_counts = {expected_body_count_with_world}
    elif include_world_from_names is False:
      allowed_body_counts = {expected_body_count_no_world}
    else:
      allowed_body_counts = {expected_body_count_with_world, expected_body_count_no_world}

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


def prepare_motion_npz(
  path: Path,
  *,
  cache_dir: Path | None = None,
) -> Path:
  """Prepare a motion file compatible with mjlab's tracking MotionLoader.

  mjlab expects `joint_pos` and `joint_vel` to be (T, num_nonfree_joints).
  Some datasets store MuJoCo `qpos/qvel` instead: (T, nq) and (T, nv).
  This helper slices off the floating base part into a cached normalized file.
  """
  validate_motion_npz(path)

  model = _model_from_assets()
  expected_joint_dim = len(_expected_nonfree_joint_names(model))

  with np.load(path, allow_pickle=False) as data:
    joint_pos = data["joint_pos"]
    joint_vel = data["joint_vel"]
    if joint_pos.shape[1] == expected_joint_dim and joint_vel.shape[1] == expected_joint_dim:
      return path

    if joint_pos.shape[1] != model.nq or joint_vel.shape[1] != model.nv:
      raise ValueError(
        f"Cannot convert motion file: expected qpos/qvel dims (nq={model.nq}, nv={model.nv}), "
        f"got {joint_pos.shape} and {joint_vel.shape}"
      )

    normalized_joint_pos = joint_pos[:, 7:].astype(np.float32, copy=False)
    normalized_joint_vel = joint_vel[:, 6:].astype(np.float32, copy=False)
    if (
      normalized_joint_pos.shape[1] != expected_joint_dim
      or normalized_joint_vel.shape[1] != expected_joint_dim
    ):
      raise ValueError(
        f"Converted joint dims mismatch: got {normalized_joint_pos.shape} and {normalized_joint_vel.shape}"
      )

    cache_dir_final = cache_dir or _default_motion_cache_dir()
    cache_dir_final.mkdir(parents=True, exist_ok=True)

    st = path.stat()
    key = f"{path.resolve()}:{st.st_size}:{st.st_mtime_ns}"
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]
    out_path = cache_dir_final / f"{path.stem}.adam_pro_29.norm.{digest}.npz"

    if out_path.exists():
      return out_path

    payload: dict[str, np.ndarray] = {k: data[k] for k in data.files}
    payload["joint_pos"] = normalized_joint_pos
    payload["joint_vel"] = normalized_joint_vel
    np.savez_compressed(out_path, **payload)
    return out_path

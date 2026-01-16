from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class MotionNpzMetadata:
  path: Path
  num_frames: int
  joint_dim: int
  body_count: int
  joint_names: tuple[str, ...] | None
  body_names: tuple[str, ...] | None


def _as_str_tuple(arr: np.ndarray) -> tuple[str, ...]:
  values: list[str] = []
  for item in arr.tolist():
    if isinstance(item, bytes):
      values.append(item.decode("utf-8"))
    else:
      values.append(str(item))
  return tuple(values)


def load_npz_metadata(path: Path) -> MotionNpzMetadata:
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

    joint_pos = data["joint_pos"]
    joint_vel = data["joint_vel"]
    if joint_pos.ndim != 2 or joint_vel.ndim != 2:
      raise ValueError(f"Expected joint_pos/joint_vel to be 2D, got {joint_pos.shape} and {joint_vel.shape}")
    if joint_pos.shape != joint_vel.shape:
      raise ValueError(f"joint_pos/joint_vel shape mismatch: {joint_pos.shape} vs {joint_vel.shape}")

    num_frames, joint_dim = joint_pos.shape

    body_count: int | None = None
    for key, dim in (
      ("body_pos_w", 3),
      ("body_quat_w", 4),
      ("body_lin_vel_w", 3),
      ("body_ang_vel_w", 3),
    ):
      arr = data[key]
      if arr.ndim != 3 or arr.shape[0] != num_frames or arr.shape[2] != dim:
        raise ValueError(f"Expected {key} shape (T, B, {dim}), got {arr.shape}")
      if body_count is None:
        body_count = arr.shape[1]
      elif arr.shape[1] != body_count:
        raise ValueError(
          f"Body count mismatch across arrays: expected B={body_count}, got {key} with B={arr.shape[1]}"
        )

    assert body_count is not None

    joint_names = None
    if "joint_names" in data:
      joint_names = _as_str_tuple(np.asarray(data["joint_names"]))

    body_names = None
    if "body_names" in data:
      body_names = _as_str_tuple(np.asarray(data["body_names"]))

  return MotionNpzMetadata(
    path=path,
    num_frames=num_frames,
    joint_dim=joint_dim,
    body_count=body_count,
    joint_names=joint_names,
    body_names=body_names,
  )


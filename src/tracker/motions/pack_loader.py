from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class MotionPack:
  path: Path
  meta: dict[str, Any]

  joint_pos: torch.Tensor
  joint_vel: torch.Tensor
  body_pos_w: torch.Tensor
  body_quat_w: torch.Tensor
  body_lin_vel_w: torch.Tensor
  body_ang_vel_w: torch.Tensor

  clip_start: torch.Tensor
  clip_len: torch.Tensor
  clip_task_id: torch.Tensor

  splits: dict[str, torch.Tensor]

  @classmethod
  def load(
    cls,
    path: Path,
    *,
    device: str | torch.device = "cpu",
    cast_body_to_f32: bool = True,
  ) -> "MotionPack":
    meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
    if meta.get("schema_version") != 1:
      raise ValueError(f"Unsupported motion pack schema_version={meta.get('schema_version')!r}")

    arrays_dir = path / "arrays"
    splits_dir = path / "splits"

    def _mmap(name: str) -> np.ndarray:
      return np.load(arrays_dir / f"{name}.npy", mmap_mode="r")

    def _load_split(name: str) -> np.ndarray:
      p = splits_dir / f"{name}_clip_ids.npy"
      if not p.exists():
        return np.empty((0,), dtype=np.int64)
      return np.load(p)

    np_joint_pos = _mmap("joint_pos")
    np_joint_vel = _mmap("joint_vel")

    np_body_pos_w = _mmap("body_pos_w")
    np_body_quat_w = _mmap("body_quat_w")
    np_body_lin_vel_w = _mmap("body_lin_vel_w")
    np_body_ang_vel_w = _mmap("body_ang_vel_w")

    np_clip_start = _mmap("clip_start")
    np_clip_len = _mmap("clip_len")
    np_clip_task_id = _mmap("clip_task_id")

    device_t = torch.device(device)

    if device_t.type == "cpu":
      joint_pos = torch.from_numpy(np_joint_pos)
      joint_vel = torch.from_numpy(np_joint_vel)
      body_pos_w = torch.from_numpy(np_body_pos_w)
      body_quat_w = torch.from_numpy(np_body_quat_w)
      body_lin_vel_w = torch.from_numpy(np_body_lin_vel_w)
      body_ang_vel_w = torch.from_numpy(np_body_ang_vel_w)
      clip_start = torch.from_numpy(np_clip_start)
      clip_len = torch.from_numpy(np_clip_len)
      clip_task_id = torch.from_numpy(np_clip_task_id)
    else:
      joint_pos = torch.as_tensor(np_joint_pos, device=device_t)
      joint_vel = torch.as_tensor(np_joint_vel, device=device_t)
      body_pos_w = torch.as_tensor(np_body_pos_w, device=device_t)
      body_quat_w = torch.as_tensor(np_body_quat_w, device=device_t)
      body_lin_vel_w = torch.as_tensor(np_body_lin_vel_w, device=device_t)
      body_ang_vel_w = torch.as_tensor(np_body_ang_vel_w, device=device_t)
      clip_start = torch.as_tensor(np_clip_start, device=device_t)
      clip_len = torch.as_tensor(np_clip_len, device=device_t)
      clip_task_id = torch.as_tensor(np_clip_task_id, device=device_t)

    if cast_body_to_f32:
      body_pos_w = body_pos_w.float()
      body_quat_w = body_quat_w.float()
      body_lin_vel_w = body_lin_vel_w.float()
      body_ang_vel_w = body_ang_vel_w.float()

    splits = {
      "train": torch.as_tensor(_load_split("train"), device=device_t),
      "val": torch.as_tensor(_load_split("val"), device=device_t),
      "test": torch.as_tensor(_load_split("test"), device=device_t),
    }

    return cls(
      path=path,
      meta=meta,
      joint_pos=joint_pos,
      joint_vel=joint_vel,
      body_pos_w=body_pos_w,
      body_quat_w=body_quat_w,
      body_lin_vel_w=body_lin_vel_w,
      body_ang_vel_w=body_ang_vel_w,
      clip_start=clip_start,
      clip_len=clip_len,
      clip_task_id=clip_task_id,
      splits=splits,
    )


from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


def _write_clip(path: Path, *, t: int, j: int, b: int) -> None:
  np.savez(
    path,
    joint_pos=np.zeros((t, j), dtype=np.float32),
    joint_vel=np.zeros((t, j), dtype=np.float32),
    body_pos_w=np.zeros((t, b, 3), dtype=np.float32),
    body_quat_w=np.zeros((t, b, 4), dtype=np.float32),
    body_lin_vel_w=np.zeros((t, b, 3), dtype=np.float32),
    body_ang_vel_w=np.zeros((t, b, 3), dtype=np.float32),
  )


def _make_pack(tmp_path: Path) -> Path:
  from tracker.motions.manifest import Manifest, TaskSpec
  from tracker.motions.pack_writer import write_motion_pack
  from tracker.motions.splits import SplitConfig

  root = tmp_path / "root"
  (root / "a").mkdir(parents=True)
  _write_clip(root / "a" / "0.npz", t=2, j=2, b=1)
  _write_clip(root / "a" / "1.npz", t=3, j=2, b=1)

  manifest = Manifest(schema_version=1, root=root, tasks=[TaskSpec(name="a", include=["a/*.npz"])])
  out_dir = tmp_path / "pack"
  write_motion_pack(manifest, SplitConfig(seed=0, val_ratio=0.2, test_ratio=0.2), out_dir)
  return out_dir


def test_motion_pack_loader_cpu(tmp_path: Path):
  import torch

  from tracker.motions.pack_loader import MotionPack

  pack_dir = _make_pack(tmp_path)
  pack = MotionPack.load(pack_dir, device="cpu", cast_body_to_f32=True)

  assert pack.meta["schema_version"] == 1
  assert pack.joint_pos.shape == (5, 2)
  assert pack.joint_pos.device.type == "cpu"
  assert pack.body_pos_w.dtype == torch.float32
  assert sorted(pack.splits["train"].tolist() + pack.splits["val"].tolist() + pack.splits["test"].tolist()) == [0, 1]


def test_motion_pack_loader_cuda_optional(tmp_path: Path):
  import torch

  if not torch.cuda.is_available():
    pytest.skip("CUDA not available")

  from tracker.motions.pack_loader import MotionPack

  pack_dir = _make_pack(tmp_path)
  pack = MotionPack.load(pack_dir, device="cuda:0", cast_body_to_f32=True)

  assert pack.joint_pos.device.type == "cuda"
  assert pack.body_pos_w.device.type == "cuda"
  assert pack.body_pos_w.dtype == torch.float32


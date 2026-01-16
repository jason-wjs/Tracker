import json
from pathlib import Path

import numpy as np


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


def test_motion_pack_writer_smoke(tmp_path: Path):
  from tracker.motions.manifest import Manifest, TaskSpec
  from tracker.motions.pack_writer import write_motion_pack
  from tracker.motions.splits import SplitConfig

  root = tmp_path / "root"
  (root / "a").mkdir(parents=True)
  (root / "b").mkdir(parents=True)

  _write_clip(root / "a" / "0.npz", t=2, j=2, b=1)
  _write_clip(root / "a" / "1.npz", t=3, j=2, b=1)
  _write_clip(root / "b" / "0.npz", t=1, j=2, b=1)

  manifest = Manifest(
    schema_version=1,
    root=root,
    tasks=[
      TaskSpec(name="a", include=["a/*.npz"]),
      TaskSpec(name="b", include=["b/*.npz"]),
    ],
  )

  out_dir = tmp_path / "pack"
  write_motion_pack(manifest, SplitConfig(seed=0, val_ratio=0.2, test_ratio=0.2), out_dir)

  meta = json.loads((out_dir / "meta.json").read_text())
  assert meta["schema_version"] == 1

  arrays_dir = out_dir / "arrays"
  splits_dir = out_dir / "splits"
  assert arrays_dir.is_dir()
  assert splits_dir.is_dir()

  joint_pos = np.load(arrays_dir / "joint_pos.npy", mmap_mode="r")
  joint_vel = np.load(arrays_dir / "joint_vel.npy", mmap_mode="r")
  assert joint_pos.shape == (6, 2)
  assert joint_vel.shape == (6, 2)
  assert joint_pos.dtype == np.float32
  assert joint_vel.dtype == np.float32

  body_pos_w = np.load(arrays_dir / "body_pos_w.npy", mmap_mode="r")
  assert body_pos_w.shape == (6, 1, 3)
  assert body_pos_w.dtype == np.float16

  clip_start = np.load(arrays_dir / "clip_start.npy", mmap_mode="r")
  clip_len = np.load(arrays_dir / "clip_len.npy", mmap_mode="r")
  clip_task_id = np.load(arrays_dir / "clip_task_id.npy", mmap_mode="r")
  assert clip_start.shape == (3,)
  assert clip_len.shape == (3,)
  assert clip_task_id.shape == (3,)
  assert clip_start.tolist() == [0, 2, 5]
  assert clip_len.tolist() == [2, 3, 1]
  assert sorted(
    np.load(splits_dir / "train_clip_ids.npy").tolist()
    + np.load(splits_dir / "val_clip_ids.npy").tolist()
    + np.load(splits_dir / "test_clip_ids.npy").tolist()
  ) == [0, 1, 2]


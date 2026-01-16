from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from numpy.lib.format import open_memmap

from tracker.motions.manifest import Manifest, TaskSpec, resolve_clips
from tracker.motions.npz_schema import MotionNpzMetadata, load_npz_metadata
from tracker.motions.splits import SplitAssignment, SplitConfig, assign_splits


def _task_table(tasks: list[TaskSpec]) -> list[dict[str, object]]:
  tasks_sorted = sorted(tasks, key=lambda t: t.name)
  return [
    {"name": task.name, "task_id": idx, "weight": float(task.weight)}
    for idx, task in enumerate(tasks_sorted)
  ]


def _ensure_empty_dir(path: Path) -> None:
  if path.exists():
    if not path.is_dir():
      raise ValueError(f"Output path exists and is not a directory: {path}")
    if any(path.iterdir()):
      raise ValueError(f"Output directory is not empty: {path}")
  else:
    path.mkdir(parents=True, exist_ok=True)


def write_motion_pack(
  manifest: Manifest,
  split_cfg: SplitConfig,
  out_dir: Path,
  *,
  source_manifest_path: Path | None = None,
) -> None:
  clips = resolve_clips(manifest)
  if not clips:
    raise ValueError("Manifest resolved to zero motion clips.")

  metadatas: list[MotionNpzMetadata] = []
  total_frames = 0
  ref_meta: MotionNpzMetadata | None = None
  for clip in clips:
    meta = load_npz_metadata(clip.abs_path)
    if ref_meta is None:
      ref_meta = meta
    else:
      if (meta.joint_dim, meta.body_count) != (ref_meta.joint_dim, ref_meta.body_count):
        raise ValueError(
          "Inconsistent clip shapes: "
          f"{clip.relpath} has (J={meta.joint_dim}, B={meta.body_count}) but "
          f"expected (J={ref_meta.joint_dim}, B={ref_meta.body_count})"
        )
      if ref_meta.joint_names is not None and meta.joint_names is not None:
        if meta.joint_names != ref_meta.joint_names:
          raise ValueError(f"Inconsistent joint_names in clip: {clip.relpath}")
      if ref_meta.body_names is not None and meta.body_names is not None:
        if meta.body_names != ref_meta.body_names:
          raise ValueError(f"Inconsistent body_names in clip: {clip.relpath}")

    metadatas.append(meta)
    total_frames += meta.num_frames

  assert ref_meta is not None

  splits: SplitAssignment = assign_splits(clips, split_cfg)

  _ensure_empty_dir(out_dir)
  arrays_dir = out_dir / "arrays"
  splits_dir = out_dir / "splits"
  arrays_dir.mkdir(parents=True, exist_ok=True)
  splits_dir.mkdir(parents=True, exist_ok=True)

  joint_pos = open_memmap(arrays_dir / "joint_pos.npy", mode="w+", dtype=np.float32, shape=(total_frames, ref_meta.joint_dim))
  joint_vel = open_memmap(arrays_dir / "joint_vel.npy", mode="w+", dtype=np.float32, shape=(total_frames, ref_meta.joint_dim))

  body_pos_w = open_memmap(arrays_dir / "body_pos_w.npy", mode="w+", dtype=np.float16, shape=(total_frames, ref_meta.body_count, 3))
  body_quat_w = open_memmap(arrays_dir / "body_quat_w.npy", mode="w+", dtype=np.float16, shape=(total_frames, ref_meta.body_count, 4))
  body_lin_vel_w = open_memmap(arrays_dir / "body_lin_vel_w.npy", mode="w+", dtype=np.float16, shape=(total_frames, ref_meta.body_count, 3))
  body_ang_vel_w = open_memmap(arrays_dir / "body_ang_vel_w.npy", mode="w+", dtype=np.float16, shape=(total_frames, ref_meta.body_count, 3))

  clip_start = open_memmap(arrays_dir / "clip_start.npy", mode="w+", dtype=np.int64, shape=(len(clips),))
  clip_len = open_memmap(arrays_dir / "clip_len.npy", mode="w+", dtype=np.int64, shape=(len(clips),))
  clip_task_id = open_memmap(arrays_dir / "clip_task_id.npy", mode="w+", dtype=np.int64, shape=(len(clips),))

  index_path = out_dir / "index.jsonl"
  offset = 0
  with index_path.open("w", encoding="utf-8") as index_f:
    for clip_id, clip in enumerate(clips):
      meta = metadatas[clip_id]
      t = meta.num_frames

      with np.load(clip.abs_path, allow_pickle=False) as data:
        joint_pos[offset : offset + t] = data["joint_pos"].astype(np.float32, copy=False)
        joint_vel[offset : offset + t] = data["joint_vel"].astype(np.float32, copy=False)

        body_pos_w[offset : offset + t] = data["body_pos_w"].astype(np.float16, copy=False)
        body_quat_w[offset : offset + t] = data["body_quat_w"].astype(np.float16, copy=False)
        body_lin_vel_w[offset : offset + t] = data["body_lin_vel_w"].astype(np.float16, copy=False)
        body_ang_vel_w[offset : offset + t] = data["body_ang_vel_w"].astype(np.float16, copy=False)

      clip_start[clip_id] = offset
      clip_len[clip_id] = t
      clip_task_id[clip_id] = clip.task_id

      index_f.write(
        json.dumps(
          {
            "clip_id": clip_id,
            "task_id": clip.task_id,
            "task_name": clip.task_name,
            "relpath": clip.relpath,
            "clip_start": int(offset),
            "clip_len": int(t),
          },
          sort_keys=True,
        )
        + "\n"
      )
      offset += t

  np.save(splits_dir / "train_clip_ids.npy", np.asarray(splits.train, dtype=np.int64))
  np.save(splits_dir / "val_clip_ids.npy", np.asarray(splits.val, dtype=np.int64))
  np.save(splits_dir / "test_clip_ids.npy", np.asarray(splits.test, dtype=np.int64))

  arrays_info = {
    "joint_pos": {"dtype": "float32", "shape": list(joint_pos.shape)},
    "joint_vel": {"dtype": "float32", "shape": list(joint_vel.shape)},
    "body_pos_w": {"dtype": "float16", "shape": list(body_pos_w.shape)},
    "body_quat_w": {"dtype": "float16", "shape": list(body_quat_w.shape)},
    "body_lin_vel_w": {"dtype": "float16", "shape": list(body_lin_vel_w.shape)},
    "body_ang_vel_w": {"dtype": "float16", "shape": list(body_ang_vel_w.shape)},
    "clip_start": {"dtype": "int64", "shape": list(clip_start.shape)},
    "clip_len": {"dtype": "int64", "shape": list(clip_len.shape)},
    "clip_task_id": {"dtype": "int64", "shape": list(clip_task_id.shape)},
  }

  tasks_table = _task_table(manifest.tasks)
  meta = {
    "schema_version": 1,
    "created_at": datetime.now(timezone.utc).isoformat(),
    "source_manifest": str(source_manifest_path) if source_manifest_path is not None else None,
    "root": str(manifest.root),
    "tasks": tasks_table,
    "counts": {
      "clips": len(clips),
      "frames": int(total_frames),
    },
    "splits": asdict(split_cfg),
    "arrays": arrays_info,
  }
  (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")

  joint_pos.flush()
  joint_vel.flush()
  body_pos_w.flush()
  body_quat_w.flush()
  body_lin_vel_w.flush()
  body_ang_vel_w.flush()
  clip_start.flush()
  clip_len.flush()
  clip_task_id.flush()


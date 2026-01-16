from __future__ import annotations

import hashlib
from dataclasses import dataclass

from tracker.motions.manifest import ResolvedClip


@dataclass(frozen=True)
class SplitConfig:
  seed: int = 0
  val_ratio: float = 0.02
  test_ratio: float = 0.01
  min_val_per_task: int = 1
  min_test_per_task: int = 1


@dataclass(frozen=True)
class SplitAssignment:
  train: list[int]
  val: list[int]
  test: list[int]


def _hash_int(seed: int, key: str) -> int:
  digest = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).digest()
  return int.from_bytes(digest, byteorder="big", signed=False)


def assign_splits(clips: list[ResolvedClip], cfg: SplitConfig) -> SplitAssignment:
  if cfg.val_ratio < 0 or cfg.test_ratio < 0 or cfg.val_ratio + cfg.test_ratio > 1:
    raise ValueError("val_ratio and test_ratio must be >=0 and sum to <=1")

  by_task: dict[int, list[int]] = {}
  for clip_id, clip in enumerate(clips):
    by_task.setdefault(clip.task_id, []).append(clip_id)

  train: set[int] = set()
  val: set[int] = set()
  test: set[int] = set()

  for task_id, clip_ids in sorted(by_task.items(), key=lambda kv: kv[0]):
    clip_ids = sorted(clip_ids, key=lambda cid: clips[cid].relpath)
    n = len(clip_ids)

    items = [
      (cid, _hash_int(cfg.seed, clips[cid].relpath), clips[cid].relpath) for cid in clip_ids
    ]
    items.sort(key=lambda x: (x[1], x[2]))

    if n <= 1:
      train.update(clip_ids)
      continue

    if n == 2:
      val.add(items[0][0])
      train.add(items[1][0])
      continue

    if n == 3:
      test.add(items[0][0])
      val.add(items[1][0])
      train.add(items[2][0])
      continue

    scale = 1 << 256
    test_thresh = int(cfg.test_ratio * scale)
    val_thresh = int((cfg.test_ratio + cfg.val_ratio) * scale)

    task_train: set[int] = set()
    task_val: set[int] = set()
    task_test: set[int] = set()

    for cid, h, _ in items:
      if h < test_thresh:
        task_test.add(cid)
      elif h < val_thresh:
        task_val.add(cid)
      else:
        task_train.add(cid)

    min_val = cfg.min_val_per_task if n >= 2 else 0
    min_test = cfg.min_test_per_task if n >= 3 else 0

    def _promote(dst: set[int], dst_min: int, src_order: list[set[int]]):
      while len(dst) < dst_min:
        promoted = False
        for src in src_order:
          if not src:
            continue
          for cid, _, _ in items:
            if cid in src:
              src.remove(cid)
              dst.add(cid)
              promoted = True
              break
          if promoted:
            break
        if not promoted:
          break

    if min_test > 0:
      _promote(task_test, min_test, [task_train, task_val])
    if min_val > 0:
      _promote(task_val, min_val, [task_train, task_test])

    train.update(task_train)
    val.update(task_val)
    test.update(task_test)

  return SplitAssignment(
    train=sorted(train),
    val=sorted(val),
    test=sorted(test),
  )


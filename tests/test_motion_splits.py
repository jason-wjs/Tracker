from pathlib import Path


def _make_clips(task_id: int, task_name: str, n: int):
  from tracker.motions.manifest import ResolvedClip

  root = Path("/tmp")
  return [
    ResolvedClip(
      task_id=task_id,
      task_name=task_name,
      abs_path=root / f"{task_name}/{i}.npz",
      relpath=f"{task_name}/{i}.npz",
    )
    for i in range(n)
  ]


def test_motion_splits_n1_all_train():
  from tracker.motions.splits import SplitConfig, assign_splits

  clips = _make_clips(0, "walk", 1)
  splits = assign_splits(clips, SplitConfig(seed=0, val_ratio=0.2, test_ratio=0.2))
  assert splits.train == [0]
  assert splits.val == []
  assert splits.test == []


def test_motion_splits_n2_min_val_only():
  from tracker.motions.splits import SplitConfig, assign_splits

  clips = _make_clips(0, "walk", 2)
  splits = assign_splits(clips, SplitConfig(seed=0, val_ratio=0.2, test_ratio=0.2))
  assert len(splits.train) == 1
  assert len(splits.val) == 1
  assert splits.test == []
  assert sorted(splits.train + splits.val) == [0, 1]


def test_motion_splits_n3_min_val_and_test():
  from tracker.motions.splits import SplitConfig, assign_splits

  clips = _make_clips(0, "walk", 3)
  splits = assign_splits(clips, SplitConfig(seed=0, val_ratio=0.0, test_ratio=0.0))
  assert len(splits.train) == 1
  assert len(splits.val) == 1
  assert len(splits.test) == 1
  assert sorted(splits.train + splits.val + splits.test) == [0, 1, 2]


def test_motion_splits_small_ratios_still_respect_minimums():
  from tracker.motions.splits import SplitConfig, assign_splits

  clips = _make_clips(0, "walk", 10)
  splits = assign_splits(clips, SplitConfig(seed=0, val_ratio=0.02, test_ratio=0.01))
  assert len(splits.val) >= 1
  assert len(splits.test) >= 1
  assert sorted(splits.train + splits.val + splits.test) == list(range(10))


def test_motion_splits_deterministic_given_seed():
  from tracker.motions.splits import SplitConfig, assign_splits

  clips = _make_clips(0, "walk", 10)
  cfg = SplitConfig(seed=123, val_ratio=0.2, test_ratio=0.1)
  a = assign_splits(clips, cfg)
  b = assign_splits(clips, cfg)
  assert a == b


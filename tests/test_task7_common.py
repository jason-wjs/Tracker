import pytest


def test_split_task_id():
  from tracker.cli.common import split_task_id

  task_id, remaining = split_task_id(["Tracker-Tracking-Flat-Adam-SP-23", "--foo", "1"])
  assert task_id == "Tracker-Tracking-Flat-Adam-SP-23"
  assert remaining == ["--foo", "1"]


def test_split_task_id_requires_value():
  from tracker.cli.common import split_task_id

  with pytest.raises(SystemExit):
    split_task_id([])


def test_motion_source_mutual_exclusion():
  from tracker.cli.common import ensure_motion_source_exclusive

  with pytest.raises(ValueError):
    ensure_motion_source_exclusive(["--motion-file", "a.npz", "--registry-name", "x"])

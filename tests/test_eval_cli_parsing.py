from pathlib import Path

import pytest


def test_tracker_eval_requires_required_flags(tmp_path: Path):
  from tracker.cli.eval import parse_eval_args

  with pytest.raises(SystemExit):
    parse_eval_args(["Tracker-Tracking-Flat-Adam-SP-23"])

  ckpt = tmp_path / "ckpt.pt"
  ckpt.write_bytes(b"")
  pack = tmp_path / "pack"
  pack.mkdir()

  task_id, cfg = parse_eval_args(
    [
      "Tracker-Tracking-Flat-Adam-SP-23",
      "--checkpoint",
      str(ckpt),
      "--motion-pack",
      str(pack),
    ]
  )
  assert task_id == "Tracker-Tracking-Flat-Adam-SP-23"
  assert cfg.motion_split == "val"


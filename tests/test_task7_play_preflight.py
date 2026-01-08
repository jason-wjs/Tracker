import numpy as np
import pytest


def test_play_preflight_validates_motion_file(tmp_path):
  from tracker.cli.play import preflight

  motion_path = tmp_path / "motion.npz"
  np.savez(motion_path, foo=np.zeros((1,)))

  with pytest.raises(ValueError):
    preflight("Tracker-Tracking-Flat-Adam-SP-23", ["--motion-file", str(motion_path)])

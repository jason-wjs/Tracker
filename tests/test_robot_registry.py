def test_robot_registry_lists_adam_sp_tasks():
  from tracker.robots.registry import list_known_task_ids

  task_ids = set(list_known_task_ids())
  assert "Tracker-Tracking-Flat-Adam-SP-23" in task_ids
  assert "Tracker-Tracking-Flat-Adam-SP-29" in task_ids


def test_cli_validate_motion_dispatches(tmp_path):
  import numpy as np

  motion = tmp_path / "m.npz"
  np.savez(motion, joint_pos=np.zeros((1, 1), dtype=np.float32))

  from tracker.cli.common import validate_motion_for_task

  try:
    validate_motion_for_task("Tracker-Tracking-Flat-Adam-SP-23", motion)
  except Exception as exc:
    assert "Unknown robot for task_id" not in str(exc)


def test_robot_registry_includes_adam_pro_task_ids():
  from tracker.robots.registry import list_known_task_ids

  task_ids = set(list_known_task_ids())
  assert "Tracker-Tracking-Flat-Adam-Pro-23" in task_ids
  assert "Tracker-Tracking-Flat-Adam-Pro-29" in task_ids

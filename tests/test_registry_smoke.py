from mjlab.tasks.registry import list_tasks


def test_import_tracker():
  import tracker  # noqa: F401


def test_bootstrap_registers_adam_sp_task():
  from tracker.integrations.mjlab.bootstrap import bootstrap

  bootstrap()
  tasks = set(list_tasks())
  assert "Tracker-Tracking-Flat-Adam-SP-23" in tasks
  assert "Tracker-Tracking-Flat-Adam-SP-29" in tasks
  assert "Tracker-Tracking-Flat-Adam-Pro-23" in tasks
  assert "Tracker-Tracking-Flat-Adam-Pro-29" in tasks
  assert "Tracker-Tracking-Flat-Adam-Pro-23-No-State-Estimation" in tasks
  assert "Tracker-Tracking-Flat-Adam-Pro-29-No-State-Estimation" in tasks
  assert "Tracker-Teleop-Flat-Adam-Pro-29-No-State-Estimation" in tasks

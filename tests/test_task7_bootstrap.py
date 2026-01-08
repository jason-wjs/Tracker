from mjlab.tasks.registry import list_tasks


def test_bootstrap_registers_tracker_tasks():
  from tracker.integrations.mjlab.bootstrap import bootstrap

  bootstrap()
  assert "Tracker-Tracking-Flat-Adam-SP-23" in list_tasks()

def test_tracker_list_envs_delegates_to_mjlab():
  from tracker.cli.list_envs import list_environments

  count = list_environments(keyword="Adam")
  assert count >= 1


def test_adam_sp_tracking_has_physics_domain_randomization_terms():
  from tracker.integrations.mjlab.bootstrap import bootstrap
  from mjlab.tasks.registry import load_env_cfg

  bootstrap()
  for task_id in (
    "Tracker-Tracking-Flat-Adam-SP-23",
    "Tracker-Tracking-Flat-Adam-SP-29",
  ):
    cfg = load_env_cfg(task_id)
    base_com_ranges = cfg.events["base_com"].params["ranges"]
    assert base_com_ranges[0] == (-0.02, 0.02)
    assert base_com_ranges[1] == (-0.02, 0.02)
    assert base_com_ranges[2] == (-0.02, 0.02)

    for key in (
      "physics_body_mass",
      "physics_body_inertia",
      "physics_dof_damping",
      "physics_dof_frictionloss",
      "link_com",
    ):
      assert key in cfg.events

    assert cfg.events["physics_body_mass"].params["ranges"] == (0.9, 1.1)
    assert cfg.events["physics_body_inertia"].params["ranges"] == (0.9, 1.1)
    assert cfg.events["physics_dof_frictionloss"].params["ranges"] == (0.2, 1.5)


def test_play_keeps_startup_dr_but_disables_push_robot():
  from tracker.integrations.mjlab.bootstrap import bootstrap
  from mjlab.tasks.registry import load_env_cfg

  bootstrap()
  for task_id in (
    "Tracker-Tracking-Flat-Adam-SP-23",
    "Tracker-Tracking-Flat-Adam-SP-29",
  ):
    cfg = load_env_cfg(task_id, play=True)
    assert "push_robot" not in cfg.events
    assert "physics_body_mass" in cfg.events

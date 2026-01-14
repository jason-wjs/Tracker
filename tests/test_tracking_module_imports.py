def test_tracking_config_modules_import():
  import tracker.tasks.tracking.config.env as env_cfg
  import tracker.tasks.tracking.config.patch as patch_tracking_cfg
  import tracker.tasks.tracking.config.rl as rl_cfg

  assert callable(env_cfg.adam_sp_flat_tracking_env_cfg)
  assert callable(env_cfg.adam_sp_29_flat_tracking_env_cfg)
  assert callable(rl_cfg.adam_sp_23_tracking_ppo_runner_cfg)
  assert callable(patch_tracking_cfg.make_flat_tracking_env_cfg_for_robot)

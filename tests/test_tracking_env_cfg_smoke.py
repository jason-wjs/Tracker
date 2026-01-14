def test_env_cfg_builds_for_adam_sp_variants():
  from tracker.tasks.tracking.config.env import (
    adam_sp_29_flat_tracking_env_cfg,
    adam_sp_flat_tracking_env_cfg,
  )

  assert adam_sp_flat_tracking_env_cfg() is not None
  assert adam_sp_29_flat_tracking_env_cfg() is not None

def _contains_string(obj, needle: str) -> bool:
  if obj is None:
    return False
  if isinstance(obj, str):
    return obj == needle
  if isinstance(obj, dict):
    return any(_contains_string(k, needle) or _contains_string(v, needle) for k, v in obj.items())
  if isinstance(obj, (list, tuple, set)):
    return any(_contains_string(v, needle) for v in obj)
  return False


def test_adam_pro_29_cfg_does_not_reference_missing_imu_sensors():
  from tracker.tasks.tracking.config.env import adam_pro_29_flat_tracking_env_cfg

  cfg = adam_pro_29_flat_tracking_env_cfg(has_state_estimation=True)

  missing_scene_sensor_keys = (
    "robot/imu_lin_vel",
    "robot/imu_ang_vel",
    "robot/imu_lin_acc",
    "robot/root_angmom",
  )

  for group in cfg.observations.values():
    for term_cfg in group.terms.values():
      for missing in missing_scene_sensor_keys:
        assert not _contains_string(term_cfg.params, missing), f"Found {missing} in {term_cfg}"


"""Register Adam-SP (23-DoF) tracking tasks with mjlab."""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner

from tracker.tasks.tracking.env_cfg import adam_sp_flat_tracking_env_cfg
from tracker.tasks.tracking.rl_cfg import adam_sp_23_tracking_ppo_runner_cfg

TASK_ID = "Tracker-Tracking-Flat-Adam-SP-23"

register_mjlab_task(
  task_id=TASK_ID,
  env_cfg=adam_sp_flat_tracking_env_cfg(),
  play_env_cfg=adam_sp_flat_tracking_env_cfg(play=True),
  rl_cfg=adam_sp_23_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)

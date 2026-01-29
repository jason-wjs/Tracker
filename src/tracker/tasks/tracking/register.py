"""Register tracker tracking tasks with mjlab.

This module intentionally performs import-time registration so `import tracker`
can plug tasks into `mjlab` via the `mjlab.tasks` entry point.
"""

from __future__ import annotations

from collections.abc import Callable

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner

from tracker.tasks.tracking.config.env import (
  adam_pro_29_flat_tracking_env_cfg,
  adam_pro_flat_tracking_env_cfg,
  adam_sp_29_flat_tracking_env_cfg,
  adam_sp_flat_tracking_env_cfg,
)
from tracker.tasks.tracking.config.rl import (
  adam_pro_23_tracking_ppo_runner_cfg,
  adam_pro_29_tracking_ppo_runner_cfg,
  adam_sp_23_tracking_ppo_runner_cfg,
  adam_sp_29_tracking_ppo_runner_cfg,
)

EnvCfgFn = Callable[[], object]


def _register_tracking_tasks() -> None:
  task_specs: list[tuple[str, EnvCfgFn, EnvCfgFn, object]] = [
    (
      "Tracker-Tracking-Flat-Adam-SP-23",
      adam_sp_flat_tracking_env_cfg,
      lambda: adam_sp_flat_tracking_env_cfg(play=True),
      adam_sp_23_tracking_ppo_runner_cfg(),
    ),
    (
      "Tracker-Tracking-Flat-Adam-SP-29",
      adam_sp_29_flat_tracking_env_cfg,
      lambda: adam_sp_29_flat_tracking_env_cfg(play=True),
      adam_sp_29_tracking_ppo_runner_cfg(),
    ),
    (
      "Tracker-Tracking-Flat-Adam-Pro-23",
      adam_pro_flat_tracking_env_cfg,
      lambda: adam_pro_flat_tracking_env_cfg(play=True),
      adam_pro_23_tracking_ppo_runner_cfg(),
    ),
    (
      "Tracker-Tracking-Flat-Adam-Pro-23-No-State-Estimation",
      lambda: adam_pro_flat_tracking_env_cfg(has_state_estimation=False),
      lambda: adam_pro_flat_tracking_env_cfg(has_state_estimation=False, play=True),
      adam_pro_23_tracking_ppo_runner_cfg(),
    ),
    (
      "Tracker-Tracking-Flat-Adam-Pro-29",
      adam_pro_29_flat_tracking_env_cfg,
      lambda: adam_pro_29_flat_tracking_env_cfg(play=True),
      adam_pro_29_tracking_ppo_runner_cfg(),
    ),
    (
      "Tracker-Tracking-Flat-Adam-Pro-29-No-State-Estimation",
      lambda: adam_pro_29_flat_tracking_env_cfg(has_state_estimation=False),
      lambda: adam_pro_29_flat_tracking_env_cfg(has_state_estimation=False, play=True),
      adam_pro_29_tracking_ppo_runner_cfg(),
    ),
    (
      "Tracker-Teleop-Flat-Adam-Pro-29-No-State-Estimation",
      lambda: adam_pro_29_flat_tracking_env_cfg(has_state_estimation=False),
      lambda: adam_pro_29_flat_tracking_env_cfg(has_state_estimation=False, play=True),
      adam_pro_29_tracking_ppo_runner_cfg(),
    ),
  ]

  for task_id, env_cfg_fn, play_env_cfg_fn, rl_cfg in task_specs:
    register_mjlab_task(
      task_id=task_id,
      env_cfg=env_cfg_fn(),
      play_env_cfg=play_env_cfg_fn(),
      rl_cfg=rl_cfg,
      runner_cls=MotionTrackingOnPolicyRunner,
    )


_register_tracking_tasks()

"""RL configuration for tracker motion-tracking tasks.

This module must remain import-safe: it defines runner config builders, but it
should not register tasks at import time.
"""

from mjlab.rl import (
  RslRlOnPolicyRunnerCfg,
  RslRlPpoActorCriticCfg,
  RslRlPpoAlgorithmCfg,
)


def adam_sp_23_tracking_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create RL runner configuration for Adam-SP (23-DoF) tracking task."""
  return RslRlOnPolicyRunnerCfg(
    policy=RslRlPpoActorCriticCfg(
      init_noise_std=1.0,
      actor_obs_normalization=True,
      critic_obs_normalization=True,
      actor_hidden_dims=(512, 256, 128),
      critic_hidden_dims=(512, 256, 128),
      activation="elu",
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.005,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="adam_sp_tracking_23",
    save_interval=500,
    num_steps_per_env=24,
    max_iterations=30_000,
  )


def adam_sp_29_tracking_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create RL runner configuration for Adam-SP (29-DoF) tracking task."""
  cfg = adam_sp_23_tracking_ppo_runner_cfg()
  cfg.experiment_name = "adam_sp_tracking_29"
  return cfg


def adam_pro_23_tracking_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create RL runner configuration for Adam-Pro (23-DoF) tracking task."""
  cfg = adam_sp_23_tracking_ppo_runner_cfg()
  cfg.experiment_name = "adam_pro_tracking_23"
  return cfg


def adam_pro_29_tracking_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create RL runner configuration for Adam-Pro (29-DoF) tracking task."""
  cfg = adam_sp_23_tracking_ppo_runner_cfg()
  cfg.experiment_name = "adam_pro_tracking_29"
  return cfg


def adam_pro_29_teleop_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create RL runner configuration for Adam-Pro (29-DoF) teleop task."""
  cfg = adam_pro_29_tracking_ppo_runner_cfg()
  cfg.policy.class_name = "TrackerActorCriticTeleop"
  return cfg

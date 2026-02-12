import importlib

import pytest
import torch
from tensordict import TensorDict


def test_teleop_history_encoder_policy_shapes():
  try:
    teleop_mod = importlib.import_module("tracker.rl.teleop_actor_critic")
  except ImportError as exc:
    pytest.fail(f"Missing module tracker.rl.teleop_actor_critic: {exc}")

  assert hasattr(
    teleop_mod, "TrackerActorCriticTeleop"
  ), "Expected TrackerActorCriticTeleop policy class"

  policy_cls = teleop_mod.TrackerActorCriticTeleop

  batch = 16
  num_actions = 29
  motion_obs_dim = 64  # command (58) + motion_anchor_ori_b (6)
  history_len = 8
  per_frame_dim = 3 + 3 * num_actions  # base_ang_vel + joint_pos + joint_vel + actions
  policy_obs_dim = motion_obs_dim + history_len * per_frame_dim

  obs = TensorDict(
    {
      "policy": torch.zeros(batch, policy_obs_dim),
      "critic": torch.zeros(batch, 430),
    },
    batch_size=[batch],
  )
  obs_groups = {"policy": ("policy",), "critic": ("critic",)}

  policy = policy_cls(
    obs,
    obs_groups,
    num_actions,
    actor_obs_normalization=True,
    critic_obs_normalization=True,
    actor_hidden_dims=[512, 256, 128],
    critic_hidden_dims=[512, 256, 128],
    activation="elu",
    init_noise_std=1.0,
  )

  actions = policy.act(obs)
  assert actions.shape == (batch, num_actions)

  actions_infer = policy.act_inference(obs)
  assert actions_infer.shape == (batch, num_actions)

  values = policy.evaluate(obs)
  assert values.shape == (batch, 1)


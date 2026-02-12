from __future__ import annotations

from typing import Any, Literal, cast

import torch
from rsl_rl.env import VecEnv
from tensordict import TensorDict

ResetMode = Literal["repeat"]


class PolicyHistoryVecEnvWrapper(VecEnv):
  """Stack the full `policy` observation over the last K frames.

  This wrapper is intended to be applied selectively (e.g., only for teleop
  tasks) so that other tasks keep their observation interface unchanged.
  """

  def __init__(
    self,
    env: VecEnv,
    *,
    k: int,
    group: str = "policy",
    reset_mode: ResetMode = "repeat",
  ):
    if k <= 0:
      raise ValueError("k must be >= 1")
    if reset_mode != "repeat":
      raise ValueError(f"Unsupported reset_mode: {reset_mode}")

    self.env = env
    self.k = int(k)
    self.group = group
    self.reset_mode = reset_mode

    self.num_envs = env.num_envs
    self.device = cast(torch.device, getattr(env, "device", torch.device("cpu")))
    self.max_episode_length = env.max_episode_length
    self.num_actions = env.num_actions

    self._hist: torch.Tensor | None = None  # (num_envs, k, obs_dim)
    self._obs_dim: int | None = None

  @property
  def action_space(self):  # type: ignore[override]
    return getattr(self.env, "action_space", None)

  @property
  def observation_space(self):  # type: ignore[override]
    return getattr(self.env, "observation_space", None)

  @property
  def unwrapped(self) -> Any:  # match wrappers like `mjlab.rl.RslRlVecEnvWrapper`
    return getattr(self.env, "unwrapped", self.env)

  def seed(self, seed: int = -1) -> int:  # type: ignore[override]
    if hasattr(self.env, "seed"):
      return self.env.seed(seed)  # type: ignore[misc]
    return seed

  def _init_or_validate(self, policy_obs: torch.Tensor) -> None:
    if policy_obs.ndim != 2:
      raise ValueError(f"Expected `{self.group}` obs to be (N, D), got {tuple(policy_obs.shape)}")
    if policy_obs.shape[0] != self.num_envs:
      raise ValueError(
        f"Expected `{self.group}` obs to have N={self.num_envs}, got {policy_obs.shape[0]}"
      )

    obs_dim = int(policy_obs.shape[1])
    if self._hist is None:
      self._obs_dim = obs_dim
      self._hist = policy_obs.unsqueeze(1).repeat(1, self.k, 1).clone()
      return

    assert self._obs_dim is not None
    if obs_dim != self._obs_dim:
      raise ValueError(f"Policy obs dim changed: {obs_dim} vs {self._obs_dim}")

  def _stacked(self) -> torch.Tensor:
    assert self._hist is not None
    assert self._obs_dim is not None
    return self._hist.reshape(self.num_envs, self.k * self._obs_dim)

  def _push(self, policy_obs: torch.Tensor) -> None:
    self._init_or_validate(policy_obs)
    assert self._hist is not None
    # Shift right (older) and write newest at index 0.
    self._hist[:, 1:, :] = self._hist[:, :-1, :]
    self._hist[:, 0, :] = policy_obs

  def _reset_env_hist(self, env_mask: torch.Tensor, policy_obs: torch.Tensor) -> None:
    if not torch.any(env_mask):
      return
    assert self._hist is not None
    # Repeat the current post-reset observation K times.
    self._hist[env_mask] = policy_obs[env_mask].unsqueeze(1).repeat(1, self.k, 1)

  def get_observations(self) -> TensorDict:  # type: ignore[override]
    obs = self.env.get_observations()
    policy_obs = cast(torch.Tensor, obs[self.group])
    if self._hist is None:
      self._init_or_validate(policy_obs)
    else:
      self._push(policy_obs)
    obs[self.group] = self._stacked()
    return obs

  def reset(self) -> tuple[TensorDict, dict]:  # type: ignore[override]
    obs, extras = self.env.reset()
    policy_obs = cast(torch.Tensor, obs[self.group])
    self._hist = None
    self._obs_dim = None
    self._init_or_validate(policy_obs)
    obs[self.group] = self._stacked()
    return obs, extras

  def step(  # type: ignore[override]
    self, actions: torch.Tensor
  ) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
    obs, rew, dones, extras = self.env.step(actions)
    policy_obs = cast(torch.Tensor, obs[self.group])

    if self._hist is None:
      self._init_or_validate(policy_obs)
    else:
      self._push(policy_obs)

    done_mask = dones.to(dtype=torch.bool)
    self._reset_env_hist(done_mask, policy_obs)

    obs[self.group] = self._stacked()
    return obs, rew, dones, extras

  def close(self) -> None:  # type: ignore[override]
    return self.env.close()

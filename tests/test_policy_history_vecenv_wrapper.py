import torch
from tensordict import TensorDict


def _as_frames(stacked: torch.Tensor, *, k: int, obs_dim: int) -> torch.Tensor:
  assert stacked.ndim == 2
  assert stacked.shape[1] == k * obs_dim
  return stacked.view(stacked.shape[0], k, obs_dim)


class _FakeVecEnv:
  """Minimal VecEnv-like stub for testing observation stacking."""

  def __init__(self, *, num_envs: int, obs_dim: int, device: torch.device | None = None):
    self.num_envs = num_envs
    self.obs_dim = obs_dim
    self.device = device or torch.device("cpu")
    self.max_episode_length = 999
    self.num_actions = 1
    self._t = 0

  @property
  def unwrapped(self):  # matches `mjlab.rl.RslRlVecEnvWrapper` API
    return self

  def get_observations(self) -> TensorDict:
    obs = torch.zeros((self.num_envs, self.obs_dim), device=self.device)
    for i in range(self.num_envs):
      obs[i].fill_(float(self._t + i))
    return TensorDict({"policy": obs, "critic": obs.clone()}, batch_size=[self.num_envs])

  def reset(self):
    self._t = 0
    return self.get_observations(), {}

  def step(self, actions: torch.Tensor):
    del actions
    self._t += 1
    obs = self.get_observations()
    rew = torch.zeros((self.num_envs,), device=self.device)
    dones = torch.zeros((self.num_envs,), device=self.device, dtype=torch.long)
    if self._t == 1:
      dones[1] = 1
      # Simulate "post-reset" observation returned by env.step() for done envs.
      obs["policy"][1].fill_(10000.0 + float(self._t))
      obs["critic"][1].fill_(10000.0 + float(self._t))
    extras = {}
    return obs, rew, dones, extras

  def close(self):
    return None


def test_policy_history_stacks_and_resets_done_envs():
  # Import inside the test so we get a clear failing RED until implemented.
  from tracker.rl.policy_history_vecenv import PolicyHistoryVecEnvWrapper  # noqa: PLC0415

  k = 8
  obs_dim = 5
  base_env = _FakeVecEnv(num_envs=3, obs_dim=obs_dim)
  env = PolicyHistoryVecEnvWrapper(base_env, k=k, group="policy", reset_mode="repeat")

  obs = env.get_observations()
  frames = _as_frames(obs["policy"], k=k, obs_dim=obs_dim)
  assert torch.allclose(frames[:, 0], frames[:, 1])
  assert torch.allclose(frames[:, 0], frames[:, -1])

  obs, rew, dones, _extras = env.step(torch.zeros((3, 1)))
  assert rew.shape == (3,)
  assert dones.shape == (3,)

  frames = _as_frames(obs["policy"], k=k, obs_dim=obs_dim)
  # Non-done envs: history shifts (t=1 at slot 0, t=0 at slot 1..)
  assert torch.allclose(frames[0, 0], torch.full((obs_dim,), 1.0))
  assert torch.allclose(frames[0, 1], torch.full((obs_dim,), 0.0))
  assert torch.allclose(frames[2, 0], torch.full((obs_dim,), 3.0))  # t=1 + env_id(2)
  assert torch.allclose(frames[2, 1], torch.full((obs_dim,), 2.0))

  # Done env resets: all frames equal the post-reset obs.
  assert torch.allclose(frames[1, 0], torch.full((obs_dim,), 10001.0))
  assert torch.allclose(frames[1, -1], torch.full((obs_dim,), 10001.0))


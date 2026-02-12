"""Teleoperation actor-critic with TWIST2-style history encoder.

This module is tracker-owned and must keep mjlab untouched.

Why this exists:
  - rsl_rl's default ActorCritic is a plain MLP over a flat observation.
  - TWIST2 encodes raw history with a Conv1D HistoryEncoder before the policy MLP.
  - For teleop multi-clip training, we want that same inductive bias.

Integration detail:
  - rsl_rl.runners.on_policy_runner.OnPolicyRunner constructs policies via:
      actor_critic_class = eval(policy_cfg.pop("class_name"))
    i.e. eval in *that module's globals*.
  - Therefore we must register TrackerActorCriticTeleop into
    `rsl_rl.runners.on_policy_runner` module dict before training.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.networks import EmpiricalNormalization, MLP


class HistoryEncoder(nn.Module):
  """Conv1D encoder over K-step proprio history.

  Input:
    - `history_flat`: (batch, K * input_size) with time-major flattening
      (i.e. contiguous blocks of length `input_size` per timestep).
  Output:
    - latent: (batch, output_size)
  """

  def __init__(
    self,
    activation: nn.Module,
    input_size: int,
    tsteps: int,
    output_size: int,
    channel_size: int = 20,
  ) -> None:
    super().__init__()
    self.tsteps = int(tsteps)
    self.input_size = int(input_size)

    self.encoder = nn.Sequential(
      nn.Linear(self.input_size, 3 * channel_size),
      activation,
    )

    # TWIST2 uses hand-tuned conv stacks for specific history lengths.
    # We implement the same pattern and explicitly support K=8 for tracker teleop.
    if self.tsteps == 8:
      # Input length: 8
      # Conv1: k=4, s=2 => length 3
      # Conv2: k=2, s=1 => length 2
      self.conv = nn.Sequential(
        nn.Conv1d(
          in_channels=3 * channel_size,
          out_channels=2 * channel_size,
          kernel_size=4,
          stride=2,
        ),
        activation,
        nn.Conv1d(
          in_channels=2 * channel_size,
          out_channels=channel_size,
          kernel_size=2,
          stride=1,
        ),
        activation,
        nn.Flatten(),
      )
      conv_out = channel_size * 2
    elif self.tsteps == 10:
      # Mirrors TWIST2 tsteps==10 behavior.
      self.conv = nn.Sequential(
        nn.Conv1d(
          in_channels=3 * channel_size,
          out_channels=2 * channel_size,
          kernel_size=4,
          stride=2,
        ),
        activation,
        nn.Conv1d(
          in_channels=2 * channel_size,
          out_channels=channel_size,
          kernel_size=2,
          stride=1,
        ),
        activation,
        nn.Flatten(),
      )
      conv_out = channel_size * 3
    else:
      # Fallback: preserve the "Conv1D over time" inductive bias but avoid
      # brittle shape assumptions for uncommon K.
      self.conv = nn.Sequential(
        nn.Conv1d(
          in_channels=3 * channel_size,
          out_channels=channel_size,
          kernel_size=min(3, self.tsteps),
          stride=1,
          padding=0,
        ),
        activation,
        nn.AdaptiveAvgPool1d(3),
        nn.Flatten(),
      )
      conv_out = channel_size * 3

    self.linear_out = nn.Linear(conv_out, output_size)

  def forward(self, history_flat: torch.Tensor) -> torch.Tensor:
    batch = history_flat.shape[0]
    expected = self.tsteps * self.input_size
    if history_flat.shape[1] != expected:
      raise ValueError(
        f"HistoryEncoder expected shape (B, {expected}) but got {tuple(history_flat.shape)}"
      )

    # Project per-timestep
    proj = self.encoder(history_flat.reshape(batch * self.tsteps, self.input_size))
    proj = proj.reshape(batch, self.tsteps, -1).permute(0, 2, 1)  # (B, C, T)
    out = self.conv(proj)
    return self.linear_out(out)


class TrackerActorCriticTeleop(nn.Module):
  """ActorCritic variant that encodes teleop history before the actor MLP.

  The teleop policy observation is assumed to be the concatenation:
    - motion context: command (58) + motion_anchor_ori_b (6) = 64
    - history blocks (K frames, flattened per term):
      base_ang_vel (3), joint_pos (A), joint_vel (A), actions (A)
    where A = num_actions.

  We reconstruct a (B, K, 3 + 3A) history tensor from the flattened blocks,
  encode it via HistoryEncoder, and feed:
    [motion_context, current_proprio, history_latent]
  into an actor backbone MLP.
  """

  is_recurrent = False

  def __init__(
    self,
    obs,
    obs_groups,
    num_actions: int,
    actor_obs_normalization: bool = False,
    critic_obs_normalization: bool = False,
    actor_hidden_dims=(256, 256, 256),
    critic_hidden_dims=(256, 256, 256),
    activation: str = "elu",
    init_noise_std: float = 1.0,
    noise_std_type: str = "scalar",
    history_latent_dim: int = 128,
    motion_context_dim: int = 64,
    base_ang_vel_dim: int = 3,
    **kwargs,
  ) -> None:
    if kwargs:
      print(
        "TrackerActorCriticTeleop.__init__ got unexpected arguments, which will be ignored: "
        + str([key for key in kwargs.keys()])
      )
    super().__init__()

    act = self._get_activation(activation)

    self.obs_groups = obs_groups
    self.num_actions = int(num_actions)
    self.motion_context_dim = int(motion_context_dim)
    self.base_ang_vel_dim = int(base_ang_vel_dim)

    # Determine raw observation sizes from provided obs dict.
    self._raw_policy_obs_dim = 0
    for group in obs_groups["policy"]:
      assert (
        len(obs[group].shape) == 2
      ), "TrackerActorCriticTeleop only supports 1D observations."
      self._raw_policy_obs_dim += obs[group].shape[-1]

    self._raw_critic_obs_dim = 0
    for group in obs_groups["critic"]:
      assert (
        len(obs[group].shape) == 2
      ), "TrackerActorCriticTeleop only supports 1D observations."
      self._raw_critic_obs_dim += obs[group].shape[-1]

    # Infer K from total size, assuming teleop layout.
    per_frame_dim = self.base_ang_vel_dim + 3 * self.num_actions
    remaining = self._raw_policy_obs_dim - self.motion_context_dim
    if remaining <= 0 or remaining % per_frame_dim != 0:
      raise ValueError(
        "Cannot infer history length from policy obs dim. "
        f"policy_dim={self._raw_policy_obs_dim}, motion_context_dim={self.motion_context_dim}, "
        f"per_frame_dim={per_frame_dim}."
      )
    self.history_len = remaining // per_frame_dim

    # Actor feature: motion_context + current_proprio + history_latent
    self._current_proprio_dim = per_frame_dim
    self._actor_feature_dim = (
      self.motion_context_dim + self._current_proprio_dim + int(history_latent_dim)
    )

    self.history_encoder = HistoryEncoder(
      activation=act,
      input_size=per_frame_dim,
      tsteps=self.history_len,
      output_size=int(history_latent_dim),
    )
    self.actor = MLP(
      self._actor_feature_dim, self.num_actions, list(actor_hidden_dims), activation
    )

    self.actor_obs_normalization = bool(actor_obs_normalization)
    if self.actor_obs_normalization:
      self.actor_obs_normalizer = EmpiricalNormalization(self._actor_feature_dim)
    else:
      self.actor_obs_normalizer = torch.nn.Identity()
    print(f"Teleop Actor MLP: {self.actor}")

    self.critic = MLP(self._raw_critic_obs_dim, 1, list(critic_hidden_dims), activation)
    self.critic_obs_normalization = bool(critic_obs_normalization)
    if self.critic_obs_normalization:
      self.critic_obs_normalizer = EmpiricalNormalization(self._raw_critic_obs_dim)
    else:
      self.critic_obs_normalizer = torch.nn.Identity()
    print(f"Teleop Critic MLP: {self.critic}")

    # Action noise
    self.noise_std_type = noise_std_type
    if self.noise_std_type == "scalar":
      self.std = nn.Parameter(init_noise_std * torch.ones(self.num_actions))
    elif self.noise_std_type == "log":
      self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(self.num_actions)))
    else:
      raise ValueError(
        f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'"
      )

    self.distribution = None
    Normal.set_default_validate_args(False)

  def reset(self, dones=None):
    return None

  def forward(self):
    raise NotImplementedError

  @property
  def action_mean(self):
    return self.distribution.mean

  @property
  def action_std(self):
    return self.distribution.stddev

  @property
  def entropy(self):
    return self.distribution.entropy().sum(dim=-1)

  def update_distribution(self, actor_features: torch.Tensor) -> None:
    mean = self.actor(actor_features)
    if self.noise_std_type == "scalar":
      std = self.std.expand_as(mean)
    else:
      std = torch.exp(self.log_std).expand_as(mean)
    self.distribution = Normal(mean, std)

  def act(self, obs, **kwargs):
    actor_features = self.get_actor_obs(obs)
    actor_features = self.actor_obs_normalizer(actor_features)
    self.update_distribution(actor_features)
    return self.distribution.sample()

  def act_inference(self, obs):
    actor_features = self.get_actor_obs(obs)
    actor_features = self.actor_obs_normalizer(actor_features)
    return self.actor(actor_features)

  def evaluate(self, obs, **kwargs):
    critic_obs = self.get_critic_obs(obs)
    critic_obs = self.critic_obs_normalizer(critic_obs)
    return self.critic(critic_obs)

  def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
    return self.distribution.log_prob(actions).sum(dim=-1)

  def update_normalization(self, obs) -> None:
    if self.actor_obs_normalization:
      self.actor_obs_normalizer.update(self.get_actor_obs(obs))
    if self.critic_obs_normalization:
      self.critic_obs_normalizer.update(self.get_critic_obs(obs))

  def get_actor_obs(self, obs) -> torch.Tensor:
    raw = self._get_raw_policy_obs(obs)
    return self._build_actor_features(raw)

  def get_critic_obs(self, obs) -> torch.Tensor:
    obs_list = []
    for group in self.obs_groups["critic"]:
      obs_list.append(obs[group])
    return torch.cat(obs_list, dim=-1)

  def _get_raw_policy_obs(self, obs) -> torch.Tensor:
    obs_list = []
    for group in self.obs_groups["policy"]:
      obs_list.append(obs[group])
    return torch.cat(obs_list, dim=-1)

  def _build_actor_features(self, raw_policy_obs: torch.Tensor) -> torch.Tensor:
    # Split motion context first.
    motion_ctx = raw_policy_obs[:, : self.motion_context_dim]
    flat = raw_policy_obs[:, self.motion_context_dim :]

    k = self.history_len
    a = self.num_actions
    i = 0

    # These blocks are flattened (B, K*D) with time-major ordering.
    base_hist = flat[:, i : i + k * self.base_ang_vel_dim].view(
      -1, k, self.base_ang_vel_dim
    )
    i += k * self.base_ang_vel_dim

    joint_pos_hist = flat[:, i : i + k * a].view(-1, k, a)
    i += k * a

    joint_vel_hist = flat[:, i : i + k * a].view(-1, k, a)
    i += k * a

    actions_hist = flat[:, i : i + k * a].view(-1, k, a)
    i += k * a

    if i != flat.shape[1]:
      raise ValueError(
        f"Unexpected teleop policy obs layout. Consumed {i} dims, total={flat.shape[1]}"
      )

    history_seq = torch.cat(
      [base_hist, joint_pos_hist, joint_vel_hist, actions_hist], dim=-1
    )  # (B, K, per_frame_dim)

    current_proprio = history_seq[:, -1, :]
    history_flat = history_seq.reshape(history_seq.shape[0], -1)
    history_latent = self.history_encoder(history_flat)

    return torch.cat([motion_ctx, current_proprio, history_latent], dim=-1)

  @staticmethod
  def _get_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "elu":
      return nn.ELU()
    if name == "relu":
      return nn.ReLU()
    if name == "tanh":
      return nn.Tanh()
    if name == "leaky_relu":
      return nn.LeakyReLU()
    raise ValueError(f"Unknown activation: {name}")


def register_rsl_rl_teleop_policy() -> None:
  """Register tracker teleop policy class into rsl_rl eval scope."""

  import rsl_rl.runners.on_policy_runner as on_policy_runner

  # `eval("TrackerActorCriticTeleop")` runs in `on_policy_runner` module globals.
  on_policy_runner.TrackerActorCriticTeleop = TrackerActorCriticTeleop  # type: ignore[attr-defined]


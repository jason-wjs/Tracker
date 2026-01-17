from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from mjlab.managers import CommandTerm
from mjlab.tasks.tracking.mdp.commands import MotionCommand as MjlabMotionCommand
from mjlab.tasks.tracking.mdp.commands import MotionCommandCfg
from mjlab.utils.lab_api.math import (
  quat_apply,
  quat_error_magnitude,
  quat_from_euler_xyz,
  quat_inv,
  quat_mul,
  sample_uniform,
  yaw_quat,
)

from tracker.motions.pack_loader import MotionPack
from tracker.motions.sampling import HierarchicalSampler

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv


@dataclass(kw_only=True)
class MotionPackCommandCfg(MotionCommandCfg):
  """Tracker extension to select which split to sample from."""

  motion_split: str = "train"


@dataclass(frozen=True)
class _ExportMotion:
  joint_pos: torch.Tensor
  joint_vel: torch.Tensor
  body_pos_w: torch.Tensor
  body_quat_w: torch.Tensor
  body_lin_vel_w: torch.Tensor
  body_ang_vel_w: torch.Tensor
  time_step_total: int


class MultiClipMotionCommand(MjlabMotionCommand):
  """Multi-clip motion command backed by a packed motion directory."""

  cfg: MotionCommandCfg
  _env: ManagerBasedRlEnv

  def __init__(self, cfg: MotionCommandCfg, env: ManagerBasedRlEnv):
    # IMPORTANT: We intentionally do NOT call `MjlabMotionCommand.__init__` because it
    # loads a single-clip MotionLoader from `motion_file`. We instead initialize the
    # `CommandTerm` base directly and provide a compatible `.motion` view.
    CommandTerm.__init__(self, cfg, env)

    self.robot: Entity = env.scene[cfg.entity_name]
    self.robot_anchor_body_index = self.robot.body_names.index(self.cfg.anchor_body_name)
    self.motion_anchor_body_index = self.cfg.body_names.index(self.cfg.anchor_body_name)

    self.body_indexes = torch.tensor(
      self.robot.find_bodies(self.cfg.body_names, preserve_order=True)[0],
      dtype=torch.long,
      device=self.device,
    )

    pack_dir = Path(self.cfg.motion_file)
    self.pack = MotionPack.load(pack_dir, device=self.device, cast_body_to_f32=False)

    split = getattr(cfg, "motion_split", "train")
    if split not in self.pack.splits:
      raise ValueError(f"Unknown motion split: {split!r}")

    split_clip_ids = self.pack.splits[split].to(dtype=torch.int64)
    self.sampler = HierarchicalSampler(
      clip_task_id=self.pack.clip_task_id.to(dtype=torch.int64),
      split_clip_ids=split_clip_ids,
    )

    # mjlab's tracking ONNX exporter expects `env.command_manager.get_term("motion")`
    # to be a `MotionCommand` instance and to expose `.motion.<arrays>`.
    # For multi-clip training, export a representative single clip (clip_id=0).
    self._set_export_motion(pack=self.pack, body_indexes=self.body_indexes, clip_id=0)

    self.clip_id = torch.zeros(self.num_envs, dtype=torch.int64, device=self.device)
    self.task_id = torch.zeros(self.num_envs, dtype=torch.int64, device=self.device)
    self.time_steps = torch.zeros(self.num_envs, dtype=torch.int64, device=self.device)

    self.body_pos_relative_w = torch.zeros(
      self.num_envs, len(cfg.body_names), 3, device=self.device
    )
    self.body_quat_relative_w = torch.zeros(
      self.num_envs, len(cfg.body_names), 4, device=self.device
    )
    self.body_quat_relative_w[:, :, 0] = 1.0

    self.metrics["error_anchor_pos"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_anchor_rot"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_anchor_lin_vel"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_anchor_ang_vel"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_body_pos"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_body_rot"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_body_lin_vel"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_body_ang_vel"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_joint_pos"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_joint_vel"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["sampling_entropy"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["sampling_top1_prob"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["sampling_top1_bin"] = torch.zeros(self.num_envs, device=self.device)

  def _set_export_motion(self, *, pack: MotionPack, body_indexes: torch.Tensor, clip_id: int) -> None:
    """Attach a single-clip `.motion` view for mjlab's ONNX exporter."""
    if clip_id < 0 or clip_id >= int(pack.clip_start.numel()):
      raise ValueError(f"clip_id out of range: {clip_id}")

    start = int(pack.clip_start[clip_id].item())
    length = int(pack.clip_len[clip_id].item())
    end = start + length

    joint_pos = pack.joint_pos[start:end]
    joint_vel = pack.joint_vel[start:end]

    body_pos_w = pack.body_pos_w[start:end].index_select(1, body_indexes).float()
    body_quat_w = pack.body_quat_w[start:end].index_select(1, body_indexes).float()
    body_lin_vel_w = pack.body_lin_vel_w[start:end].index_select(1, body_indexes).float()
    body_ang_vel_w = pack.body_ang_vel_w[start:end].index_select(1, body_indexes).float()

    self.motion = _ExportMotion(  # type: ignore[attr-defined]
      joint_pos=joint_pos.to(dtype=torch.float32),
      joint_vel=joint_vel.to(dtype=torch.float32),
      body_pos_w=body_pos_w,
      body_quat_w=body_quat_w,
      body_lin_vel_w=body_lin_vel_w,
      body_ang_vel_w=body_ang_vel_w,
      time_step_total=length,
    )

  @property
  def command(self) -> torch.Tensor:
    return torch.cat([self.joint_pos, self.joint_vel], dim=1)

  def _frame_index(self) -> torch.Tensor:
    return self.pack.clip_start[self.clip_id] + self.time_steps

  @property
  def joint_pos(self) -> torch.Tensor:
    return self.pack.joint_pos[self._frame_index()]

  @property
  def joint_vel(self) -> torch.Tensor:
    return self.pack.joint_vel[self._frame_index()]

  def _body_frames(self, frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    pos = self.pack.body_pos_w[frames].index_select(1, self.body_indexes).float()
    quat = self.pack.body_quat_w[frames].index_select(1, self.body_indexes).float()
    lin = self.pack.body_lin_vel_w[frames].index_select(1, self.body_indexes).float()
    ang = self.pack.body_ang_vel_w[frames].index_select(1, self.body_indexes).float()
    return pos, quat, lin, ang

  @property
  def body_pos_w(self) -> torch.Tensor:
    frames = self._frame_index()
    pos, _, _, _ = self._body_frames(frames)
    return pos + self._env.scene.env_origins[:, None, :]

  @property
  def body_quat_w(self) -> torch.Tensor:
    frames = self._frame_index()
    _, quat, _, _ = self._body_frames(frames)
    return quat

  @property
  def body_lin_vel_w(self) -> torch.Tensor:
    frames = self._frame_index()
    _, _, lin, _ = self._body_frames(frames)
    return lin

  @property
  def body_ang_vel_w(self) -> torch.Tensor:
    frames = self._frame_index()
    _, _, _, ang = self._body_frames(frames)
    return ang

  @property
  def anchor_pos_w(self) -> torch.Tensor:
    return self.body_pos_w[:, self.motion_anchor_body_index]

  @property
  def anchor_quat_w(self) -> torch.Tensor:
    return self.body_quat_w[:, self.motion_anchor_body_index]

  @property
  def anchor_lin_vel_w(self) -> torch.Tensor:
    return self.body_lin_vel_w[:, self.motion_anchor_body_index]

  @property
  def anchor_ang_vel_w(self) -> torch.Tensor:
    return self.body_ang_vel_w[:, self.motion_anchor_body_index]

  @property
  def robot_joint_pos(self) -> torch.Tensor:
    return self.robot.data.joint_pos

  @property
  def robot_joint_vel(self) -> torch.Tensor:
    return self.robot.data.joint_vel

  @property
  def robot_body_pos_w(self) -> torch.Tensor:
    return self.robot.data.body_link_pos_w[:, self.body_indexes]

  @property
  def robot_body_quat_w(self) -> torch.Tensor:
    return self.robot.data.body_link_quat_w[:, self.body_indexes]

  @property
  def robot_body_lin_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_lin_vel_w[:, self.body_indexes]

  @property
  def robot_body_ang_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_ang_vel_w[:, self.body_indexes]

  @property
  def robot_anchor_pos_w(self) -> torch.Tensor:
    return self.robot.data.body_link_pos_w[:, self.robot_anchor_body_index]

  @property
  def robot_anchor_quat_w(self) -> torch.Tensor:
    return self.robot.data.body_link_quat_w[:, self.robot_anchor_body_index]

  @property
  def robot_anchor_lin_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_lin_vel_w[:, self.robot_anchor_body_index]

  @property
  def robot_anchor_ang_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_ang_vel_w[:, self.robot_anchor_body_index]

  def _update_metrics(self) -> None:
    self.metrics["error_anchor_pos"] = torch.norm(
      self.anchor_pos_w - self.robot_anchor_pos_w, dim=-1
    )
    self.metrics["error_anchor_rot"] = quat_error_magnitude(
      self.anchor_quat_w, self.robot_anchor_quat_w
    )
    self.metrics["error_anchor_lin_vel"] = torch.norm(
      self.anchor_lin_vel_w - self.robot_anchor_lin_vel_w, dim=-1
    )
    self.metrics["error_anchor_ang_vel"] = torch.norm(
      self.anchor_ang_vel_w - self.robot_anchor_ang_vel_w, dim=-1
    )

    self.metrics["error_body_pos"] = torch.norm(
      self.body_pos_relative_w - self.robot_body_pos_w, dim=-1
    ).mean(dim=-1)
    self.metrics["error_body_rot"] = quat_error_magnitude(
      self.body_quat_relative_w, self.robot_body_quat_w
    ).mean(dim=-1)
    self.metrics["error_body_lin_vel"] = torch.norm(
      self.body_lin_vel_w - self.robot_body_lin_vel_w, dim=-1
    ).mean(dim=-1)
    self.metrics["error_body_ang_vel"] = torch.norm(
      self.body_ang_vel_w - self.robot_body_ang_vel_w, dim=-1
    ).mean(dim=-1)

    self.metrics["error_joint_pos"] = torch.norm(
      self.joint_pos - self.robot_joint_pos, dim=-1
    )
    self.metrics["error_joint_vel"] = torch.norm(
      self.joint_vel - self.robot_joint_vel, dim=-1
    )

  def _sample_motion(self, env_ids: torch.Tensor) -> None:
    clip_ids, task_ids, t = self.sampler.sample(
      n=int(env_ids.numel()),
      clip_len=self.pack.clip_len.to(dtype=torch.int64),
    )
    self.clip_id[env_ids] = clip_ids
    self.task_id[env_ids] = task_ids

    if self.cfg.sampling_mode == "start":
      t.zero_()
    self.time_steps[env_ids] = t

    self.metrics["sampling_entropy"][:] = 1.0
    self.metrics["sampling_top1_prob"][:] = 0.0
    self.metrics["sampling_top1_bin"][:] = 0.5

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    if self.cfg.sampling_mode not in ("start", "uniform", "adaptive"):
      raise ValueError(f"Unknown sampling_mode: {self.cfg.sampling_mode!r}")

    self._sample_motion(env_ids)

    root_pos = self.body_pos_w[:, 0].clone()
    root_ori = self.body_quat_w[:, 0].clone()
    root_lin_vel = self.body_lin_vel_w[:, 0].clone()
    root_ang_vel = self.body_ang_vel_w[:, 0].clone()

    range_list = [
      self.cfg.pose_range.get(key, (0.0, 0.0))
      for key in ["x", "y", "z", "roll", "pitch", "yaw"]
    ]
    ranges = torch.tensor(range_list, device=self.device)
    rand_samples = sample_uniform(
      ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device
    )
    root_pos[env_ids] += rand_samples[:, 0:3]
    orientations_delta = quat_from_euler_xyz(
      rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5]
    )
    root_ori[env_ids] = quat_mul(orientations_delta, root_ori[env_ids])

    range_list = [
      self.cfg.velocity_range.get(key, (0.0, 0.0))
      for key in ["x", "y", "z", "roll", "pitch", "yaw"]
    ]
    ranges = torch.tensor(range_list, device=self.device)
    rand_samples = sample_uniform(
      ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device
    )
    root_lin_vel[env_ids] += rand_samples[:, :3]
    root_ang_vel[env_ids] += rand_samples[:, 3:]

    joint_pos = self.joint_pos.clone()
    joint_vel = self.joint_vel.clone()
    joint_pos += sample_uniform(
      lower=self.cfg.joint_position_range[0],
      upper=self.cfg.joint_position_range[1],
      size=joint_pos.shape,
      device=joint_pos.device,  # type: ignore[arg-type]
    )
    soft_joint_pos_limits = self.robot.data.soft_joint_pos_limits[env_ids]
    joint_pos[env_ids] = torch.clip(
      joint_pos[env_ids], soft_joint_pos_limits[:, :, 0], soft_joint_pos_limits[:, :, 1]
    )
    self.robot.write_joint_state_to_sim(
      joint_pos[env_ids], joint_vel[env_ids], env_ids=env_ids
    )

    root_state = torch.cat(
      [
        root_pos[env_ids],
        root_ori[env_ids],
        root_lin_vel[env_ids],
        root_ang_vel[env_ids],
      ],
      dim=-1,
    )
    self.robot.write_root_state_to_sim(root_state, env_ids=env_ids)
    self.robot.clear_state(env_ids=env_ids)

  def _update_command(self) -> None:
    self.time_steps += 1
    clip_lens = self.pack.clip_len[self.clip_id].to(dtype=torch.int64)
    env_ids = torch.where(self.time_steps >= clip_lens)[0]
    if env_ids.numel() > 0:
      self._resample_command(env_ids)

    anchor_pos_w_repeat = self.anchor_pos_w[:, None, :].repeat(
      1, len(self.cfg.body_names), 1
    )
    anchor_quat_w_repeat = self.anchor_quat_w[:, None, :].repeat(
      1, len(self.cfg.body_names), 1
    )
    robot_anchor_pos_w_repeat = self.robot_anchor_pos_w[:, None, :].repeat(
      1, len(self.cfg.body_names), 1
    )
    robot_anchor_quat_w_repeat = self.robot_anchor_quat_w[:, None, :].repeat(
      1, len(self.cfg.body_names), 1
    )

    delta_pos_w = robot_anchor_pos_w_repeat
    delta_pos_w[..., 2] = anchor_pos_w_repeat[..., 2]
    delta_ori_w = yaw_quat(
      quat_mul(robot_anchor_quat_w_repeat, quat_inv(anchor_quat_w_repeat))
    )

    self.body_quat_relative_w = quat_mul(delta_ori_w, self.body_quat_w)
    self.body_pos_relative_w = delta_pos_w + quat_apply(
      delta_ori_w, self.body_pos_w - anchor_pos_w_repeat
    )

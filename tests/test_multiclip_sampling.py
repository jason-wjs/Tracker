from __future__ import annotations

import torch


def test_hierarchical_sampler_task_uniform_and_clip_length_weighted():
  from tracker.motions.sampling import HierarchicalSampler

  torch.manual_seed(0)

  # 10 clips: task 0 has 2 clips, task 1 has 8 clips.
  clip_task_id = torch.tensor([0, 0] + [1] * 8, dtype=torch.int64)
  split_clip_ids = torch.arange(10, dtype=torch.int64)
  # Make clip 1 in task 0 three times longer to test length-weighted sampling.
  clip_len = torch.tensor([100, 300] + [100] * 8, dtype=torch.int64)

  sampler = HierarchicalSampler(clip_task_id=clip_task_id, split_clip_ids=split_clip_ids)
  clip_ids, task_ids, t = sampler.sample(n=10_000, clip_len=clip_len)

  assert clip_ids.shape == (10_000,)
  assert task_ids.shape == (10_000,)
  assert t.shape == (10_000,)

  task_counts = torch.bincount(task_ids, minlength=2).float()
  task_probs = task_counts / task_counts.sum()
  assert 0.45 <= task_probs[0].item() <= 0.55
  assert 0.45 <= task_probs[1].item() <= 0.55

  # Clip length-weighted within each task.
  clip_counts = torch.bincount(clip_ids, minlength=10).float()
  task0_counts = clip_counts[:2]
  ratio = (task0_counts[1] / task0_counts[0]).item()
  assert 2.5 <= ratio <= 3.5

  # Task 1 has equal-length clips -> roughly uniform.
  task1_counts = clip_counts[2:]
  assert (task1_counts.max() - task1_counts.min()).item() / task1_counts.mean().item() < 0.30

  # Time sampling within clip bounds.
  sampled_lens = clip_len[clip_ids]
  assert torch.all(t >= 0)
  assert torch.all(t < sampled_lens)


def test_hierarchical_sampler_respects_clip_multiplier_within_task():
  from tracker.motions.sampling import HierarchicalSampler

  torch.manual_seed(0)

  clip_task_id = torch.tensor([0, 0, 0, 0], dtype=torch.int64)
  split_clip_ids = torch.arange(4, dtype=torch.int64)
  clip_len = torch.tensor([100, 100, 100, 100], dtype=torch.int64)
  # Make clip 2 "hard" via multiplier.
  clip_multiplier = torch.tensor([1.0, 1.0, 10.0, 1.0], dtype=torch.float32)

  sampler = HierarchicalSampler(clip_task_id=clip_task_id, split_clip_ids=split_clip_ids)
  clip_ids, _, _ = sampler.sample(n=20_000, clip_len=clip_len, clip_multiplier=clip_multiplier)

  counts = torch.bincount(clip_ids, minlength=4).float()
  assert counts[2] > counts.mean() * 2.0


def test_multiclip_command_adaptive_sampling_sets_metrics():
  from types import SimpleNamespace

  from tracker.motions.sampling import HierarchicalSampler
  from tracker.tasks.tracking.multiclip_command import MultiClipMotionCommand

  torch.manual_seed(0)

  # Build a minimal MultiClipMotionCommand instance without full env.
  cmd = MultiClipMotionCommand.__new__(MultiClipMotionCommand)
  cmd.cfg = SimpleNamespace(sampling_mode="adaptive")

  clip_task_id = torch.tensor([0, 0], dtype=torch.int64)
  split_clip_ids = torch.tensor([0, 1], dtype=torch.int64)
  cmd.sampler = HierarchicalSampler(clip_task_id=clip_task_id, split_clip_ids=split_clip_ids)
  cmd.pack = SimpleNamespace(clip_len=torch.tensor([10, 20], dtype=torch.int64))

  num_envs = 4
  cmd.clip_id = torch.zeros(num_envs, dtype=torch.int64)
  cmd.task_id = torch.zeros(num_envs, dtype=torch.int64)
  cmd.time_steps = torch.zeros(num_envs, dtype=torch.int64)
  cmd.metrics = {
    "sampling_entropy": torch.zeros(num_envs, dtype=torch.float32),
    "sampling_top1_prob": torch.zeros(num_envs, dtype=torch.float32),
    "sampling_top1_bin": torch.zeros(num_envs, dtype=torch.float32),
  }

  class DummyAdaptive:
    def sample_time(self, *, clip_ids, generator=None, return_metrics=False):
      assert return_metrics is True
      n = clip_ids.numel()
      t = torch.full((n,), 7, dtype=torch.int64)
      entropy = torch.full((n,), 0.3)
      top1_prob = torch.full((n,), 0.8)
      top1_bin = torch.full((n,), 0.6)
      return t, entropy, top1_prob, top1_bin

  cmd.adaptive_sampler = DummyAdaptive()

  env_ids = torch.arange(num_envs, dtype=torch.int64)
  cmd._sample_motion(env_ids)

  assert torch.all(cmd.time_steps == 7)
  assert torch.allclose(cmd.metrics["sampling_entropy"], torch.full((num_envs,), 0.3))
  assert torch.allclose(cmd.metrics["sampling_top1_prob"], torch.full((num_envs,), 0.8))
  assert torch.allclose(cmd.metrics["sampling_top1_bin"], torch.full((num_envs,), 0.6))

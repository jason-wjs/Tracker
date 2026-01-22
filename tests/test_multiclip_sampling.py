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

from __future__ import annotations

import torch


def test_hierarchical_sampler_task_uniform_and_clip_uniform():
  from tracker.motions.sampling import HierarchicalSampler

  torch.manual_seed(0)

  # 10 clips: task 0 has 2 clips, task 1 has 8 clips.
  clip_task_id = torch.tensor([0, 0] + [1] * 8, dtype=torch.int64)
  split_clip_ids = torch.arange(10, dtype=torch.int64)
  clip_len = torch.tensor([100] * 10, dtype=torch.int64)

  sampler = HierarchicalSampler(clip_task_id=clip_task_id, split_clip_ids=split_clip_ids)
  clip_ids, task_ids, t = sampler.sample(n=10_000, clip_len=clip_len)

  assert clip_ids.shape == (10_000,)
  assert task_ids.shape == (10_000,)
  assert t.shape == (10_000,)

  task_counts = torch.bincount(task_ids, minlength=2).float()
  task_probs = task_counts / task_counts.sum()
  assert 0.45 <= task_probs[0].item() <= 0.55
  assert 0.45 <= task_probs[1].item() <= 0.55

  # Clip uniform within each task.
  clip_counts = torch.bincount(clip_ids, minlength=10).float()
  task0 = clip_counts[:2]
  task1 = clip_counts[2:]
  assert (task0.max() - task0.min()).item() / task0.mean().item() < 0.15
  assert (task1.max() - task1.min()).item() / task1.mean().item() < 0.30

  # Time sampling within clip bounds.
  sampled_lens = clip_len[clip_ids]
  assert torch.all(t >= 0)
  assert torch.all(t < sampled_lens)


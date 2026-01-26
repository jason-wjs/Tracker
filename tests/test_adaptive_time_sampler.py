import torch


def test_adaptive_sampler_biases_towards_failed_bins():
  # One clip with 1000 frames, step_dt=0.02 => ~50 Hz.
  # Expect ~21 bins for "per-second" style binning.
  from tracker.motions.adaptive_time_sampler import AdaptiveTimeBinSampler

  clip_len = torch.tensor([1000], dtype=torch.int64)
  sampler = AdaptiveTimeBinSampler(
    clip_len=clip_len,
    step_dt=0.02,
    adaptive_kernel_size=1,
    adaptive_lambda=0.8,
    adaptive_uniform_ratio=0.1,
    adaptive_alpha=0.001,
  )

  # Simulate repeated failures near the end of the clip.
  clip_ids = torch.zeros((128,), dtype=torch.int64)
  time_steps = torch.full((128,), 900, dtype=torch.int64)
  terminated = torch.ones((128,), dtype=torch.bool)
  sampler.update_failures(
    clip_ids=clip_ids, time_steps=time_steps, terminated=terminated
  )

  g = torch.Generator().manual_seed(0)
  sampled_t = sampler.sample_time(
    clip_ids=torch.zeros((2048,), dtype=torch.int64), generator=g
  )

  # Heuristic: after seeding failures late, the sampler should prefer late times.
  assert sampled_t.float().mean().item() > 450


def test_adaptive_sampler_respects_clip_length():
  from tracker.motions.adaptive_time_sampler import AdaptiveTimeBinSampler

  clip_len = torch.tensor([10, 25], dtype=torch.int64)
  sampler = AdaptiveTimeBinSampler(
    clip_len=clip_len,
    step_dt=0.02,
    adaptive_kernel_size=1,
    adaptive_lambda=0.8,
    adaptive_uniform_ratio=0.1,
    adaptive_alpha=0.001,
  )

  g = torch.Generator().manual_seed(0)
  clip_ids = torch.tensor([0, 1, 1, 0], dtype=torch.int64)
  t = sampler.sample_time(clip_ids=clip_ids, generator=g)
  assert torch.all(t >= 0)
  assert t[0].item() < 10
  assert t[1].item() < 25

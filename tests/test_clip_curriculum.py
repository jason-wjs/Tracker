import torch


def test_clip_hardness_is_row_max():
  from tracker.motions.clip_curriculum import clip_hardness_from_bin_failed_count

  # 3 clips, 4 bins
  b = torch.tensor(
    [
      [0.0, 1.0, 0.0, 0.0],  # max=1
      [2.0, 0.5, 0.0, 0.0],  # max=2
      [0.0, 0.0, 0.0, 0.0],  # max=0
    ],
    dtype=torch.float32,
  )
  h = clip_hardness_from_bin_failed_count(b)
  assert torch.allclose(h, torch.tensor([1.0, 2.0, 0.0]))


def test_hardness_to_multiplier_is_monotonic_and_clamped():
  from tracker.motions.clip_curriculum import hardness_to_multiplier

  h = torch.tensor([0.0, 0.1, 1.0, 10.0], dtype=torch.float32)
  tau = torch.tensor(0.1, dtype=torch.float32)
  m = hardness_to_multiplier(h, tau=tau, strength=2.0, m_max=3.0)
  assert torch.all(m >= 1.0)
  assert torch.all(m <= 3.0)
  # monotonic for increasing h
  assert torch.all(m[1:] >= m[:-1])


def test_curriculum_mix_is_bonus_only_floor():
  from tracker.motions.clip_curriculum import mix_baseline_and_curriculum

  # Baseline weights
  w_base = torch.tensor([1.0, 1.0, 1.0])
  # Curriculum boosts clip 0
  w_curr = torch.tensor([10.0, 1.0, 1.0])
  w = mix_baseline_and_curriculum(w_base=w_base, w_curr=w_curr, mix=0.5)
  # Floor: nothing drops below baseline-scaled contribution
  assert torch.all(w > 0.0)
  assert w[0] > w[1] and w[0] > w[2]


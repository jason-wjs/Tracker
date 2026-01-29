from __future__ import annotations

import torch


def clip_hardness_from_bin_failed_count(bin_failed_count: torch.Tensor) -> torch.Tensor:
  if bin_failed_count.ndim != 2:
    raise ValueError("bin_failed_count must be 2D (num_clips, num_bins)")
  return bin_failed_count.max(dim=1).values


def hardness_to_multiplier(
  h: torch.Tensor,
  *,
  tau: torch.Tensor,
  strength: float,
  m_max: float,
  eps: float = 1e-6,
) -> torch.Tensor:
  if strength < 0:
    raise ValueError("strength must be >= 0")
  if m_max < 1.0:
    raise ValueError("m_max must be >= 1")
  h_norm = h / (h + tau + eps)
  m = 1.0 + strength * h_norm
  return torch.clamp(m, min=1.0, max=m_max)


def mix_baseline_and_curriculum(*, w_base: torch.Tensor, w_curr: torch.Tensor, mix: float) -> torch.Tensor:
  if w_base.shape != w_curr.shape:
    raise ValueError("w_base and w_curr must have the same shape")
  if not (0.0 <= mix <= 1.0):
    raise ValueError("mix must be in [0, 1]")
  return (1.0 - mix) * w_base + mix * w_curr


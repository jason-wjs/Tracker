from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F


@dataclass
class AdaptiveTimeBinSampler:
  """Per-clip adaptive time-bin sampler (mjlab-style).

  Maintains a failure histogram per clip, and samples time bins proportionally to
  failure frequency with optional kernel smoothing and a uniform mixing ratio.
  """

  clip_len: torch.Tensor
  step_dt: float
  adaptive_kernel_size: int
  adaptive_lambda: float
  adaptive_uniform_ratio: float
  adaptive_alpha: float

  def __post_init__(self) -> None:
    if self.clip_len.ndim != 1:
      raise ValueError("clip_len must be 1D (num_clips,)")
    if self.clip_len.dtype != torch.int64:
      raise ValueError("clip_len must be int64")
    if self.step_dt <= 0:
      raise ValueError("step_dt must be > 0")
    if self.adaptive_kernel_size < 1:
      raise ValueError("adaptive_kernel_size must be >= 1")
    if not (0.0 <= self.adaptive_uniform_ratio <= 1.0):
      raise ValueError("adaptive_uniform_ratio must be in [0, 1]")
    if not (0.0 < self.adaptive_lambda <= 1.0):
      raise ValueError("adaptive_lambda must be in (0, 1]")
    if not (0.0 < self.adaptive_alpha <= 1.0):
      raise ValueError("adaptive_alpha must be in (0, 1]")

    device = self.clip_len.device
    frames_per_bin = max(int(1.0 / self.step_dt), 1)
    self.bin_count = self.clip_len // frames_per_bin + 1
    if torch.any(self.clip_len <= 0):
      raise ValueError("clip_len must be > 0 for all clips")

    max_bins = int(self.bin_count.max().item())
    self.bin_failed_count = torch.zeros(
      (self.clip_len.numel(), max_bins), dtype=torch.float32, device=device
    )
    self.current_bin_failed = torch.zeros_like(self.bin_failed_count)

    kernel = torch.tensor(
      [self.adaptive_lambda**i for i in range(self.adaptive_kernel_size)],
      device=device,
      dtype=torch.float32,
    )
    self.kernel = kernel / kernel.sum()

  def update_failures(
    self,
    *,
    clip_ids: torch.Tensor,
    time_steps: torch.Tensor,
    terminated: torch.Tensor,
  ) -> None:
    if clip_ids.numel() == 0:
      return
    if clip_ids.shape != time_steps.shape:
      raise ValueError("clip_ids and time_steps must have the same shape")
    if terminated.shape != clip_ids.shape:
      raise ValueError("terminated must match clip_ids shape")

    mask = terminated
    if not torch.any(mask):
      return

    clip_ids = clip_ids[mask].to(dtype=torch.int64)
    time_steps = time_steps[mask].to(dtype=torch.int64)

    for clip_id in torch.unique(clip_ids).tolist():
      sel = clip_ids == clip_id
      steps = time_steps[sel]
      bin_count = int(self.bin_count[clip_id].item())
      clip_len = int(self.clip_len[clip_id].item())
      if bin_count <= 0 or clip_len <= 0:
        continue

      bins = (steps * bin_count) // clip_len
      bins = torch.clamp(bins, 0, bin_count - 1)
      counts = torch.bincount(bins, minlength=bin_count).to(dtype=torch.float32)
      self.current_bin_failed[clip_id, :bin_count] += counts

  def update_ema(self) -> None:
    self.bin_failed_count = (
      self.adaptive_alpha * self.current_bin_failed
      + (1.0 - self.adaptive_alpha) * self.bin_failed_count
    )
    self.current_bin_failed.zero_()

  def _sampling_probabilities(self, clip_id: int) -> tuple[torch.Tensor, float, float, float]:
    bin_count = int(self.bin_count[clip_id].item())
    if bin_count <= 0:
      raise ValueError("bin_count must be > 0")

    probs = self.bin_failed_count[clip_id, :bin_count].clone()
    probs += self.adaptive_uniform_ratio / float(bin_count)

    if self.adaptive_kernel_size > 1:
      probs = F.pad(probs.view(1, 1, -1), (0, self.adaptive_kernel_size - 1), mode="replicate")
      probs = F.conv1d(probs, self.kernel.view(1, 1, -1)).view(-1)

    total = probs.sum()
    if total <= 0:
      probs = torch.full_like(probs, 1.0 / float(bin_count))
    else:
      probs = probs / total

    if bin_count == 1:
      return probs, 0.0, 1.0, 0.0

    H = -(probs * (probs + 1e-12).log()).sum()
    H_norm = (H / math.log(bin_count)).item()
    pmax, imax = probs.max(dim=0)
    top1_bin = (imax.float() / float(bin_count)).item()
    return probs, H_norm, pmax.item(), top1_bin

  def sample_time(
    self,
    *,
    clip_ids: torch.Tensor,
    generator: torch.Generator | None = None,
    return_metrics: bool = False,
  ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if clip_ids.ndim != 1:
      raise ValueError("clip_ids must be 1D")
    if clip_ids.dtype != torch.int64:
      clip_ids = clip_ids.to(dtype=torch.int64)

    device = clip_ids.device
    num_samples = clip_ids.numel()
    time_steps = torch.zeros(num_samples, dtype=torch.int64, device=device)
    entropy = torch.zeros(num_samples, dtype=torch.float32, device=device)
    top1_prob = torch.zeros(num_samples, dtype=torch.float32, device=device)
    top1_bin = torch.zeros(num_samples, dtype=torch.float32, device=device)

    for clip_id in torch.unique(clip_ids).tolist():
      sel = (clip_ids == clip_id).nonzero().flatten()
      if sel.numel() == 0:
        continue

      probs, H_norm, pmax, bin_norm = self._sampling_probabilities(clip_id)
      sampled_bins = torch.multinomial(
        probs, num_samples=sel.numel(), replacement=True, generator=generator
      )

      bin_count = int(self.bin_count[clip_id].item())
      clip_len = int(self.clip_len[clip_id].item())

      if clip_len <= 1:
        t = torch.zeros(sel.numel(), dtype=torch.int64, device=device)
      else:
        rand = torch.rand((sel.numel(),), device=device, generator=generator)
        t = ((sampled_bins.to(dtype=torch.float32) + rand) / bin_count * (clip_len - 1)).to(
          dtype=torch.int64
        )

      time_steps[sel] = t
      entropy[sel] = H_norm
      top1_prob[sel] = pmax
      top1_bin[sel] = bin_norm

    if return_metrics:
      return time_steps, entropy, top1_prob, top1_bin
    return time_steps

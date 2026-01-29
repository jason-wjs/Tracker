from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class HierarchicalSampler:
  """Hierarchical sampling: task → clip → time.

  - Tasks are sampled uniformly across tasks present in the chosen split.
  - Clips are sampled proportional to clip length within the selected task (frame-uniform).
  - Time is sampled uniformly within the selected clip length.
  """

  clip_task_id: torch.Tensor
  split_clip_ids: torch.Tensor

  def __post_init__(self) -> None:
    if self.clip_task_id.ndim != 1:
      raise ValueError("clip_task_id must be 1D (num_clips,)")
    if self.split_clip_ids.ndim != 1:
      raise ValueError("split_clip_ids must be 1D (num_split_clips,)")
    if self.clip_task_id.dtype != torch.int64:
      raise ValueError("clip_task_id must be int64")
    if self.split_clip_ids.dtype != torch.int64:
      raise ValueError("split_clip_ids must be int64")

  def _task_ids(self) -> torch.Tensor:
    task_ids = torch.unique(self.clip_task_id[self.split_clip_ids], sorted=True)
    if task_ids.numel() == 0:
      raise ValueError("No clips available in split.")
    return task_ids

  def sample(
    self,
    *,
    n: int,
    clip_len: torch.Tensor,
    clip_multiplier: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if n < 0:
      raise ValueError("n must be >= 0")
    if clip_len.ndim != 1 or clip_len.dtype != torch.int64:
      raise ValueError("clip_len must be int64 1D (num_clips,)")
    if clip_multiplier is not None:
      if clip_multiplier.ndim != 1:
        raise ValueError("clip_multiplier must be 1D (num_clips,)")
      if clip_multiplier.shape != clip_len.shape:
        raise ValueError("clip_multiplier must have shape (num_clips,)")
      if not clip_multiplier.is_floating_point():
        raise ValueError("clip_multiplier must be a floating tensor")
      if torch.any(clip_multiplier < 0):
        raise ValueError("clip_multiplier must be >= 0")

    device = self.split_clip_ids.device
    task_ids = self._task_ids().to(device=device)

    task_indices = torch.randint(
      0, task_ids.numel(), (n,), device=device, generator=generator
    )
    sampled_task_ids = task_ids[task_indices]

    sampled_clip_ids = torch.empty((n,), dtype=torch.int64, device=device)
    for task_id in task_ids.tolist():
      env_sel = (sampled_task_ids == task_id).nonzero().flatten()
      if env_sel.numel() == 0:
        continue
      candidates = self.split_clip_ids[
        (self.clip_task_id[self.split_clip_ids] == task_id).nonzero().flatten()
      ]
      if candidates.numel() == 0:
        raise RuntimeError(f"Task {task_id} has no clips in split (unexpected).")
      candidate_lens = clip_len[candidates].to(dtype=torch.float32)
      if torch.any(candidate_lens <= 0):
        raise ValueError(f"Found non-positive clip_len for task_id={task_id}")

      weights = candidate_lens
      if clip_multiplier is not None:
        weights = candidate_lens * clip_multiplier[candidates].to(dtype=torch.float32)
        if weights.sum() <= 0:
          raise ValueError(f"All sampling weights are zero for task_id={task_id}")

      pick = torch.multinomial(
        weights,
        num_samples=env_sel.numel(),
        replacement=True,
        generator=generator,
      )
      sampled_clip_ids[env_sel] = candidates[pick]

    sampled_lens = clip_len[sampled_clip_ids]
    t = (
      torch.rand((n,), device=device, generator=generator)
      * sampled_lens.to(dtype=torch.float32)
    ).to(dtype=torch.int64)
    return sampled_clip_ids, sampled_task_ids, t

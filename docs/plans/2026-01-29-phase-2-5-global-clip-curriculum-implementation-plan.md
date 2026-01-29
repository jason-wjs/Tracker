# Phase 2.5: Global Clip Curriculum + Teleop Task Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Improve multi-clip motion-pack training by adding a **global clip-wise curriculum** (EMA failure-based, within-task only) while keeping the existing **per-clip adaptive time-bin sampling**, and register a dedicated **teleop training task** for isolation/future defaults.

**Architecture:** Keep the existing two-level sampling design:
1) **Task → clip selection** (global curriculum, within each `clip_task_id` bucket).
2) **Time selection inside each clip** (existing `AdaptiveTimeBinSampler`, mjlab-style).

Global curriculum is **bonus-only** (never reduces below the baseline length-weighted sampler) and includes a **floor** to avoid clip starvation.

**Tech Stack:** Python 3.10, PyTorch (CUDA), `mjlab` (do not modify), tracker-owned motion pack loaders/samplers.

---

## Constraints / Non-goals

- **Do not modify** `mjlab/` (hard constraint).
- Keep training horizon `episode_length_s` unchanged (≈10s default); play/eval can run full clips via play env cfg override.
- Curriculum only changes **clip choice** within each task bucket; task choice remains uniform.
- Keep all “undetermined” hyperparameters as **placeholders** until we have A/B results.

---

## Proposed Behavior (what “done” means)

- When training with `--motion-pack`, clip sampling uses a curriculum derived from **EMA’d per-bin failure counts** (`AdaptiveTimeBinSampler.bin_failed_count`).
- Curriculum uses **EMA’d** counts (stable), not `current_bin_failed` (reactive/too spiky).
- Curriculum is **enabled by default** for motion-pack training, with a CLI/CFG override to disable for baselines.
- A new teleop task ID exists and is registered (initially reusing the Adam-Pro-29 No-SE tracking cfg).
- Tests cover: curriculum weight behavior, “no starvation” floor, and registry includes teleop task.

---

## Curriculum Design (locked choices + placeholders)

### Clip hardness signal (locked)
For each clip:
- `hardness_clip = max(bin_failed_count[clip, :])`

Rationale: a clip remains “hard” if it contains *any* hard 10s window, even if other windows are easy.

### Mapping hardness → curriculum multiplier (recommended defaults)
We need a mapping that works for both:
- small datasets (e.g., ~272 clips), where per-clip failure counts are relatively dense, and
- very large datasets (10,000+ clips), where per-clip failure counts are sparse and absolute scales vary.

Use a **per-task dynamic normalization** (no hand-tuned absolute `tau`):

- Let `h` be `hardness_clip` for the candidate clips of a **single task bucket**.
- Let `tau = tau_scale * mean(h)` with `tau_scale = 1.0` (default).
- Let `h_norm = h / (h + tau + eps)`, with `eps = 1e-6`.
- Let `m = clamp(1 + strength * h_norm, min=1.0, max=m_max)`,
  where:
  - `strength = 2.0` (default)
  - `m_max = 3.0` (default)

Properties:
- **Bonus-only**: `m ≥ 1.0` always.
- **Scale-robust**: if failure counts are tiny (large dataset), `tau` shrinks accordingly.
- **Saturating**: very hard clips cap at `m_max`.

### Bonus-only + floor (recommended default)
Within a task bucket, compute baseline weights and curriculum weights:
- Baseline: `w_base ∝ clip_len`
- Curriculum: `w_curr ∝ clip_len * m(hardness)`
- Mix with baseline floor:
  - `w = (1 - mix) * w_base + mix * w_curr`, where:
    - `mix = 1.0` (default; apply the curriculum fully)

This guarantees every clip keeps a non-zero probability (no starvation), while hard clips get extra mass.

---

# Implementation Tasks

### Task 1: Add curriculum math utilities + unit tests

**Files:**
- Create: `src/tracker/motions/clip_curriculum.py`
- Create: `tests/test_clip_curriculum.py`

**Step 1: Write failing tests for hardness + mapping + mixing**

Create `tests/test_clip_curriculum.py`:
```python
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
```

**Step 2: Run tests (expect fail)**

Run: `uv run pytest -q tests/test_clip_curriculum.py`
Expected: FAIL (module/function not found)

**Step 3: Implement minimal utilities**

Create `src/tracker/motions/clip_curriculum.py`:
```python
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
```

**Step 4: Run tests (expect pass)**

Run: `uv run pytest -q tests/test_clip_curriculum.py`
Expected: PASS

**Step 5: Commit**

```bash
git add src/tracker/motions/clip_curriculum.py tests/test_clip_curriculum.py
git commit -m "feat: add clip curriculum utilities"
```

---

### Task 2: Extend hierarchical clip sampling to accept per-clip multipliers (TDD)

**Files:**
- Modify: `src/tracker/motions/sampling.py`
- Modify: `tests/test_multiclip_sampling.py`

**Step 1: Add a failing test for multiplier-biased sampling**

Append to `tests/test_multiclip_sampling.py`:
```python
def test_hierarchical_sampler_respects_clip_multiplier_within_task():
    from tracker.motions.sampling import HierarchicalSampler

    torch.manual_seed(0)
    clip_task_id = torch.tensor([0, 0, 0, 0], dtype=torch.int64)
    split_clip_ids = torch.arange(4, dtype=torch.int64)
    clip_len = torch.tensor([100, 100, 100, 100], dtype=torch.int64)
    # Make clip 2 "hard" via multiplier
    clip_multiplier = torch.tensor([1.0, 1.0, 10.0, 1.0], dtype=torch.float32)

    sampler = HierarchicalSampler(clip_task_id=clip_task_id, split_clip_ids=split_clip_ids)
    clip_ids, _, _ = sampler.sample(n=20000, clip_len=clip_len, clip_multiplier=clip_multiplier)
    counts = torch.bincount(clip_ids, minlength=4).float()
    assert counts[2] > counts.mean() * 2.0
```

**Step 2: Run the test (expect fail)**

Run: `uv run pytest -q tests/test_multiclip_sampling.py::test_hierarchical_sampler_respects_clip_multiplier_within_task`
Expected: FAIL (unexpected keyword arg)

**Step 3: Implement multiplier support (bonus-only floor stays outside)**

Modify `src/tracker/motions/sampling.py`:
- Add optional arg `clip_multiplier: torch.Tensor | None = None` to `HierarchicalSampler.sample`.
- If provided, validate shape `(num_clips,)` and dtype `float32/float64`.
- Within each task bucket, use:
  - `weights = candidate_lens * clip_multiplier[candidates]`
  - fall back to `candidate_lens` if multiplier is None.

**Step 4: Run tests**

Run: `uv run pytest -q tests/test_multiclip_sampling.py`
Expected: PASS

**Step 5: Commit**

```bash
git add src/tracker/motions/sampling.py tests/test_multiclip_sampling.py
git commit -m "feat: allow hierarchical sampler clip multipliers"
```

---

### Task 3: Add clip-curriculum config fields for motion-pack command (TDD)

**Files:**
- Modify: `src/tracker/tasks/tracking/multiclip_command.py`
- Modify: `src/tracker/tasks/tracking/config/patch.py`
- Modify: `src/tracker/cli/train.py`
- Modify: `src/tracker/cli/common.py`
- Modify: `src/tracker/integrations/mjlab/offline_train.py`
- Modify: `tests/test_task8_train_cli.py`

**Step 1: Add CLI parsing tests (fail first)**

In `tests/test_task8_train_cli.py`, add a case asserting we can pass:
- `--motion-pack-clip-curriculum-mode <mode>`
- `--motion-pack-clip-curriculum-mix <float>` (default: `1.0`)

Recommended `<mode>` values:
- `ema_bin_failed` (default; enabled for `--motion-pack`)
- `off`

**Step 2: Run tests (expect fail)**

Run: `uv run pytest -q tests/test_task8_train_cli.py`
Expected: FAIL (unknown args)

**Step 3: Implement config surface**

Implement:
- Extend `MotionPackCommandCfg` in `src/tracker/tasks/tracking/multiclip_command.py` with:
  - `clip_curriculum_mode: str = "ema_bin_failed"`
  - `clip_curriculum_mix: float = 1.0`
  - `clip_curriculum_strength: float = 2.0`
  - `clip_curriculum_tau_scale: float = 1.0`
  - `clip_curriculum_max_mult: float = 3.0`
- Update `apply_motion_pack` in `src/tracker/tasks/tracking/config/patch.py` to accept optional overrides for these (keep defaults).
- Update `src/tracker/cli/common.py` to:
  - enforce `--motion-pack-clip-curriculum-*` requires `--motion-pack`
  - parse the values (similar to `get_motion_pack_sampling_mode`)
- Update `src/tracker/cli/train.py` to:
  - strip the new flags before passing args to tyro
  - pass the parsed values down to `launch_training_offline(...)`
- Update `src/tracker/integrations/mjlab/offline_train.py` to:
  - accept `motion_pack_clip_curriculum_*` args
  - forward them into `apply_offline_motion_pack(...)`, then into `apply_motion_pack(...)`

**Step 4: Run tests**

Run: `uv run pytest -q tests/test_task8_train_cli.py`
Expected: PASS

**Step 5: Commit**

```bash
git add src/tracker/tasks/tracking/multiclip_command.py src/tracker/tasks/tracking/config/patch.py src/tracker/cli/train.py tests/test_task8_train_cli.py
git commit -m "feat: expose motion-pack clip curriculum flags"
```

---

### Task 4: Implement global clip curriculum in `MultiClipMotionCommand` (TDD + smoke)

**Files:**
- Modify: `src/tracker/tasks/tracking/multiclip_command.py`
- Modify: `tests/test_multiclip_sampling.py`

**Step 1: Add a focused unit test for curriculum-enabled sampling**

Extend `tests/test_multiclip_sampling.py` with a minimal `MultiClipMotionCommand.__new__` harness:
- Provide `cmd.adaptive_sampler.bin_failed_count` with one clip having large max.
- Provide cfg fields enabling curriculum mode.
- Assert `_sample_motion()` produces skew towards hard clip over many samples (statistical).

**Step 2: Run test (expect fail)**

Run: `uv run pytest -q tests/test_multiclip_sampling.py::<new_test_name>`
Expected: FAIL (not implemented)

**Step 3: Implement curriculum computation**

In `MultiClipMotionCommand._sample_motion`:
- If `cfg.clip_curriculum_mode == "off"`: use current behavior.
- If `"ema_bin_failed"`:
  - compute `hardness = clip_hardness_from_bin_failed_count(self.adaptive_sampler.bin_failed_count)`
  - per task bucket, compute `tau = cfg.clip_curriculum_tau_scale * mean(hardness_task)`
  - map to `multiplier` using `hardness_to_multiplier(...)` with:
    - `strength = cfg.clip_curriculum_strength`
    - `m_max = cfg.clip_curriculum_max_mult`
  - construct `clip_multiplier` such that:
    - `w_base ∝ clip_len`
    - `w_curr ∝ clip_len * multiplier`
    - `w = mix_baseline_and_curriculum(...)`
  - pass resulting per-clip multiplier (or per-clip effective weight) into `HierarchicalSampler.sample`.

**Step 4: Run tests**

Run: `uv run pytest -q tests/test_multiclip_sampling.py`
Expected: PASS

**Step 5: Smoke train (optional, local)**

Run a short training (keep small envs first):
```bash
uv run tracker-train Tracker-Tracking-Flat-Adam-Pro-29-No-State-Estimation \
  --motion-pack <PACK_DIR> \
  --motion-split train \
  --gpu-ids 0 \
  --agent.max-iterations 50 \
  --env.scene.num-envs 128
```
Expected: starts and logs curriculum-related metrics (placeholders ok initially).

**Step 6: Commit**

```bash
git add src/tracker/tasks/tracking/multiclip_command.py tests/test_multiclip_sampling.py
git commit -m "feat: add EMA failure-based global clip curriculum"
```

---

### Task 5: Add a dedicated teleop task ID (alias config) + registry test

**Files:**
- Modify: `src/tracker/tasks/tracking/register.py`
- Modify: `tests/test_registry_smoke.py`
- Modify: `docs/plans/2026-01-29-phase-2-5-teleop-policy-design.md`

**Step 1: Add failing registry test**

Update `tests/test_registry_smoke.py` to assert:
```python
assert "Tracker-Teleop-Flat-Adam-Pro-29-No-State-Estimation" in tasks
```

Run: `uv run pytest -q tests/test_registry_smoke.py`
Expected: FAIL

**Step 2: Register the new task**

Update `src/tracker/tasks/tracking/register.py` to add a new task spec:
- `Tracker-Teleop-Flat-Adam-Pro-29-No-State-Estimation`
- env cfg: reuse `adam_pro_29_flat_tracking_env_cfg(has_state_estimation=False)`
- play env cfg: reuse the play builder
- rl cfg: reuse `adam_pro_29_tracking_ppo_runner_cfg()`

**Step 3: Run tests**

Run: `uv run pytest -q tests/test_registry_smoke.py`
Expected: PASS

**Step 4: Commit**

```bash
git add src/tracker/tasks/tracking/register.py tests/test_registry_smoke.py docs/plans/2026-01-29-phase-2-5-teleop-policy-design.md
git commit -m "feat: register teleop tracking task"
```

---

### Task 6: Documentation + “how to train teleop”

**Files:**
- Modify: `docs/plans/2026-01-29-phase-2-5-teleop-policy-design.md`
- Create: `docs/plans/2026-01-29-phase-2-5-teleop-policy-training-notes.md` (optional)

**Step 1: Add canonical train/eval commands**
- Use the teleop task ID.
- Document how to disable curriculum for baselines (`--motion-pack-clip-curriculum-mode off`).

**Step 2: Commit**

```bash
git add docs/plans/2026-01-29-phase-2-5-teleop-policy-design.md docs/plans/2026-01-29-phase-2-5-teleop-policy-training-notes.md
git commit -m "docs: add teleop curriculum training notes"
```

---

## Final Verification

Run:
```bash
uv run pytest -q
```
Expected: PASS

---

## Execution Options

Plan complete and saved to `docs/plans/2026-01-29-phase-2-5-global-clip-curriculum-implementation-plan.md`.

Two execution options:
1) **This session:** implement task-by-task using `superpowers:executing-plans`
2) **Separate session:** open a fresh session in worktree `tracker/.worktrees/phase-2.5` and use `superpowers:executing-plans`

Which approach?

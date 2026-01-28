# Phase 2: Multi-Clip Adaptive Sampling (Polish) Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Close the performance gap between single-clip and multi-clip tracking by adding **mjlab-style adaptive time-bin sampling per clip** for multi-clip motion packs, while keeping clip selection **length-weighted** (frame-uniform) within each task.

**Architecture:** Keep the existing hierarchical sampler (task → clip). Replace uniform-in-clip time sampling with a per-clip adaptive sampler that mirrors mjlab’s `MotionCommand` adaptive logic (failure-driven bin distribution + optional kernel smoothing). Keep `mjlab/` untouched; all changes live in `tracker/`.

**Tech Stack:** Python, PyTorch, mjlab (unchanged), rsl-rl, uv.

---

## Background / Motivation

Single-clip tracking converges quickly and strongly because mjlab defaults use `sampling_mode="adaptive"` and therefore concentrate sampling on hard time segments (non-uniform `sampling_entropy < 1`).

Multi-clip tracking currently underperforms even on the same evaluation clip because:
- The motion-pack path forces `sampling_mode="uniform"` (no curriculum),
- Multi-clip time sampling is uniform within clips (max-entropy sampling distribution),
- Result: hard segments are under-sampled, and the policy remains noisier / less accurate.

---

## Decisions (resolved)

- **Bin granularity:** Use **mjlab-style** bins roughly “per second”:
  - `frames_per_bin = int(1.0 / env.step_dt)`
  - `bin_count = clip_len // frames_per_bin + 1`
- **Failure signal:** Use `env.termination_manager.terminated` (matches mjlab; excludes timeouts).
- **Clip selection:** Keep current length-weighted (frame-uniform) within each task.
- **Kernel smoothing:** **Enable** mjlab-style smoothing via `MotionCommandCfg.adaptive_kernel_size/adaptive_lambda`
  - Note: keep `kernel_size=1` as a baseline test case; add a second test case for `kernel_size>1` if needed.
- **Safety / experimentation:** Keep motion-pack sampling settings **recoverable**:
  - `apply_motion_pack(...)` should **preserve** the env-cfg’s existing `sampling_mode` by default (mjlab default is `adaptive`).
  - Add a **tracker-owned CLI override** to force motion-pack sampling mode for A/B tests:
    - `adaptive` (new per-clip adaptive time bins)
    - `uniform` (current baseline behavior)
    - `start` (debug)
  - **CLI flag name:** `--motion-pack-sampling-mode {adaptive,uniform,start}`

---

## Acceptance Criteria (MVP)

- Multi-clip motion-pack training supports `sampling_mode="adaptive"` and yields non-trivial sampling metrics (`sampling_entropy < 1.0`, `sampling_top1_prob > 0`).
- For a “single clip in a pack” experiment, `--motion-pack` + adaptive sampling achieves similar performance to `--motion-file` baseline (within a reasonable tolerance; TODO define).
- `mjlab/` remains unchanged.
- Tests added/updated, and `uv run pytest -q` passes.

---

### Task 1: Add adaptive time-bin sampler (unit-tested)

**Files:**
- Create: `src/tracker/motions/adaptive_time_sampler.py`
- Test: `tests/test_adaptive_time_sampler.py`

**Step 1: Write failing tests**

Create `tests/test_adaptive_time_sampler.py`:

```py
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
  sampler.update_failures(clip_ids=clip_ids, time_steps=time_steps, terminated=terminated)

  g = torch.Generator().manual_seed(0)
  sampled_t = sampler.sample_time(clip_ids=torch.zeros((2048,), dtype=torch.int64), generator=g)

  # Heuristic: after seeding failures late, the sampler should prefer late times.
  assert sampled_t.float().mean().item() > 500


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
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest -q tests/test_adaptive_time_sampler.py`
Expected: FAIL (`ModuleNotFoundError: tracker.motions.adaptive_time_sampler`).

**Step 3: Implement minimal sampler**

Create `src/tracker/motions/adaptive_time_sampler.py` implementing:
- per-clip `bin_count` (per decision)
- per-clip `bin_failed_count` storage
- `update_failures(...)` (uses `terminated` mask only)
- `sample_time(...)` (group by unique clip_id; sample bins via `torch.multinomial`)
- optional kernel smoothing (placeholder)

**Step 4: Run tests**

Run: `uv run pytest -q tests/test_adaptive_time_sampler.py`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_adaptive_time_sampler.py src/tracker/motions/adaptive_time_sampler.py
git commit -m "phase-2: add per-clip adaptive time-bin sampler"
```

---

### Task 2: Wire adaptive sampling into `MultiClipMotionCommand`

**Files:**
- Modify: `src/tracker/tasks/tracking/multiclip_command.py`
- Test: `tests/test_multiclip_sampling.py` (or create a new focused test; placeholder)

**Step 1: Write a failing test (sampling metrics become non-uniform)**

TODO: Decide the lightest-weight test harness:
- Option A (recommended): unit-test a helper function that converts `(clip_id, time_step)` failures into per-clip bin updates, then verify sampling bias.
- Option B: integration test with a tiny fake pack object (more work).

**Step 2: Implement wiring**

Update `MultiClipMotionCommand`:
- Construct an `AdaptiveTimeBinSampler` from `pack.clip_len` and `env.step_dt`.
- When `cfg.sampling_mode == "adaptive"`:
  - before sampling new time, call `adaptive_sampler.update_failures(...)` using:
    - `clip_ids=self.clip_id[env_ids]`
    - `time_steps=self.time_steps[env_ids]`
    - `terminated=self._env.termination_manager.terminated[env_ids]`
  - choose `t = adaptive_sampler.sample_time(clip_ids=<sampled_clip_ids>)`
- Keep `sampling_mode == "uniform"` behavior unchanged.
- Populate `sampling_entropy/top1_prob/top1_bin` (placeholder for definition; recommended: match mjlab’s metric computation for each selected clip distribution).

**Step 3: Run tests**

Run: `uv run pytest -q`
Expected: PASS

**Step 4: Commit**

```bash
git add src/tracker/tasks/tracking/multiclip_command.py tests/<UPDATED_TESTS>
git commit -m "phase-2: enable adaptive sampling for motion packs"
```

---

### Task 3: Preserve + override motion-pack sampling mode (A/B friendly)

**Files:**
- Modify: `src/tracker/tasks/tracking/config/patch.py`
- Modify: `src/tracker/cli/train.py` (offline path only)
- Modify: `src/tracker/cli/common.py` (flag parsing/stripping; if needed)
- Modify: `src/tracker/integrations/mjlab/offline_train.py` (thread the override)
- Test: `tests/test_registry_smoke.py` (or a focused new test; placeholder)

**Step 1: Write failing test**

TODO: Add a test that `apply_motion_pack(...)`:
- preserves the existing `sampling_mode` from the base tracking env-cfg (expected default: `adaptive`), and
- supports overriding to `uniform` for motion-pack training (exact flag plumbing TBD).

**Step 2: Implement**

In `apply_motion_pack(...)`:
- Stop overwriting `sampling_mode` to `"uniform"`.
- Preserve the original `motion_cmd.sampling_mode` unless an override was provided.

In `tracker-train` offline mode:
- Parse a tracker-owned override flag:
  - `--motion-pack-sampling-mode {adaptive,uniform,start}`
- Strip it before passing the remaining args to tyro/mjlab.
- Thread it into `launch_training_offline(...)` and then into `apply_motion_pack(...)`.

NOTE: Keep this override **tracker-only** (do not modify `mjlab/` CLI/config types).

**Step 3: Run tests**

Run: `uv run pytest -q`
Expected: PASS

**Step 4: Commit**

```bash
git add src/tracker/tasks/tracking/config/patch.py src/tracker/cli/train.py src/tracker/cli/common.py src/tracker/integrations/mjlab/offline_train.py tests/<UPDATED_TESTS>
git commit -m "phase-2: allow motion-pack sampling mode override"
```

---

### Task 4: Smoke experiments (manual)

**Goal:** Validate that multi-clip + adaptive sampling closes most of the gap on a “single clip in a pack” control experiment.

**Step 1: Build a pack containing exactly one clip**

TODO: Provide a manifest path and pack output directory.

**Step 2: Train with `--motion-pack` (adaptive)**

Run (placeholder):
```bash
uv run tracker-train <TASK_ID> \
  --motion-pack <PACK_DIR> --motion-split train \
  --gpu-ids 0 --agent.max-iterations <N> --env.scene.num-envs <M>
```

**Step 3: Compare to `--motion-file` baseline**

Compare:
- `sampling_entropy` and related metrics (should be < 1)
- motion errors and termination rates

**Step 4: (Optional) Train on a small multi-clip pack and verify it no longer collapses**

TODO: Define “good enough” thresholds.

---

## Execution Handoff

Plan saved to `docs/plans/2026-01-26-phase-2-multiclip-adaptive-sampling-polish-implementation-plan.md`.

Two execution options:
1) **Subagent-Driven (this session)** — use `superpowers:subagent-driven-development`
2) **Parallel Session (separate)** — open a new session and use `superpowers:executing-plans`

Which approach?

# Phase 2.5: Teleoperation Policy (Adam‑Pro‑29 No‑SE) — Design & Plan

**Goal:** Train a **low‑level whole‑body RL controller** suitable for teleoperation, i.e., it should robustly track **any** motion clip drawn from a large multi‑clip dataset (motion pack), and generalize across clips.

**Scope (this doc):** Training + evaluation design for a universal controller. (Deployment and sim‑to‑real details remain out of scope for now.)

**Target task:** `Tracker-Tracking-Flat-Adam-Pro-29-No-State-Estimation` (training from a motion pack).

**Constraints / invariants:**
- Keep `mjlab/` untouched.
- Keep training `episode_length_s` at the current tracker default (≈10s) for stability/throughput.
- Preserve the existing **per‑clip adaptive time‑bin sampling** (mjlab‑style) because it is already validated via “pack‑of‑one” experiments.

**Success criteria:**
- Multi‑clip policy approaches single‑clip policy quality on held‑out clips (error + reward metrics comparable).
- Stable **30–60s** continuous rollouts in play/eval (no slow drift / no collapse).
- No clip starvation (all clips retain a non‑zero sampling probability over time).

---

## Background / Why Phase‑2.5

We observed:
- Single‑clip training (mjlab default `sampling_mode="adaptive"`) converges strongly.
- Multi‑clip training without a clip‑wise curriculum converges but yields noticeably worse tracking and higher failure/termination rates.
- “Pack‑of‑one” experiments show the motion‑pack + adaptive sampler path can match single‑clip performance, so the gap is largely a **clip selection / curriculum** issue (not model capacity or pack IO).

The core hypothesis for Phase‑2.5 is:
> We need a **global clip‑wise curriculum** (within task) that prioritizes hard clips while the existing **per‑clip** adaptive time‑bin sampler prioritizes hard windows *inside* each clip.

---

## Proposed Sampling Architecture (Two Levels)

### Level 1: Clip selection (global curriculum, within each task)

Keep existing “task → clip” structure, but modify **clip selection** to be curriculum‑aware:
- Task sampling remains uniform across `clip_task_id` buckets (for now we may only have one task bucket, but keep the structure).
- Clip sampling within each task uses:
  - **Base weight:** proportional to clip length (frame‑uniform).
  - **Curriculum bonus:** a per‑clip hardness multiplier derived from within‑clip failure statistics.
  - **Uniform floor:** ensure no starvation (mixture with uniform).

**Design choice (locked):** Clip hardness is derived from the **maximum** failed‑bin count for that clip:
```
hardness_clip = max(bin_failed_count[clip, :])
```
Rationale: a clip remains “hard” if it contains *any* hard 10s window, even if other windows are easy.

### Level 2: Time selection (within‑clip adaptive sampling)

Keep the existing per‑clip adaptive time‑bin sampler:
- For the selected `clip_id`, choose a start time `t0` from a distribution biased toward bins where failures occur.
- Keep `adaptive_uniform_ratio` so bins do not collapse to a single region forever.

---

## Curriculum Update Signal (Length‑Invariant)

We should not use “completion rate” defined by episode_time / clip_length (TWIST2 style) because tracker training uses a short fixed horizon (~10s) while clip lengths are often ~30s, so completion would be biased by design.

Instead, use a length‑invariant signal:
- `fail = termination_manager.terminated` (non‑timeout failures)
- `timeout = termination_manager.time_outs` (not counted as fail)

The per‑clip adaptive sampler already updates bin failures based on `terminated`. Phase‑2.5 reuses those stats to compute clip hardness.

---

## Avoiding Starvation (Important)

To ensure we never skip hard windows due to global clip downweighting:
- Global curriculum should be **bonus‑only** (never reduces below baseline).
- Apply a uniform mixture at clip selection:
  - `p = (1 - eps_clip_uniform) * p_weighted + eps_clip_uniform * p_uniform`
  - `eps_clip_uniform`: **placeholder** (to be tuned)

This guarantees every clip stays discoverable, allowing within‑clip adaptive sampling to find and focus on hard windows.

---

## Teleop Task (Why still add one)

Even if curriculum is default‑on for all motion‑pack training, we still want a dedicated “teleop policy” task registry entry later to:
- Provide a stable “train the universal controller” entrypoint.
- Allow future teleop‑specific defaults (curriculum strength, DR, eval horizons, logging) without silently changing baseline tracking tasks.

**Phase‑2.5 decision (locked):** add a dedicated teleop task ID now (initially an alias of the existing Adam‑Pro‑29 No‑SE tracking config) so we have isolation and room for later teleop‑specific defaults without touching baseline tracking tasks.

---

## Metrics & Logging (must-have)

Add/track metrics that tell us whether the curriculum is actually doing something:
- Clip sampling:
  - clip‑sampling entropy (normalized)
  - top‑1 clip probability (how peaked)
  - fraction of unique clips visited per N iterations (anti‑starvation)
- Curriculum stats:
  - mean/max `hardness_clip` (or its normalized proxy)
  - (optional) per‑task bucket summaries if multiple `clip_task_id`s exist
- Keep existing within‑clip adaptive metrics:
  - `Metrics/motion/sampling_entropy`
  - `Metrics/motion/sampling_top1_prob`
  - `Metrics/motion/sampling_top1_bin`

---

## Training / Eval Workflow (reference)

### Train (multi‑clip pack)
```bash
uv run tracker-train Tracker-Tracking-Flat-Adam-Pro-29-No-State-Estimation \
  --motion-pack <PACK_DIR> \
  --motion-split train \
  --gpu-ids 0 \
  --agent.max-iterations <N> \
  --env.scene.num-envs 4096
```

### Eval (val split)
```bash
uv run tracker-eval Tracker-Tracking-Flat-Adam-Pro-29-No-State-Estimation \
  --checkpoint <CHECKPOINT.pt> \
  --motion-pack <PACK_DIR> \
  --motion-split val \
  --num-episodes 50 \
  --num-envs 128 \
  --device cuda:0
```

### Long-horizon play sanity (full clip)
`tracker-play` uses the play env cfg, which effectively removes the 10s cap, so it can play full‑length clips unless non‑timeout terminations trigger:
```bash
uv run tracker-play Tracker-Tracking-Flat-Adam-Pro-29-No-State-Estimation \
  --checkpoint_file <CHECKPOINT.pt> \
  --motion_file <CLIP.npz> \
  --device cuda:0 \
  --num_envs 1
```

---

## Open Questions / Placeholders (to resolve during implementation)

- Curriculum defaults (Phase‑2.5 v0; tune later if needed):
  - `clip_curriculum_mode`: `ema_bin_failed` (default-on for motion-pack training)
  - `clip_curriculum_mix`: `1.0`
  - `clip_curriculum_strength`: `2.0`
  - `clip_curriculum_tau_scale`: `1.0` (uses per-task `tau = tau_scale * mean(h)`)
  - `clip_curriculum_max_mult`: `3.0`
  - `eps` for normalization: `1e-6`
- Update cadence: recompute per-task `tau` and multipliers at sampling time (each `_sample_motion` call).
- Whether to incorporate error‑mining later (clip‑level reward/error EMA): `<TBD>`
- Dataset bucket definitions (`clip_task_id` meaning): keep placeholders until your categorization is finalized: `<TBD>`

# Phase-2 Multi-Clip + Multi-GPU Tracking Implementation Plan


> **For Claude/ GPT-5.2:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task-by-task.

**Goal:** Train tracking policies on a massive multi-clip motion dataset (10k+ clips) with a deterministic pack format, task-balanced sampling, optional adaptive curriculum, and offline multi-GPU training support.

**Architecture:** Add a tracker-owned “motion pack” format (directory with `meta.json` + `arrays/*.npy` + `splits/*.npy`) produced by an explicit `tracker-pack-motions` CLI. In training, patch the tracking command config to instantiate a tracker-owned `MultiClipMotionCommand` via `MotionCommandCfg.class_type`, keeping `mjlab/` unmodified. For multi-GPU offline training, mirror `mjlab/scripts/train.py`’s torchrunx launcher in `tracker/integrations/mjlab/offline_train.py`.

**Tech Stack:** Python 3.10+, `uv`, `numpy`, `torch`, `mjlab` (unmodified), `torchrunx` (via `mjlab`), `pytest`.

---

## Requirements (locked in by brainstorming)

- **No changes to `mjlab/`.** All multi-clip logic lives in `tracker/`.
- **Motion source:** per-clip `.npz` files already matching mjlab tracking loader keys (example: `/home/humanoid/Projects/Junsong_WU/ADAM/adam_data/tests/adam_test_data/sub10_largebox_049_bm_new.npz`).
- **Dataset grouping:** clips are bucketed by task (e.g., `walk/ run/ squat/ ...`).
- **Sampling (MVP):** hierarchical sampling `task → clip → time`:
  - tasks sampled **uniformly** by default
  - clips sampled **uniformly** within task by default
  - time sampled uniformly within clip
- **Splits:** deterministic, stratified by task, with minimums:
  - if task has `N>=2`: ensure `min_val=1`
  - if task has `N>=3`: ensure `min_test=1`
  - otherwise: train-only for that task
- **Multi-GPU:** interpret `--env.scene.num-envs` as **total across GPUs** and shard evenly; require divisibility.
- **Pack dtypes:** `joint_*` stored float32; `body_*` stored float16 (cast to float32 on GPU when used).
- **Curriculum:** adopt TWIST2-style “completion-based” adaptive difficulty later:
  - completion = `env.episode_length_buf / env.max_episode_length`

---

## Deliverables

1. `tracker-pack-motions` CLI that builds a motion pack directory from a small manifest.
2. `--motion-pack` + `--motion-split {train,val,test}` support in `tracker-train` (offline mode).
3. Tracker-owned `MultiClipMotionCommand` selected via `MotionCommandCfg.class_type` patching.
4. Offline multi-GPU support in `tracker/integrations/mjlab/offline_train.py` (torchrunx).
5. `tracker-eval` CLI that evaluates a checkpoint on a chosen split and prints per-task metrics.
6. Tests covering pack determinism, pack loading, sampling sanity, split rules, and env sharding logic.

---

### Task 1: Define motion manifest + pack directory schema (docs-first)

**Files:**
- Create: `docs/motions/motion-pack-format.md`
- Create: `docs/motions/motion-manifest-format.md`

**Step 1: Write minimal docs (no code yet)**

Document:
- Manifest schema (root + tasks with include/exclude globs, optional weights, split config).
- Pack layout (`meta.json`, `index.jsonl`, `arrays/*.npy`, `splits/*.npy`).
- Versioning (`schema_version: 1`).

**Step 2: Add a tiny example manifest**

Include an example with placeholder tasks (e.g., a single `all` bucket) and a `splits` section.

**Step 3: Commit**

```bash
git add docs/motions/motion-pack-format.md docs/motions/motion-manifest-format.md
git commit -m "docs: add motion pack + manifest formats"
```

---

### Task 2: Implement manifest parsing + clip resolution (pure functions)

**Files:**
- Create: `src/tracker/motions/__init__.py`
- Create: `src/tracker/motions/manifest.py`
- Test: `tests/test_motion_manifest_resolve.py`

**Step 1: Write failing test for resolving paths**

In `tests/test_motion_manifest_resolve.py`, create a temporary directory structure like:

```
root/
  walk/a.npz
  walk/b.npz
  run/c.npz
  bad/tmp_bad.npz
```

Test that:
- include globs match expected files
- exclude globs remove expected files
- output is sorted deterministically by relative path
- `task_id` mapping is stable

**Step 2: Run test to verify it fails**

Run: `uv run pytest -q tests/test_motion_manifest_resolve.py`
Expected: FAIL (module/function missing).

**Step 3: Implement minimal resolver**

In `src/tracker/motions/manifest.py`, implement:

```python
@dataclass(frozen=True)
class TaskSpec:
  name: str
  include: list[str]
  exclude: list[str] = field(default_factory=list)
  weight: float = 1.0

@dataclass(frozen=True)
class Manifest:
  schema_version: int
  root: Path
  tasks: list[TaskSpec]
  # splits config lives elsewhere (Task 3)

def resolve_clips(manifest: Manifest) -> list[ResolvedClip]:
  ...
```

Keep this stage “filesystem only”: no `.npz` loading yet.

**Step 4: Run test to verify it passes**

Run: `uv run pytest -q tests/test_motion_manifest_resolve.py`
Expected: PASS.

**Step 5: Commit**

```bash
git add src/tracker/motions/__init__.py src/tracker/motions/manifest.py tests/test_motion_manifest_resolve.py
git commit -m "feat: parse and resolve motion manifests"
```

---

### Task 3: Implement deterministic stratified splits with minimums

**Files:**
- Create: `src/tracker/motions/splits.py`
- Test: `tests/test_motion_splits.py`

**Step 1: Write failing tests**

Test cases:
- For `N=1` clips in a task: all train.
- For `N=2`: ensure 1 val, 1 train.
- For `N=3`: ensure 1 test, 1 val, 1 train.
- For `N=10` with ratios (e.g., val=0.02 test=0.01): apply ratios but still respect minimums.
- Determinism: same inputs + seed yields same split assignments.

**Step 2: Run test to verify it fails**

Run: `uv run pytest -q tests/test_motion_splits.py`
Expected: FAIL.

**Step 3: Implement splitter**

In `src/tracker/motions/splits.py`, implement:
- stable hashing: `sha256(f"{seed}:{relpath}")`
- per-task split assignment using hash percentiles
- post-process to enforce `min_val/min_test` by “promoting” boundary items deterministically

**Step 4: Run tests**

Run: `uv run pytest -q tests/test_motion_splits.py`
Expected: PASS.

**Step 5: Commit**

```bash
git add src/tracker/motions/splits.py tests/test_motion_splits.py
git commit -m "feat: deterministic stratified motion splits with minimums"
```

---

### Task 4: Implement motion pack writer (concat + `.npy` arrays)

**Files:**
- Create: `src/tracker/motions/pack_writer.py`
- Create: `src/tracker/motions/npz_schema.py`
- Create: `src/tracker/cli/pack_motions.py`
- Modify: `pyproject.toml` (add script entrypoint)
- Test: `tests/test_motion_pack_writer_smoke.py`

**Step 1: Write failing smoke test**

In `tests/test_motion_pack_writer_smoke.py`, create 2–3 minimal mjlab-format `.npz` clips (small `T`, small `B`, small `J`) and a manifest pointing at them.

Test that packer:
- writes `meta.json`, `index.jsonl`, `arrays/*.npy`, `splits/*.npy`
- concatenated shapes match expected totals
- `clip_start` and `clip_len` correctly segment arrays
- body arrays saved as float16, joint arrays as float32

**Step 2: Run test to verify it fails**

Run: `uv run pytest -q tests/test_motion_pack_writer_smoke.py`
Expected: FAIL.

**Step 3: Implement minimal schema validator**

In `src/tracker/motions/npz_schema.py`, implement:
- `load_npz_metadata(path)` that validates required keys exist
- verifies shapes are consistent and `time_step_total` matches across keys
- returns `joint_names/body_names` if present, and verifies they match across clips

**Step 4: Implement pack writer**

In `src/tracker/motions/pack_writer.py` implement:
- iterative load `.npz` clip arrays (streaming: do not keep all clips in RAM)
- accumulate into preallocated `.npy` files using `numpy.lib.format.open_memmap`
  - compute total frames first (first pass loads only shapes)
  - second pass writes arrays sequentially and records `clip_start/clip_len`
- emit:
  - `arrays/joint_pos.npy`, `arrays/joint_vel.npy` (float32)
  - `arrays/body_pos_w.npy`, `arrays/body_quat_w.npy`, `arrays/body_lin_vel_w.npy`, `arrays/body_ang_vel_w.npy` (float16)
  - `arrays/clip_start.npy`, `arrays/clip_len.npy`, `arrays/clip_task_id.npy`
  - `splits/train_clip_ids.npy`, etc.
  - `meta.json` and `index.jsonl`

**Step 5: Add CLI entrypoint**

In `src/tracker/cli/pack_motions.py` implement:
- `tracker-pack-motions --manifest path.yaml --out pack_dir`
- prints counts per task and split
- exits non-zero if any `.npz` fails schema validation

Add to `pyproject.toml`:

```toml
[project.scripts]
tracker-pack-motions = "tracker.cli.pack_motions:main"
```

**Step 6: Run smoke test**

Run: `uv run pytest -q tests/test_motion_pack_writer_smoke.py`
Expected: PASS.

**Step 7: Commit**

```bash
git add pyproject.toml src/tracker/motions src/tracker/cli/pack_motions.py tests/test_motion_pack_writer_smoke.py
git commit -m "feat: motion pack writer + tracker-pack-motions"
```

---

### Task 5: Implement motion pack loader (mmap + torch upload)

**Files:**
- Create: `src/tracker/motions/pack_loader.py`
- Test: `tests/test_motion_pack_loader.py`

**Step 1: Write failing test**

Using the pack produced in Task 4’s smoke test, verify loader can:
- read `meta.json`
- mmap `.npy` arrays
- load split clip ids
- materialize torch tensors on a requested device (cpu and cuda if available)

**Step 2: Run test to verify it fails**

Run: `uv run pytest -q tests/test_motion_pack_loader.py`
Expected: FAIL.

**Step 3: Implement loader**

`MotionPack.load(path, device=..., cast_body_to_f32=True)` should:
- mmap arrays
- on `device="cuda:0"`: upload arrays once
- expose helpers: `clips_for_task(task_id, split)`, `sample_clip_ids(...)`

**Step 4: Run tests**

Run: `uv run pytest -q tests/test_motion_pack_loader.py`
Expected: PASS.

**Step 5: Commit**

```bash
git add src/tracker/motions/pack_loader.py tests/test_motion_pack_loader.py
git commit -m "feat: motion pack loader with mmap + torch upload"
```

---

### Task 6: Implement MultiClipMotionCommand (hierarchical uniform sampling)

**Files:**
- Create: `src/tracker/tasks/tracking/multiclip_command.py`
- Modify: `src/tracker/tasks/tracking/config/patch.py`
- Modify: `src/tracker/cli/common.py`
- Modify: `src/tracker/cli/train.py`
- Modify: `src/tracker/integrations/mjlab/offline_train.py`
- Test: `tests/test_multiclip_sampling.py`

**Step 1: Write failing tests for sampler**

In `tests/test_multiclip_sampling.py`, unit test the sampling logic without requiring mujoco:
- Given tasks {A,B} with 2 and 8 clips, assert task sampling is uniform.
- Assert clip sampling is uniform within a task.
- Assert sampled `t` is within `[0, clip_len)`.

These tests should target a pure helper (e.g., `HierarchicalSampler`) used by the command.

**Step 2: Run test to verify it fails**

Run: `uv run pytest -q tests/test_multiclip_sampling.py`
Expected: FAIL.

**Step 3: Implement command + sampler**

In `src/tracker/tasks/tracking/multiclip_command.py`:
- Define `MultiClipMotionCommand(MotionCommand)`-like class (subclass `mjlab.managers.CommandTerm` directly).
- Accept `cfg: MotionCommandCfg` but treat `cfg.motion_file` as pack dir.
- Load `MotionPack` (Task 5) at init.
- Maintain per-env `clip_id`, `t`, `task_id`.
- Implement reset/compute to advance time and resample at clip end.

**Step 4: Patch env config**

In `src/tracker/tasks/tracking/config/patch.py`, add a helper:
- `apply_motion_pack(cfg, pack_dir, split)` that:
  - sets `motion_cmd.motion_file = pack_dir`
  - sets `motion_cmd.class_type = MultiClipMotionCommand`
  - stores the chosen split in an agreed place:
    - recommended: replace `env_cfg.commands["motion"]` with a tracker subclass of `MotionCommandCfg` that adds `motion_split`
    - fallback: set `os.environ["TRACKER_MOTION_SPLIT"]=...` and read in command

Prefer the subclass approach to avoid env vars.

**Step 5: Extend tracker-train CLI**

In `src/tracker/cli/common.py`:
- parse `--motion-pack` (path) and `--motion-split` (`train|val|test`)
- enforce exclusivity among `--motion-file`, `--motion-pack`, `--registry-name`

In `src/tracker/cli/train.py`:
- If `--motion-pack` present, select offline mode, strip flags before tyro parsing, and call offline launcher with pack_dir.

In `src/tracker/integrations/mjlab/offline_train.py`:
- accept `motion_pack_dir` + `motion_split` (or reuse `motion_file` for pack_dir)
- call `apply_motion_pack(...)` before env creation

**Step 6: Run tests**

Run: `uv run pytest -q tests/test_multiclip_sampling.py`
Expected: PASS.

**Step 7: Commit**

```bash
git add src/tracker/tasks/tracking/multiclip_command.py src/tracker/tasks/tracking/config/patch.py \
  src/tracker/cli/common.py src/tracker/cli/train.py src/tracker/integrations/mjlab/offline_train.py \
  tests/test_multiclip_sampling.py
git commit -m "feat: multi-clip motion command + tracker-train --motion-pack"
```

---

### Task 7: Offline multi-GPU support for `tracker-train` offline mode

**Files:**
- Modify: `src/tracker/integrations/mjlab/offline_train.py`
- Test: `tests/test_offline_env_sharding.py`

**Step 1: Write failing test for env sharding logic**

Test a pure function:

```python
def shard_num_envs(total: int, world_size: int) -> int:
  ...
```

Requirements:
- raises if `total % world_size != 0`
- returns `total // world_size` otherwise

**Step 2: Run test to verify it fails**

Run: `uv run pytest -q tests/test_offline_env_sharding.py`
Expected: FAIL.

**Step 3: Implement sharding helper + multi-GPU launcher**

In `src/tracker/integrations/mjlab/offline_train.py`:
- Mirror `mjlab/scripts/train.py`’s torchrunx path:
  - set `CUDA_VISIBLE_DEVICES` from selected GPUs
  - if `num_gpus > 1`, use torchrunx to spawn workers
  - in worker function:
    - set `MUJOCO_EGL_DEVICE_ID=local_rank`
    - set device `cuda:{local_rank}`
    - set `env_cfg.scene.num_envs = shard_num_envs(total_envs, world_size)`
    - rank-0 only logging/videos/config dumps

**Step 4: Run tests**

Run: `uv run pytest -q tests/test_offline_env_sharding.py`
Expected: PASS.

**Step 5: Manual smoke run**

Run (2 GPUs example):

```bash
UV_CACHE_DIR=$PWD/.uv-cache WANDB_DIR=$PWD/.wandb \
uv run tracker-train Tracker-Tracking-Flat-Adam-Pro-29 \
  --motion-pack /path/to/pack_dir --motion-split train \
  --gpu-ids 0 1 \
  --env.scene.num-envs 4096 \
  --agent.max-iterations 10
```

Expected:
- two workers start
- each uses 2048 envs
- only one log dir created

**Step 6: Commit**

```bash
git add src/tracker/integrations/mjlab/offline_train.py tests/test_offline_env_sharding.py
git commit -m "feat: offline multi-gpu training for motion-pack runs"
```

---

### Task 8: Add `tracker-eval` for split evaluation

**Files:**
- Create: `src/tracker/cli/eval.py`
- Modify: `pyproject.toml`
- Test: `tests/test_eval_cli_parsing.py`

**Step 1: Decide minimal CLI contract**

Recommended:
- `tracker-eval <task_id> --checkpoint <path> --motion-pack <pack_dir> --motion-split {val,test} --num-episodes N`

**Step 2: Write failing test for CLI arg parsing**

Test that required flags are enforced and split defaults to `val`.

**Step 3: Implement eval runner**

Implement evaluation loop (rank-0 only if multi-gpu support is added later):
- create env with `apply_motion_pack(..., split=val/test)` and `play=True` env cfg
- load checkpoint using rsl_rl runner’s `load()`
- run N episodes and aggregate mjlab extras + termination stats
- print per-task and global summaries

**Step 4: Register entrypoint**

Add to `pyproject.toml`:

```toml
tracker-eval = "tracker.cli.eval:main"
```

**Step 5: Run tests**

Run: `uv run pytest -q tests/test_eval_cli_parsing.py`
Expected: PASS.

**Step 6: Commit**

```bash
git add src/tracker/cli/eval.py pyproject.toml tests/test_eval_cli_parsing.py
git commit -m "feat: tracker-eval for motion-pack splits"
```

---

### Task 9 (Later): Add per-clip adaptive curriculum (TWIST2-style completion)

**Files:**
- Modify: `src/tracker/tasks/tracking/multiclip_command.py`
- Test: `tests/test_clip_curriculum_update.py`

**Step 1: Write failing test**

Test that:
- low completion envs increase difficulty for that clip
- high completion envs decrease difficulty
- difficulty is clamped
- sampling mixes in uniform probability (non-zero floor)

**Step 2: Implement update**

In `MultiClipMotionCommand.reset(env_ids)`:
- read `env.episode_length_buf[env_ids]` and `env.max_episode_length`
- completion = steps / max_steps
- update per-clip EMA stats using scatter_add
- update `clip_difficulty` with thresholds (0.5 / 0.95 / 0.99) and gamma

**Step 3: Run test**

Run: `uv run pytest -q tests/test_clip_curriculum_update.py`
Expected: PASS.

**Step 4: Commit**

```bash
git add src/tracker/tasks/tracking/multiclip_command.py tests/test_clip_curriculum_update.py
git commit -m "feat: adaptive curriculum via completion-based clip difficulty"
```

---

### Task 10 (Later): Sync curriculum across ranks (multi-GPU)

**Files:**
- Modify: `src/tracker/tasks/tracking/multiclip_command.py`

**Steps:**
- Add optional periodic `torch.distributed.all_reduce` for clip stats.
- Gate behind `torch.distributed.is_initialized()`.
- Only every K resets to limit overhead.

---

## Done Checklist

- `uv run pytest -q` passes.
- `uv build` succeeds.
- `tracker-pack-motions` can pack a small synthetic dataset.
- `tracker-train ... --motion-pack ... --motion-split train` runs on 1 GPU.
- `tracker-train ... --gpu-ids 0 1 ...` runs with env sharding and single log stream.
- `tracker-eval ... --motion-split val` runs and prints per-task metrics.

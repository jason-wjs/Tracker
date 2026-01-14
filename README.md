# tracker

A lightweight motion-tracking task pack built on top of [`mjlab`](https://github.com/mujocolab/mjlab.git).

This repo is organized as an installable Python package (`src/` layout) that:
- Registers tracking environments into `mjlab` (task IDs are discoverable via `tracker-list-envs`)
- Ships robot assets (MJCF + meshes) as package data, resolved at runtime
- Provides wrapper CLIs (`tracker-list-envs`, `tracker-train`, `tracker-play`) with motion-source preflight/validation
- Supports offline training via `--motion-file` without requiring W&B artifact resolution

Notes:
- Feature sets vary by branch/tag. Prefer runtime discovery (e.g., `tracker-list-envs`) and consult `docs/plans/` for phase-specific status.
- `--motion-file` and `--registry-name` are mutually exclusive.

## Recent refinements (Phase 1.5)

Depending on branch/tag, the repo includes refinements beyond the initial Adam-SP baseline:
- Added a second robot family (Adam-Pro, 23/29 DoF) and registered corresponding tracking task IDs.
- Introduced a small robot-adapter registry for task-id → motion validation/prep dispatch.
- Added `tracker-view-robot` for quick asset/actuator/collision inspection during iteration.
- Refactored `tracker.tasks.tracking` to separate config builders (`tracking/config/`) from registration (`tracking/register.py`).
- Tightened packaging hygiene: wheel includes runtime assets, excludes local-only reference files, and supports wheel/zip asset extraction for MuJoCo mesh loading.
- Added robustness patches for Adam-Pro: sensor-name aliasing for state-estimation observations, and collision pruning for hard-motion scaling at large `num-envs`.

## Setup

```bash
UV_CACHE_DIR=$PWD/.uv-cache uv sync --group dev --frozen
```

Notes:
- Use `--gpu-ids 0` for CUDA:0, or `--gpu-ids None` for CPU.
- Keep caches/logs local (untracked): `.uv-cache/`, `.venv/`, `.wandb/`, `.tracker-cache/`, `logs/`, `artifacts/`.
- If you use W&B, set `WANDB_DIR=$PWD/.wandb` so runs are stored in-repo.

## List all environments (task_ids)

```bash
uv run tracker-list-envs
```

## Inspect robot assets (`tracker-view-robot`)

Quickly load and sanity-check packaged MJCFs (and optionally open a MuJoCo viewer):

```bash
# Headless: just load and print model summary.
uv run tracker-view-robot --robot-id adam_pro --variant 29 --viewer none

# Interactive viewer (requires a working GUI/OpenGL setup):
uv run tracker-view-robot --robot-id adam_sp --variant 23 --viewer native
```

## Play

```bash
uv run tracker-play <task_id> \
  --checkpoint-file /path/to/checkpoint.pt  \
  --motion-file /path/to/motion.npz \
  --gpu-ids 0 \
  --num-envs 8
# or:
uv run tracker-play <task_id> \
  --registry-name <entity/project/motions:alias> \
  --gpu-ids 0 \
  --num-envs 8
```

## Train

```bash
uv run tracker-train <task_id> \
    --motion-file /path/to/motion.npz \
    --gpu-ids 0 \
    --agent.logger wandb \
    --agent.wandb-project <your-project> \
    --agent.max-iterations 10000 \
    --env.scene.num-envs 4096
# or:
uv run tracker-train <task_id> \ 
    --registry-name <entity/project/motions:alias> \
    --gpu-ids 0 \
    --agent.max-iterations 10000 \
    --env.scene.num-envs 4096
```

## Development

```bash
uv run pytest -q
uv build
```

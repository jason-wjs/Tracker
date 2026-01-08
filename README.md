# tracker

Motion-tracking RL tasks built on top of [`mjlab`](https://github.com/mujocolab/mjlab.git).

This branch (`phase-1`) is a minimal, usable snapshot focused on Adam-SP motion tracking:
- Registers two tracking tasks: `Tracker-Tracking-Flat-Adam-SP-23` and `Tracker-Tracking-Flat-Adam-SP-29`
- Provides wrapper CLIs: `tracker-list-envs`, `tracker-train`, `tracker-play`
- Supports both motion sources (mutually exclusive): `--motion-file` (offline) or `--registry-name` (W&B artifact)
- Includes a tracker-owned offline training path (so offline training does not depend on W&B motion resolution)
- Packages Adam-SP assets in the wheel and resolves them at runtime (no absolute paths / CWD assumptions)
- 29-DoF motions are validated strictly against the 29-DoF asset ordering; some `qpos/qvel` formats are auto-converted for offline use

What to expect in Phase 2 (planned): multi-clip sampling within one run, “one long motion.npz” dataset support, multi-GPU training, and a viewer/inspection helper for fast asset validation.

## Setup

```bash
UV_CACHE_DIR=$PWD/.uv-cache uv sync --group dev --frozen
```

Notes:
- Use `--gpu-ids 0` for CUDA:0, or `--gpu-ids None` for CPU.
- Local caches/logs are untracked: `.uv-cache/`, `.venv/`, `.wandb/`, `.tracker-cache/`, `logs/`, `artifacts/`.

## List environments

```bash
tracker-list-envs
```

## Play

```bash
tracker-play <task_id> --motion-file /path/to/motion.npz --gpu-ids 0
# or:
tracker-play <task_id> --registry-name <entity/project/motions:alias> --gpu-ids 0
```

## Train

```bash
tracker-train <task_id> --motion-file /path/to/motion.npz --gpu-ids 0
# or:
tracker-train <task_id> --registry-name <entity/project/motions:alias> --gpu-ids 0
```

`--motion-file` and `--registry-name` are mutually exclusive.

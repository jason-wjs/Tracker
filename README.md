# tracker

Motion-tracking RL tasks built on top of `mjlab`.

This branch (`phase-1`) is a minimal, usable snapshot focused on Adam-SP motion tracking:
- Registers two tracking tasks: `Tracker-Tracking-Flat-Adam-SP-23` and `Tracker-Tracking-Flat-Adam-SP-29`
- Supports `tracker-train` / `tracker-play` with either `--motion-file` (offline) or `--registry-name` (W&B artifact)

Task IDs:
- `Tracker-Tracking-Flat-Adam-SP-23`
- `Tracker-Tracking-Flat-Adam-SP-29`

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


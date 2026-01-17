# tracker

NOTE: This branch is experimental. See `docs/CHANGELOG.md`.

**Lightweight whole-body motion tracking for the Adam-SP robot family.**

`tracker` is a task package built on top of [`mjlab`](https://github.com/mujocolab/mjlab.git), designed for high-fidelity motion tracking with offline motion datasets.

## Setup

Initialize the environment using `uv`:
```bash
UV_CACHE_DIR=$PWD/.uv-cache uv sync --group dev --frozen
```

### Notes
- **Hardware**: Set `--gpu-ids 0` for CUDA acceleration or `None` for CPU execution.
- **Caches**: Directories such as `.uv-cache/`, `.venv/`, and `.tracker-cache/` are untracked and should be kept local.

## Usage

### List environments
```bash
uv run tracker-list-envs
```

### Train (offline motion file)
```bash
uv run tracker-train Tracker-Tracking-Flat-Adam-SP-29 \
  --motion-file /path/to/motion.npz \
  --gpu-ids 0 \
  --agent.max-iterations 5000 \
  --env.scene.num-envs 4096
```

### Play (rollout)
```bash
uv run tracker-play Tracker-Tracking-Flat-Adam-SP-29 \
  --checkpoint-file /path/to/checkpoint.pt \
  --motion-file /path/to/motion.npz \
  --device cuda:0 \
  --num-envs 8
```

## Development
```bash
uv run pytest -q
uv build
```

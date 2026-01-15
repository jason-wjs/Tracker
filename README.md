# tracker

**Lightweight whole-body motion tracking for the Adam-SP robot family.**

`tracker` is a specialized task package built on top of [`mjlab`](https://github.com/mujocolab/mjlab.git), designed for high-fidelity motion tracking. This branch (`phase-1`) provides a stable snapshot focused on the Adam-SP robot family, offering a streamlined workflow for training tracking policies from preprocessed motion data.

## Core Features (Phase 1)

Designed for modularity and ease of use, this snapshot includes:
- **Adam-SP Support**: Registered environments for both 23-DOF and 29-DOF Adam-SP configurations.
- **Self-Contained Assets**: Robot MJCFs and meshes are packaged with the library and resolved at runtime.
- **Workflow Utilities**: Specialized CLIs for environment listing, policy training, and playback.
- **Data Validation**: Strict validation for 29-DOF motion files, including automatic `qpos/qvel` normalization.
- **Offline Training**: Direct support for trained policies via `--motion-file` without external dependencies.

## Setup

### Installation
Initialize the environment using `uv`:
```bash
UV_CACHE_DIR=$PWD/.uv-cache uv sync --group dev --frozen
```

### Technical Notes
- **Hardware**: Set `--gpu-ids 0` for CUDA acceleration or `None` for CPU execution.
- **Caches**: Directories such as `.uv-cache/`, `.venv/`, and `.tracker-cache/` are untracked and should be kept local.

## Usage Guide

### 1. List Environments
Discover available task IDs and robot configurations:
```bash
uv run tracker-list-envs
```

### 2. Evaluate Policies (Play)
Run a trained checkpoint against a specific motion:
```bash
./play.sh
```

### 3. Train Policies
Launch RL training for motion tracking:
```bash
./train.sh
```

Key arguments:
- `<task_id>`: environment ID from `tracker-list-envs`.
- `--motion-file` (or `--motion_file`): local `.npz` motion file (offline training).
- `--gpu-ids`: CUDA device indices (e.g. `0`).
- `--agent.logger`: logger backend (e.g. `wandb`).
- `--agent.wandb-project`: W&B project name for logging.
- `--agent.max-iterations`: total training iterations.
- `--env.scene.num-envs`: number of parallel environments.


---
*Looking ahead: Phase 2 will introduce multi-clip sampling, multi-GPU training support, and enhanced asset validation tools.*

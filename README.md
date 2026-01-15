# tracker

**High-fidelity, whole-body motion tracking for the Adam robot family.**

`tracker` is a modular task package built on top of [mjlab](https://github.com/mujocolab/mjlab.git), specializing in universal motion tracking policies. By leveraging [BeyondMimic](https://beyondmimic.github.io/) as its core algorithmic engine, it provides a streamlined workflow for training agile humanoid behaviors from motion capture data.

## Core Features

Developed with a focus on speed, modularity, and high-fidelity simulation, `tracker` provides:
- **Environment Registration**: Seamless integration with the `mjlab` ecosystem.
- **Robot Assets**: Self-contained MJCF and mesh data for the Adam robot family, resolved dynamically.
- **Workflow Utilities**: Specialized CLIs for model inspection, policy evaluation, and training.
- **Flexible Data Sources**: Native support for offline `.npz` files and W&B artifact resolution.

## Phase 1.5 Refinements

This version introduces significant enhancements to the initial Adam-SP baseline:
- **Adam-Pro Support**: Added the Adam-Pro robot family in both 23-DOF and 29-DOF configurations.
- **Robot Adapter Registry**: Implemented a robust adapter system for task-specific motion validation and dataset preparation.
- **Visual Inspection**: New `tracker-view-robot` tool for rapid validation of assets, actuators, and collision geometries.
- **Architecture Refactor**: Clean separation of configuration builders from environment registration.
- **Package Hygiene**: Optimized package data handling, including support for mesh extraction from wheels/zips.
- **Stability Patches**: Introduced sensor-name aliasing and collision pruning to ensure stable training for "hard-motion" clips at scale.

## Getting Started

### Installation
Ensure you have `uv` installed, then synchronize the environment:
```bash
UV_CACHE_DIR=$PWD/.uv-cache uv sync --group dev --frozen
```

### Technical Notes
- **Hardware**: Set `--gpu-ids 0` for NVIDIA acceleration or `None` for CPU execution.
- **Logging**: Use `WANDB_DIR=$PWD/.wandb` to maintain localized W&B run logs.
- **Caches**: Local directories like `.uv-cache/`, `.venv/`, and `.tracker-cache/` are ignored by version control.

## Usage Guide

### 1. Discover Environments
List all registered task IDs and robot configurations:
```bash
uv run tracker-list-envs
```

### 2. Inspect Robot Assets
Validate robot models and verify joint/actuator configurations:
```bash
# Headless: just load and print model summary.
uv run tracker-view-robot --robot-id adam_pro --variant 29 --viewer none

# Interactive viewer (requires a working GUI/OpenGL setup):
uv run tracker-view-robot --robot-id adam_sp --variant 23 --viewer native
```

### 3. Evaluate Policies (Play)
Run a trained checkpoint against a specific motion:
```bash
./play.sh
```

### 4. Train Policies
Launch large-scale RL training for motion tracking:
```bash
./train.sh
```

Key arguments:
- `<task_id>`: environment ID from `tracker-list-envs`.
- `--motion-file` (or `--motion_file`): local `.npz` motion file (offline training).
- `--registry-name`: W&B motion artifact alias (use instead of `--motion-file`).
- `--gpu-ids`: CUDA device indices (e.g. `0`).
- `--agent.logger`: logger backend (e.g. `wandb`).
- `--agent.wandb-project`: W&B project name for logging.
- `--agent.max-iterations`: total training iterations.
- `--env.scene.num-envs`: number of parallel environments.


## Development
Run the test suite or build the distribution package:
```bash
uv run pytest -q
uv build
```

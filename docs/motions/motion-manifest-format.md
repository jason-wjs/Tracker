# Motion Manifest Format (schema_version: 1)

This document specifies the YAML manifest used by `tracker-pack-motions` to resolve per-clip `.npz` motion files into a deterministic *motion pack*.

## Overview

- The manifest points to a dataset `root` directory.
- Motions are organized into **tasks** (e.g., `walk`, `run`, `squat`), each defined by include/exclude globs.
- Splits (`train/val/test`) are computed **deterministically** and **stratified by task**, with minimum counts per task.

## Schema

```yaml
schema_version: 1

# Dataset root directory; task globs are evaluated relative to this.
root: /abs/path/to/motions_root

# Task buckets; each task resolves to a set of .npz clips.
tasks:
  # While you are still organizing a large dataset, start with a single catch-all
  # task (and later repack with more granular task buckets).
  - name: all
    include:
      - "**/*.npz"
    exclude: []
    weight: 1.0

# Deterministic split settings (stratified per task).
splits:
  seed: 0
  val_ratio: 0.02
  test_ratio: 0.01

  # Minimum clips per task in each split (when possible).
  min_val_per_task: 1
  min_test_per_task: 1
```

### Field meanings

- `schema_version` (`int`, required): format version; currently only `1` is supported.
- `root` (`str`, required): dataset root path.
- `tasks` (`list`, required): list of task specifications.
  - `name` (`str`, required): task name; used to generate a stable `task_id`.
  - `include` (`list[str]`, required): glob patterns selecting motion `.npz` files under `root`.
  - `exclude` (`list[str]`, optional): glob patterns removing matched files.
  - `weight` (`float`, optional, default `1.0`): task sampling weight (used in training; pack records it).
- `splits` (`object`, optional): split configuration.
  - `seed` (`int`, optional, default `0`): seed for deterministic hashing.
  - `val_ratio`, `test_ratio` (`float`, optional): target fractions within each task.
  - `min_val_per_task`, `min_test_per_task` (`int`, optional): minimum clips per task in val/test when possible.

## Resolution rules

1. A task’s clip set is computed as:
   - `include` matches minus `exclude` matches.
2. Only files with `.npz` extension are considered valid clips.
3. The resolved clip list is **sorted by relative path** (`root`-relative) for determinism.
4. `task_id` assignment is stable:
   - `tasks` are sorted by `name` and assigned sequential ids `0..(T-1)`.

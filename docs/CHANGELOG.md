# Changelog

This file records milestone features present in this branch snapshot (not semantic-versioned releases).

## 2026-01-04 — Phase 1 (Adam-SP baseline)

### Added
- Adam-SP tracking environments for 23-DoF and 29-DoF variants.
- Offline training and rollout using local motion `.npz` via `--motion-file`, including task-specific motion validation/preparation.
- Packaged robot MJCF assets shipped with the Python package (`tracker/assets/**`).

### References
- `docs/plans/2026-01-04-phase-1-implementation-plan.md`


## 2026-01-13 — Phase 1.5 (Adam-Pro + tracking cleanup)

### Added
- Adam-Pro robot assets and tracking tasks (23-DoF and 29-DoF variants).
- `tracker-view-robot` for quick validation of assets, joints/actuators, and collision geometry.
- Improved stability/packaging hygiene for assets and motion tracking configs.

### References
- `docs/plans/2026-01-13-phase-1-5-implementation-plan.md`
- `docs/plans/2026-01-14-tracking-register-consolidation-implementation-plan.md`
- `docs/plans/2026-01-14-tracking-structure-refactor-implementation-plan.md`


## 2026-01-16 — Phase 2 (multi-clip packs + evaluation)

### Added
- Packed multi-clip motion datasets via `tracker-pack-motions` (manifest → pack directory).
- Offline training on packed datasets via `--motion-pack` + `--motion-split` (no changes required in `mjlab/`).
- Offline evaluation on packed datasets via `tracker-eval` (supports `val`/`test` splits).

### Notes
- Multi-clip training reuses mjlab’s “load once, query on GPU” workflow by swapping the tracking command term to a tracker-owned multi-clip implementation.

### References
- `docs/plans/2026-01-16-phase-2-multiclip-multigpu-implementation-plan.md`


## 2026-01-26 — Phase 2 polish (adaptive sampling for motion packs)

### Added
- Per-clip adaptive time-bin sampling for packed datasets (motion-pack) to better support large multi-clip training.
- `--motion-pack-sampling-mode` CLI override for training with packed datasets (`adaptive|uniform|start`).
- Compatibility shim for mjlab ONNX export expectations when using the tracker-owned multi-clip command term (keeps `mjlab/` untouched).

### Notes
- This branch does **not** implement a clip-wise global curriculum/difficulty scheduler yet. Global curriculum will be implemented and benchmarked in a follow-up branch.

### References
- `docs/plans/2026-01-26-phase-2-multiclip-adaptive-sampling-polish-implementation-plan.md`

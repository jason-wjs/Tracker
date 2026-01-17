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

### Not included (added in later phases)
- Packed multi-clip datasets (`tracker-pack-motions`, `--motion-pack`, `--motion-split`) and evaluation CLI (`tracker-eval`).

### References
- `docs/plans/2026-01-13-phase-1-5-implementation-plan.md`
- `docs/plans/2026-01-14-tracking-register-consolidation-implementation-plan.md`
- `docs/plans/2026-01-14-tracking-structure-refactor-implementation-plan.md`

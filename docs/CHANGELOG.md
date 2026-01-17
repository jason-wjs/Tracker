# Changelog

This file records milestone features present in this branch snapshot (not semantic-versioned releases).

## 2026-01-04 — Phase 1 (Adam-SP baseline)

### Added
- Adam-SP tracking environments for 23-DoF and 29-DoF variants.
- Offline training and rollout using local motion `.npz` via `--motion-file`, including task-specific motion validation/preparation.
- Packaged robot MJCF assets shipped with the Python package (`tracker/assets/**`).

### Not included (added in later phases)
- Robot viewer CLI (`tracker-view-robot`).
- Packed multi-clip datasets (`tracker-pack-motions`, `--motion-pack`, `--motion-split`) and evaluation CLI (`tracker-eval`).

### References
- `docs/plans/2026-01-04-phase-1-implementation-plan.md`

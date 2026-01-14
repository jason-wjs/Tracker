# Phase 1.5 (Multi-Robot + Adam-Pro) Implementation Plan

> **For Claude/ Codex:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Extend `tracker` beyond Adam-SP by adding a second robot family (`adam_pro`, 23/29 DoF) and refactoring the codebase so future robots/variants do not require scattered branching logic; add a robot viewer/inspection helper to accelerate asset iteration.

**Architecture:** Introduce a small “robot adapter” layer that maps `task_id -> motion validation/prep + robot identifiers`, and factor tracking env-cfg patching so a new robot/variant is mostly “add robot module + add registration entry”. Keep strict `.npz` motion ordering (no name-based reordering) for all robots/variants.

**Tech Stack:** Python, `uv`, `mjlab`, MuJoCo, rsl-rl, optional W&B.

**Status (2026-01-15):**
- Completed: Tasks 1–8, 10 (multi-robot scaffolding, Adam-Pro support, tracking structure cleanup, packaging sanity).
- Deferred: Task 9 (wrist dynamics calibration) until deployment-focused work with trusted baselines.

## Inputs (must provide before Task 6)

- Adam-Pro 23-DoF MJCF source path (to copy into `src/tracker/assets/adam_pro/adam_pro.xml`).
- Adam-Pro 29-DoF MJCF source path (to copy into `src/tracker/assets/adam_pro/adam_pro_29dof.xml`).
- Adam-Pro mesh folder source path (to copy into `src/tracker/assets/adam_pro/meshes/` if referenced by the MJCF).
- Trusted wrist dynamics baselines for:
  - Adam-SP-29 wrists (armature, effort limits, PD gains)
  - Adam-Pro-29 wrists (armature, effort limits, PD gains)
  (e.g., URDF/CAD/datasheet constants, or a repo file path we can reference.)

## Addendum (2026-01-14): Collision optimization

- Collision geoms were optimized in:
  - `src/tracker/assets/adam_sp/adam_sp.xml`
  - `src/tracker/assets/adam_pro/adam_pro.xml`
  - `src/tracker/assets/adam_pro/adam_pro_29dof.xml`
  to reduce unnecessary contact complexity while preserving training-relevant contacts (e.g., toe/foot collisions for friction DR targeting).
- `adam_pro.xml` and `adam_pro_29dof.xml` were further refined after the initial optimization pass (e.g., collision/contact tuning to reduce contact complexity while preserving training-relevant contacts).
- Validate changes via `tracker-view-robot` (visual sanity + contacts) and by running a short `tracker-play` smoke with a representative motion.

## Addendum (2026-01-15): Adam-Pro sensor + hard-motion robustness fixes

- **Sensor naming compatibility (Adam-Pro):** `mjlab` tracking defaults expect IMU sensors named `imu_lin_vel` / `imu_ang_vel` (Adam-SP style), but Adam-Pro MJCFs expose `BodyVel` / `BodyGyro` / `BodyAcc`. For Adam-Pro tasks, `tracker` now aliases the expected observation sensor names to the available Adam-Pro sensors at env-config build time (keeps MJCF closest to reference).
- **Hard-motion robustness (Adam-Pro-29):** `nefc/njmax` overflows at large scale (`--env.scene.num-envs 4096`) indicate rare worst-case contact explosions. Since `mjwarp` uses fixed buffers, Adam-Pro-29 collision enabling is restricted to match Adam-SP-29 (feet + pelvis/torso only; hands/fingers disabled) to reduce worst-case contact/constraint counts without increasing `njmax/nconmax`.

---

### Task 1: Add task-id → robot adapter registry (no behavior change yet)

**Files:**
- Create: `src/tracker/robots/registry.py`
- Modify: `src/tracker/robots/__init__.py`

**Step 1: Write failing test (registry exposes known task IDs)**

Create: `tests/test_robot_registry.py`

```py
def test_robot_registry_lists_adam_sp_tasks():
  from tracker.robots.registry import list_known_task_ids

  task_ids = set(list_known_task_ids())
  assert "Tracker-Tracking-Flat-Adam-SP-23" in task_ids
  assert "Tracker-Tracking-Flat-Adam-SP-29" in task_ids
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest -q tests/test_robot_registry.py::test_robot_registry_lists_adam_sp_tasks`
Expected: FAIL (module/function missing)

**Step 3: Implement minimal registry**

Create `src/tracker/robots/registry.py` with:
- `RobotAdapter` dataclass holding:
  - `task_id: str`
  - `robot_id: str` (e.g., `adam_sp`, `adam_pro`)
  - `variant: str` (e.g., `23`, `29`)
  - `validate_motion_npz(path: Path) -> None`
  - optional `prepare_motion_npz(path: Path) -> Path`
- `get_adapter_for_task_id(task_id: str) -> RobotAdapter`
- `list_known_task_ids() -> list[str]`

Populate registry with existing Adam-SP task IDs using current modules:
- `tracker.robots.adam_sp`
- `tracker.robots.adam_sp_29`

**Step 4: Run test to verify it passes**

Run: `uv run pytest -q tests/test_robot_registry.py::test_robot_registry_lists_adam_sp_tasks`
Expected: PASS

---

### Task 2: Refactor CLI motion handling to use registry (no behavior change)

**Files:**
- Modify: `src/tracker/cli/common.py`
- Test: `tests/test_robot_registry.py`

**Step 1: Add a failing test (CLI dispatch works for existing tasks)**

Append to `tests/test_robot_registry.py`:

```py
def test_cli_validate_motion_dispatches(monkeypatch, tmp_path):
  import numpy as np

  # Minimal file; actual schema checked by robot module.
  motion = tmp_path / "m.npz"
  np.savez(motion, joint_pos=np.zeros((1, 1), dtype=np.float32))

  from tracker.cli.common import validate_motion_for_task

  # Expect: unknown schema raises, but dispatch layer should not raise "unknown task".
  try:
    validate_motion_for_task("Tracker-Tracking-Flat-Adam-SP-23", motion)
  except Exception as exc:
    assert "Unknown robot for task_id" not in str(exc)
```

**Step 2: Run test to verify it fails (if still hard-coded)**

Run: `uv run pytest -q tests/test_robot_registry.py::test_cli_validate_motion_dispatches`
Expected: FAIL (still hard-coded / unknown mapping in future steps)

**Step 3: Implement refactor**

Change `src/tracker/cli/common.py`:
- Remove `TASK_ID_ADAM_SP_*` constants or keep as thin aliases.
- Implement `validate_motion_for_task()` and `prepare_motion_for_task()` by calling
  `tracker.robots.registry.get_adapter_for_task_id(task_id)`.

**Step 4: Run test**

Run: `uv run pytest -q tests/test_robot_registry.py::test_cli_validate_motion_dispatches`
Expected: PASS

---

### Task 3: Factor shared tracking env-cfg patching for robot variants

**Files:**
- Modify: `src/tracker/tasks/tracking/env_cfg.py`
- Create: `src/tracker/tasks/tracking/patch_tracking_cfg.py`
- Test: `tests/test_tracking_env_cfg_smoke.py`

**Step 1: Write failing test (env cfg builds for both Adam-SP variants)**

Create: `tests/test_tracking_env_cfg_smoke.py`

```py
def test_env_cfg_builds_for_adam_sp_variants():
  from tracker.tasks.tracking.env_cfg import (
    adam_sp_flat_tracking_env_cfg,
    adam_sp_29_flat_tracking_env_cfg,
  )

  assert adam_sp_flat_tracking_env_cfg() is not None
  assert adam_sp_29_flat_tracking_env_cfg() is not None
```

**Step 2: Run test to verify it passes before refactor**

Run: `uv run pytest -q tests/test_tracking_env_cfg_smoke.py::test_env_cfg_builds_for_adam_sp_variants`
Expected: PASS

**Step 3: Implement shared patch helper (no behavior change)**

Create `src/tracker/tasks/tracking/patch_tracking_cfg.py` with helpers like:
- `patch_tracking_robot(cfg, *, get_robot_cfg, tracking_anchor_body, tracking_body_names, action_scale, ...)`

Update `src/tracker/tasks/tracking/env_cfg.py` to:
- Keep the two existing functions, but have them delegate to the shared helper.

**Step 4: Run tests**

Run: `uv run pytest -q tests/test_tracking_env_cfg_smoke.py`
Expected: PASS

---

### Task 4: Add viewer/inspection helper CLI (`tracker-view-robot`)

**Files:**
- Create: `src/tracker/cli/view_robot.py`
- Modify: `pyproject.toml` (add console script)
- Test: `tests/test_view_robot_cli_smoke.py`

**Step 1: Decide minimal interface**

Support:
- `tracker-view-robot --robot-id <adam_sp|adam_pro> --variant <23|29>`
- optional `--headless` (renders offscreen or just builds model)

**Step 2: Write failing test (CLI module imports)**

Create: `tests/test_view_robot_cli_smoke.py`

```py
def test_view_robot_cli_imports():
  import tracker.cli.view_robot  # noqa: F401
```

**Step 3: Implement minimal CLI**

In `src/tracker/cli/view_robot.py`:
- Parse args (tyro or argparse; keep consistent with existing CLIs).
- Build a robot cfg via robot module (e.g., `get_robot_cfg()`).
- Instantiate a minimal mjlab env or directly load the MuJoCo model to show:
  - collisions on/off sanity
  - actuator settings visible

**Step 4: Wire console script**

Update `pyproject.toml` `[project.scripts]`:
- Add `tracker-view-robot = "tracker.cli.view_robot:main"`

**Step 5: Run tests**

Run: `uv run pytest -q tests/test_view_robot_cli_smoke.py`
Expected: PASS

---

### Task 5: Add Adam-Pro robot IDs + task IDs (constants only)

**Files:**
- Modify: `src/tracker/cli/common.py`
- Modify: `src/tracker/robots/registry.py`
- Test: `tests/test_robot_registry.py`

**Step 1: Add tests for future Adam-Pro IDs**

Extend `tests/test_robot_registry.py`:

```py
def test_robot_registry_includes_adam_pro_task_ids():
  from tracker.robots.registry import list_known_task_ids

  task_ids = set(list_known_task_ids())
  assert "Tracker-Tracking-Flat-Adam-Pro-23" in task_ids
  assert "Tracker-Tracking-Flat-Adam-Pro-29" in task_ids
```

**Step 2: Make it fail**

Run: `uv run pytest -q tests/test_robot_registry.py::test_robot_registry_includes_adam_pro_task_ids`
Expected: FAIL

**Step 3: Add placeholder adapters that raise “not implemented”**

In `src/tracker/robots/registry.py`, register Adam-Pro task IDs but have adapter functions raise
`NotImplementedError("adam_pro not implemented yet")`.

**Step 4: Run test**

Run: `uv run pytest -q tests/test_robot_registry.py::test_robot_registry_includes_adam_pro_task_ids`
Expected: PASS

---

### Task 6: Add Adam-Pro packaged assets (MJCF + meshes)

**Files:**
- Add: `src/tracker/assets/adam_pro/adam_pro.xml`
- Add: `src/tracker/assets/adam_pro/adam_pro_29dof.xml`
- Add: `src/tracker/assets/adam_pro/meshes/**`
- Modify: `tests/test_assets_paths.py`

**Step 1: Copy assets**
- Copy provided Adam-Pro assets into the target paths above.

**Step 2: Extend assets resolver test**

Update `tests/test_assets_paths.py` to also assert:
- `get_asset_path("adam_pro", "adam_pro.xml")` exists
- `get_asset_path("adam_pro", "adam_pro_29dof.xml")` exists

**Step 3: Run tests**

Run: `uv run pytest -q tests/test_assets_paths.py`
Expected: PASS

---

### Task 7: Implement Adam-Pro (23/29) robot modules and strict motion validation

**Files:**
- Create: `src/tracker/robots/adam_pro_constants.py`
- Create: `src/tracker/robots/adam_pro.py`
- Create: `src/tracker/robots/adam_pro_29_constants.py`
- Create: `src/tracker/robots/adam_pro_29.py`
- Modify: `src/tracker/robots/registry.py`
- Test: `tests/test_motion_validation_adam_pro.py`

**Step 1: Write failing motion schema tests**

Create: `tests/test_motion_validation_adam_pro.py` with minimal `.npz` writers analogous to Adam-SP tests:
- `validate_motion_npz()` rejects missing keys
- shape checks enforce `(T, dof)` and `(T, nbody, 3/4)`

**Step 2: Implement constants + cfg construction**

Mirror Adam-SP patterns:
- Use `get_asset_path("adam_pro", ...)` to resolve MJCF.
- Define actuator groups, `armature`, `effort_limit`, PD gains.
- Define tracking anchor body + body set (robot module owns source of truth).

**Step 3: Implement strict motion validation**

`validate_motion_npz()` must:
- enforce required keys and expected shapes derived from MJCF spec
- enforce ordering is already correct (no reordering by name)

**Step 4: Update registry to real Adam-Pro adapters**

Replace placeholder `NotImplementedError` with real module calls.

**Step 5: Run tests**

Run: `uv run pytest -q tests/test_motion_validation_adam_pro.py`
Expected: PASS

---

### Task 8: Register Adam-Pro tracking tasks into mjlab

**Files:**
- Create: `src/tracker/tasks/tracking/register_adam_pro_23.py`
- Create: `src/tracker/tasks/tracking/register_adam_pro_29.py`
- Modify: `src/tracker/tasks/register_all.py`
- Modify: `src/tracker/tasks/tracking/env_cfg.py`
- Modify: `src/tracker/tasks/tracking/rl_cfg.py`
- Test: `tests/test_registry_smoke.py`

**Step 1: Add env cfg builders**
- Add `adam_pro_flat_tracking_env_cfg()` and `adam_pro_29_flat_tracking_env_cfg()` built via the shared patch helper.

**Step 2: Register tasks**
- Task IDs:
  - `Tracker-Tracking-Flat-Adam-Pro-23`
  - `Tracker-Tracking-Flat-Adam-Pro-29`

**Step 3: Update smoke test**
- Extend registry smoke tests to assert all 4 task IDs are present.

**Step 4: Run tests**

Run: `uv run pytest -q`
Expected: PASS

---

### Task 9: Calibrate wrist dynamics (29-DoF) from trusted baselines

**Files:**
- Modify: `src/tracker/robots/adam_sp_29_constants.py`
- Modify: `src/tracker/robots/adam_pro_29_constants.py`
- Test: `tests/test_robot_cfg_wrist_constants.py`

**Status:** Deferred until deployment-focused work with trusted wrist baselines (armature, effort limits, PD gains).

**Note:** While baselines are pending, keep a smoke test ensuring the expected wrist joints exist in the 29-DoF MJCFs (so later calibration edits can't silently “remove wrists”): `tests/test_wrist_joints_present.py`.

**Step 1: Write test that asserts no placeholder sentinel values**

Create: `tests/test_robot_cfg_wrist_constants.py`

```py
def test_29dof_wrist_dynamics_are_not_placeholders():
  from tracker.robots import adam_sp_29_constants, adam_pro_29_constants

  assert adam_sp_29_constants.WRIST_ARMATURE is not None
  assert adam_pro_29_constants.WRIST_ARMATURE is not None
```

**Step 2: Implement constants**
- Extract wrist armature / effort limit / PD gains from the provided trusted source and encode as explicit constants.

**Step 3: Run tests**

Run: `uv run pytest -q tests/test_robot_cfg_wrist_constants.py`
Expected: PASS

---

### Task 10: Packaging sanity (wheel includes Adam-Pro assets)

**Files:**
- None (expected to already be covered by package data rules)

**Note:** `src/tracker/assets/adam_pro/adam_pro_stl.xml` and `src/tracker/assets/adam_pro/adam_pro.urdf` are local-only refinement references; exclude them from sdist/wheel (and keep them gitignored).

**Step 1: Build wheel**

Run: `uv build`
Expected: wheel builds.

**Step 2: Validate asset resolution from wheel**

Run:
- `python -c "from tracker.assets.paths import get_asset_path; print(get_asset_path('adam_pro','adam_pro.xml'))"`

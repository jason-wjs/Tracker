# Tracking Registration Consolidation Implementation Plan

> **For Claude/ Codex:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Simplify `tracker.tasks.tracking` by removing legacy shim modules and consolidating tracking task registration into a single `register.py`, keeping only `config/` and `register.py` as the public surface.

**Architecture:** `tracker.tasks.tracking.config/*` contains pure builders (env/rl/patch). `tracker.tasks.tracking.register` contains import-time side effects and registers all tracking tasks from a small table. Delete legacy shim modules (`env_cfg.py`, `rl_cfg.py`, `patch_tracking_cfg.py`) and per-task registration modules (`register_adam_*`) since import-path stability is no longer required.

**Tech Stack:** Python, `pytest`.

---

### Task 1: Update tests to new public API

**Files:**
- Modify: `tests/test_tracking_module_imports.py`

**Step 1: Write failing test (new imports)**

Change the test to import:
- `tracker.tasks.tracking.config.env`
- `tracker.tasks.tracking.config.rl`
- `tracker.tasks.tracking.config.patch`

and assert key callables exist.

**Step 2: Run test to verify it fails**

Run: `uv run pytest -q tests/test_tracking_module_imports.py::test_tracking_import_paths_are_stable`  
Expected: FAIL once we delete the old shim modules without updating the test.

**Step 3: Implement minimal fixes**

Update the imports in the test to match the new API.

**Step 4: Run test to verify it passes**

Run: `uv run pytest -q tests/test_tracking_module_imports.py::test_tracking_import_paths_are_stable`  
Expected: PASS.

---

### Task 2: Consolidate task registration into `tracking/register.py`

**Files:**
- Modify: `src/tracker/tasks/tracking/register.py`
- Modify: `src/tracker/tasks/register_all.py`

**Step 1: Write failing smoke test (bootstrap registers all tasks)**

Existing: `tests/test_registry_smoke.py` already asserts all 4 tasks are present.

Run: `uv run pytest -q tests/test_registry_smoke.py::test_bootstrap_registers_adam_sp_task`  
Expected: FAIL if registration doesn’t run.

**Step 2: Implement consolidated registration**

In `src/tracker/tasks/tracking/register.py`:
- Import `register_mjlab_task`, `MotionTrackingOnPolicyRunner`.
- Define a small `TASK_SPECS` list of tuples containing:
  - `task_id`
  - `env_cfg_fn`
  - `play_env_cfg_fn`
  - `rl_cfg_fn`
- Loop and call `register_mjlab_task(...)`.

Update `src/tracker/tasks/register_all.py` to import only `tracker.tasks.tracking.register` (single import-time side effect).

**Step 3: Run smoke test**

Run: `uv run pytest -q tests/test_registry_smoke.py::test_bootstrap_registers_adam_sp_task`  
Expected: PASS.

---

### Task 3: Delete legacy tracking shim modules and per-task registrars

**Files:**
- Delete: `src/tracker/tasks/tracking/env_cfg.py`
- Delete: `src/tracker/tasks/tracking/rl_cfg.py`
- Delete: `src/tracker/tasks/tracking/patch_tracking_cfg.py`
- Delete: `src/tracker/tasks/tracking/register_adam_sp_29.py`
- Delete: `src/tracker/tasks/tracking/register_adam_pro_23.py`
- Delete: `src/tracker/tasks/tracking/register_adam_pro_29.py`
- Delete: `src/tracker/tasks/tracking/registration/`

**Step 1: Find and update imports**

Search for imports of removed modules and update them to:
- `tracker.tasks.tracking.config.env`
- `tracker.tasks.tracking.config.rl`
- `tracker.tasks.tracking.config.patch`

**Step 2: Run full test suite**

Run: `uv run pytest -q`  
Expected: PASS.


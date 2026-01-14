# Tracking Module Structure Refactor Implementation Plan

> **For Claude/ Codex:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make `tracker.tasks.tracking` easier to navigate by separating config builders vs registration side effects, while keeping existing import paths stable.

**Architecture:** Introduce `tracker.tasks.tracking.config/` for “pure” builders (env/rl/patch) and `tracker.tasks.tracking.registration/` for import-time task registration modules. Keep `env_cfg.py`, `rl_cfg.py`, `patch_tracking_cfg.py`, and existing `register*.py` files as stable shims that forward to the new locations. (We cannot use a `tracking/register/` package because it would shadow the existing `tracking/register.py` module and break stable imports.)

**Tech Stack:** Python, `pytest`.

---

### Task 1: Add import-path stability smoke tests

**Files:**
- Create: `tests/test_tracking_module_imports.py`

**Step 1: Write the failing test**

```py
def test_tracking_import_paths_are_stable():
  import tracker.tasks.tracking.env_cfg as env_cfg
  import tracker.tasks.tracking.rl_cfg as rl_cfg
  import tracker.tasks.tracking.patch_tracking_cfg as patch_tracking_cfg

  assert callable(env_cfg.adam_sp_flat_tracking_env_cfg)
  assert callable(rl_cfg.adam_sp_23_tracking_ppo_runner_cfg)
  assert callable(patch_tracking_cfg.make_flat_tracking_env_cfg_for_robot)
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest -q tests/test_tracking_module_imports.py::test_tracking_import_paths_are_stable`  
Expected: FAIL (after we move code without shims).

**Step 3: Commit (optional)**

Skip unless requested.

---

### Task 2: Create `tracking/config/` and move config implementation

**Files:**
- Create: `src/tracker/tasks/tracking/config/__init__.py`
- Create: `src/tracker/tasks/tracking/config/env.py`
- Create: `src/tracker/tasks/tracking/config/patch.py`
- Create: `src/tracker/tasks/tracking/config/rl.py`
- Modify: `src/tracker/tasks/tracking/env_cfg.py`
- Modify: `src/tracker/tasks/tracking/patch_tracking_cfg.py`
- Modify: `src/tracker/tasks/tracking/rl_cfg.py`

**Step 1: Move code into config modules**
- `config/patch.py`: contains `make_flat_tracking_env_cfg_for_robot`.
- `config/env.py`: contains `adam_*_flat_tracking_env_cfg` and DR constants.
- `config/rl.py`: contains `*_tracking_ppo_runner_cfg`.

**Step 2: Add shims**
- `env_cfg.py`, `patch_tracking_cfg.py`, `rl_cfg.py` re-export names from `config/*` so import paths remain stable.

**Step 3: Run tests**

Run: `uv run pytest -q tests/test_tracking_module_imports.py`  
Expected: PASS.

**Step 4: Commit (optional)**

Skip unless requested.

---

### Task 3: Create `tracking/registration/` and move registration modules

**Files:**
- Create: `src/tracker/tasks/tracking/registration/__init__.py`
- Create: `src/tracker/tasks/tracking/registration/adam_sp_23.py`
- Create: `src/tracker/tasks/tracking/registration/adam_sp_29.py`
- Create: `src/tracker/tasks/tracking/registration/adam_pro_23.py`
- Create: `src/tracker/tasks/tracking/registration/adam_pro_29.py`
- Modify: `src/tracker/tasks/tracking/register.py`
- Modify: `src/tracker/tasks/tracking/register_adam_sp_29.py`
- Modify: `src/tracker/tasks/tracking/register_adam_pro_23.py`
- Modify: `src/tracker/tasks/tracking/register_adam_pro_29.py`
- Modify: `src/tracker/tasks/register_all.py` (only if import paths change)

**Step 1: Move registration side effects**
- New `tracking/register/*.py` should call `register_mjlab_task(...)` at import time.

**Step 2: Add shims**
- Existing `register*.py` modules should just import their `tracking/register/*` counterpart to preserve import paths and side effects.

**Step 3: Run tests**

Run: `uv run pytest -q`  
Expected: PASS.

**Step 4: Commit (optional)**

Skip unless requested.

# Phase 1 (Adam-SP Tracking) Implementation Plan

> **For Claude/ Codex:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Status (2026-01-08):** Phase 1 implemented + verified. This document is now a record of what was built and the baseline for Phase 2 planning.

**Addendum (2026-01-09):** Adam-SP-29 hard-motion robustness work is planned below (collision simplification), because some aggressive offline clips can overflow mjlab’s fixed MJWarp contact/constraint buffers (`nconmax=35`, `njmax=250`). The fix must be in the robot collision model (do **not** change `nconmax/njmax`).

**Goal:** Deliver a standalone `tracker` repo/package that registers Adam-SP tracking tasks into `mjlab`, provides `tracker-*` wrapper CLIs, supports training + play with either local `--motion-file` or W&B motion resolution (mutually exclusive), and packages Adam-SP assets (~100MB) as read-only runtime data.

**Architecture:** `tracker` is an installable Python package (src layout) that plugs into `mjlab` via the `mjlab.tasks` entry point group. Task configs are motion-agnostic and patch `mjlab` tracking defaults; wrapper CLIs inject the selected motion source at runtime and delegate to `mjlab` CLIs where possible. Offline training uses a tracker-owned path to bypass `mjlab`’s W&B-only tracking motion resolution.

**Tech Stack:** Python, `uv`, `mjlab` (pinned by SHA), MuJoCo, rsl-rl, optional W&B.

## Acceptance Criteria (“Done”)
- `tracker-list-envs` lists Adam-SP tracking task IDs:
  - `Tracker-Tracking-Flat-Adam-SP-23`
  - `Tracker-Tracking-Flat-Adam-SP-29`
- `tracker-play <task_id> --motion-file <local.npz>` runs (single GPU/CPU) and validates motion compatibility before starting.
- `tracker-play <task_id> --registry-name <wandb artifact>` runs (mutual exclusion enforced vs `--motion-file`).
- `tracker-train <task_id> --motion-file <local.npz>` runs on CPU/single GPU without requiring W&B motion download.
- `tracker-train <task_id> --registry-name <wandb artifact>` delegates to `mjlab` train (mutual exclusion enforced vs `--motion-file`).
- Adam-SP assets are loadable from an installed wheel (no absolute-path assumptions).
- For the 29-DoF task, training on “hard” motions should not emit MJWarp buffer overflow warnings at scale (e.g., `--env.scene.num-envs 4096`) **without** increasing `nconmax/njmax`.

## Notes / Constraints
- Do not modify `mjlab` source.
- Phase 1: single GPU only for offline training; multi-GPU/torchrunx deferred.
- Robot module owns canonical tracking body set and motion compatibility validation.
- Assets are read-only runtime inputs once `adam_sp_constants.py` is established.
- For faithful MuJoCo dynamics, Adam-SP must define motor reflected inertia (`armature`) explicitly (do not rely on missing/zero values in the MJCF).
- Repo hygiene: do not commit local caches/venvs (`.venv/`, `.uv-cache/`, `.wandb/`, `.tracker-cache/`, `logs/`, `artifacts/`, `dist/`); keep `UV_CACHE_DIR` inside the repo for sandbox safety.
- Phase 2 (deferred): add a viewer/inspection helper (either `__main__` block or a `tracker-view-robot` CLI) to quickly validate assets, collisions, and actuator edits interactively.
- Newer offline datasets may target a 29-DoF Adam-SP variant (wrists enabled). Treat this as a separate task ID + separate robot/MJCF variant (Task 9).

## Domain Randomization (Current Phase 1 Setting)

This is the current Adam-SP tracking DR setting implemented in `tracker/src/tracker/tasks/tracking/env_cfg.py`.

**Baseline events (from `mjlab` tracking):**
- `push_robot` (interval): `(1.0, 3.0)` seconds; uses `VELOCITY_RANGE` with 6D components `x/y/z/roll/pitch/yaw`.
- `encoder_bias` (startup): joint encoder bias `(-0.01, 0.01)` radians (per joint).
- `foot_friction` (startup): `geom_friction` `operation="abs"` with range `(0.3, 1.2)` on Adam-SP foot contact geoms:
  - 23-DoF: `toeLeft_collision`, `toeRight_collision`
  - 29-DoF: `left_foot*_collision`, `right_foot*_collision`
- `base_com` (startup): `body_ipos` `operation="add"` on `("torso",)` with per-axis ranges `{0: (-0.02, 0.02), 1: (-0.02, 0.02), 2: (-0.02, 0.02)}` meters.

**Added physics DR (startup):**
- `link_com`: `body_ipos` `operation="add"` on the whitelist bodies below with x/y/z `(-0.02, 0.02)` meters.
- `physics_body_mass`: `body_mass` `operation="scale"`, `distribution="uniform"`, range `(0.9, 1.1)` on the whitelist bodies below.
- `physics_body_inertia`: `body_inertia` `operation="scale"`, `distribution="uniform"`, range `(0.9, 1.1)` on the whitelist bodies below.
- `physics_dof_damping`: `dof_damping` `operation="scale"`, `distribution="uniform"`, range `(0.6, 1.4)` on all DOFs (effective because `tracker` sets a nonzero damping baseline via `spec.joint(...).damping`).
- `physics_dof_frictionloss`: `dof_frictionloss` `operation="scale"`, `distribution="log_uniform"`, range `(0.2, 1.5)` on all DOFs (currently no effect because the packaged MJCF has `dof_frictionloss=0`).

**Whitelist bodies used by `link_com` / `physics_body_mass` / `physics_body_inertia`:**
- `pelvis`, `hipPitchLeft`, `hipRollLeft`, `thighLeft`, `shinLeft`, `hipPitchRight`, `hipRollRight`, `thighRight`, `shinRight`, `shoulderPitchLeft`, `shoulderRollLeft`, `shoulderYawLeft`, `elbowLeft`, `shoulderPitchRight`, `shoulderRollRight`, `shoulderYawRight`, `elbowRight`

**Play-mode DR semantics (current):**
- Keeps all startup DR enabled; disables observation corruption; removes `push_robot`.

---

### Task 1: Create package metadata + `uv` workflow

**Files:**
- Create: `tracker/pyproject.toml`
- Create: `tracker/README.md`
- Create: `tracker/.python-version` (optional)

**Step 1: Define distribution + deps**
- Set package import name to `tracker` (`src/tracker/`).
- Add `mjlab` dependency as VCS pinned by SHA (placeholder to fill with real SHA).
- Add dev deps: `pytest`, `ruff`, `pyright` (minimal).

**Step 2: Register plugin entry point**
- Add `[project.entry-points."mjlab.tasks"]` mapping that imports `tracker.tasks.register_all` (or equivalent) to register tasks on `import mjlab`.

**Step 3: Define console scripts**
- Add `[project.scripts]`: `tracker-train`, `tracker-play`, `tracker-list-envs`.

**Step 4: Document setup/run**
- `README.md` documents: `uv sync`, `tracker-list-envs`, example invocations for local/W&B motion.

**Test/Verify (local):**
- Run: `uv sync --group dev`
- Run: `python -c "import tracker; print('ok')"`

---

### Task 2: Establish src-layout package skeleton (no behavior yet)

**Files:**
- Create: `tracker/src/tracker/__init__.py`
- Create: `tracker/src/tracker/cli/__init__.py`
- Create: `tracker/src/tracker/integrations/mjlab/__init__.py`
- Create: `tracker/src/tracker/tasks/__init__.py`
- Create: `tracker/src/tracker/tasks/register_all.py`
- Create: `tracker/src/tracker/tasks/tracking/__init__.py`
- Create: `tracker/src/tracker/tasks/tracking/register.py`
- Create: `tracker/src/tracker/tasks/tracking/env_cfg.py`
- Create: `tracker/src/tracker/tasks/tracking/rl_cfg.py`
- Create: `tracker/src/tracker/robots/__init__.py`
- Create: `tracker/src/tracker/robots/adam_sp.py`
- Create: `tracker/src/tracker/robots/adam_sp_constants.py`
- Create: `tracker/src/tracker/motion/__init__.py`
- Create: `tracker/src/tracker/motion/sources.py`
- Create: `tracker/src/tracker/assets/__init__.py`
- Create: `tracker/src/tracker/assets/paths.py`

**Step 1: Wire imports**
- `tracker.tasks.register_all` should import tracking registration modules (import-time registration).

**Step 2: Keep boundaries**
- No file I/O in `tasks/*`.
- CLI parsing + I/O only in `cli/*`.

**Test/Verify (local):**
- Run: `python -c "import tracker.tasks.register_all; print('ok')"`

---

### Task 3: Asset relocation + path resolver (Phase 1 = pure path utility)

**Files:**
- Move: `/home/humanoid/Projects/Junsong_WU/ADAM/TRACKER/adam_sp` → `tracker/src/tracker/assets/adam_sp/` (exact final subpath may be adjusted)
- Modify: `tracker/pyproject.toml` (include package data for `tracker/assets/**`)
- Implement: `tracker/src/tracker/assets/paths.py`

**Step 1: Implement pure resolver**
- Provide `get_asset_path(robot_id: str, relative_path: str) -> Path` using package resources so it works from wheel and from source.

**Step 2: Package data configuration**
- Ensure wheel/sdist includes `tracker/assets/**` and that runtime access is not based on cwd.

**Test/Verify (local):**
- Run: `python -c "from tracker.assets.paths import get_asset_path; print(get_asset_path('adam_sp','adam_sp.xml'))"`

---

### Task 4: Adam-SP robot dynamics description module

**Files:**
- Implement: `tracker/src/tracker/robots/adam_sp_constants.py`
- Implement: `tracker/src/tracker/robots/adam_sp.py`

**Step 1: Mirror mjlab robot-constants pattern**
- Create functions analogous to `mjlab.asset_zoo.robots.*_constants.get_*_robot_cfg()`:
  - resolve MJCF path via `tracker.assets.paths.get_asset_path(...)`
  - define actuator parameters (including motor `armature`) / collision presets / initial keyframe state as needed
  - return an `mjlab`-compatible robot entity config object (exact type per mjlab patterns)

**Step 1a: Set motor reflected inertia (`armature`) via actuator configs**
- In `mjlab`, `Builtin*ActuatorCfg.armature` is applied at build-time by mutating the MuJoCo spec (`spec.joint(name).armature = armature`), so the simulator’s mass matrix includes the motor reflected inertia.
- In `tracker`, set these values in `tracker/src/tracker/robots/adam_sp_constants.py` by assigning `armature=...` on each `BuiltinPositionActuatorCfg` group (preferred), not by editing `adam_sp.xml` (assets remain read-only).
- Use the same actuator grouping strategy as mjlab robots (e.g., leg pitch/roll/yaw, ankle, waist, shoulder, elbow) so each group can carry its own `effort_limit`, `stiffness/damping`, and `armature`.

**Armature baseline (confirmed)**
- Use these per-actuator-group armature values (MuJoCo `joint.armature`, scalar per hinge):
  - `ARMATURE_130_92_7_P = 0.13426` (hip/knee pitch)
  - `ARMATURE_80_20_30_S = 0.281573` (hip roll)
  - `ARMATURE_60_17_50_S = 0.23409` (hip yaw + waist)
  - `ARMATURE_50_52_30_P = 0.0549` (ankle pitch/roll)
  - `ARMATURE_50_14A_50_S = 0.1578807` (shoulder pitch/roll)
  - `ARMATURE_30_14A_50_S = 0.0423963` (shoulder yaw + elbow)

**PD gains baseline (confirmed)**
- Express gains as `STIFFNESS_*` / `DAMPING_*` constants (mjlab convention), and split actuator groups when gains differ:
  - Hip/knee pitch: `stiffness=305.0`, `damping=5.0` (use `*_130_92_7_P`)
  - Hip roll: `stiffness=255.0`, `damping=3.5` (use `*_80_20_30_S`)
  - Hip yaw: `stiffness=255.0`, `damping=3.5` (use `*_60_17_50_S`)
  - Ankle pitch: `stiffness=50.0`, `damping=0.8` (separate from ankle roll)
  - Ankle roll: `stiffness=30.0`, `damping=0.35`
  - Waist roll: `stiffness=255.0`, `damping=3.5` (separate from waist pitch)
  - Waist pitch: `stiffness=305.0`, `damping=5.0`
  - Waist yaw: `stiffness=255.0`, `damping=3.5`
  - Shoulders (pitch/roll/yaw) + elbows: `stiffness=40.0`, `damping=1.0`

```py
# PD gains (mjlab-style constants).
STIFFNESS_130_92_7_P = 305.0
DAMPING_130_92_7_P = 5.0

STIFFNESS_80_20_30_S = 255.0
DAMPING_80_20_30_S = 3.5

STIFFNESS_60_17_50_S = 255.0
DAMPING_60_17_50_S = 3.5
STIFFNESS_60_17_50_S_WAIST_PITCH = 305.0
DAMPING_60_17_50_S_WAIST_PITCH = 5.0

STIFFNESS_50_52_30_P_ANKLE_PITCH = 50.0
DAMPING_50_52_30_P_ANKLE_PITCH = 0.8
STIFFNESS_50_52_30_P_ANKLE_ROLL = 30.0
DAMPING_50_52_30_P_ANKLE_ROLL = 0.35

STIFFNESS_50_14A_50_S = 40.0
DAMPING_50_14A_50_S = 1.0

STIFFNESS_30_14A_50_S = 40.0
DAMPING_30_14A_50_S = 1.0
```

**Effort limits baseline (Phase 1)**
- Use motor torque maxima (N*m) as `effort_limit` for `BuiltinPositionActuatorCfg` groups:
  - Hip/knee pitch: `340.0`
  - Hip roll: `120.0`
  - Hip yaw + waist: `89.0`
  - Ankle pitch/roll: `46.0`
  - Shoulder pitch/roll: `60.0`
  - Shoulder yaw + elbow: `17.5`

**Collision baseline (Phase 1)**
- Align with mjlab’s approach:
  - Name collision geoms as `<mesh>_collision` in `adam_sp.xml` so they can be targeted by `CollisionCfg` patterns.
  - Keep MJCF defaults for what collides, but apply targeted overrides:
    - toe friction: `(0.6,)` for `toeLeft_collision` / `toeRight_collision`
    - disable hand/finger collisions via `^(L_|R_).*_collision$` (Phase 1)
  - Avoid `geom_names_expr=(r".*",)` because it can accidentally enable collisions on visual geoms and unnamed geoms.

**Step 2: Own tracking semantics**
- Define canonical `anchor_body_name` and `body_names` for tracking (robot module is source of truth).

**Step 3: Own motion compatibility validation**
- Provide `validate_motion_npz(path: Path) -> None` (or similar) that checks required keys/shapes and any robot-specific assumptions (joint count/order if representable).

**Test/Verify (local):**
- Run: `python -c "from tracker.robots.adam_sp import get_robot_cfg; print(get_robot_cfg())"`

---

### Task 5: Tracking task config (patch mjlab defaults)

**Files:**
- Implement: `tracker/src/tracker/tasks/tracking/env_cfg.py`
- Implement: `tracker/src/tracker/tasks/tracking/rl_cfg.py`

**Step 1: Patch base tracking env**
- Build from `mjlab.tasks.tracking.tracking_env_cfg.make_tracking_env_cfg()`.
- Set `cfg.scene.entities = {'robot': <adam_sp robot cfg>}`.
- Patch action scale and any robot-specific event/termination/viewer settings similarly to mjlab’s `g1` tracking config.
- Leave `motion_cmd.motion_file` empty/placeholder in registration-time config.
- Use robot module’s tracking `anchor_body_name`/`body_names` to patch `MotionCommandCfg`.

**Step 2: RL cfg**
- Start identical to mjlab tracking PPO config; keep minimal differences.

---

### Task 6: Task registration (`task_id` motion-agnostic)

**Files:**
- Implement: `tracker/src/tracker/tasks/tracking/register.py`
- Implement: `tracker/src/tracker/tasks/register_all.py`

**Step 1: Register Adam-SP 23-DoF and 29-DoF tasks**
- Task IDs encode robot/terrain/control only (no motion identity):
  - `Tracker-Tracking-Flat-Adam-SP-23`
  - `Tracker-Tracking-Flat-Adam-SP-29`
- Do not register a compatibility alias; use only the explicit `...-23` and `...-29` IDs.

**Step 2: Provide entry point import target**
- `tracker.tasks.register_all` is the module referenced by the `mjlab.tasks` entry point and used by wrappers for explicit bootstrap.

**Test/Verify (local):**
- Run: `python -c \"import mjlab; import tracker.tasks.register_all; from mjlab.tasks.registry import list_tasks; print([t for t in list_tasks() if 'Adam' in t or 'ADAM' in t])\"`

---

### Task 7: Wrapper CLIs (task-id-first; mutual exclusion; validation)

**Files:**
- Implement: `tracker/src/tracker/cli/list_envs.py`
- Implement: `tracker/src/tracker/cli/play.py`
- Implement: `tracker/src/tracker/cli/train.py`
- Implement: `tracker/src/tracker/cli/common.py`
- Implement: `tracker/src/tracker/integrations/mjlab/bootstrap.py`

**Step 1: Bootstrap**
- Ensure tracker tasks are registered before delegating: call `tracker.integrations.mjlab.bootstrap.bootstrap()`.

**Step 2: `tracker-list-envs`**
- Delegate to `mjlab.scripts.list_envs.main()` after bootstrap.

**Step 3: `tracker-play`**
- Parse `task_id` first (mjlab-style).
- Accept `--motion-file` and W&B identifiers (`--registry-name`, etc.) as mutually exclusive.
- If `--motion-file`, call `tracker.robots.adam_sp.validate_motion_npz(...)` before delegating.
- Inject `--motion_file <path>` into the delegated `mjlab play` argv when offline.

**Step 4: `tracker-train`**
- Same mutual exclusion.
- If W&B mode: delegate to `mjlab.scripts.train.main()` with argv injection.
- If local motion mode: use the offline-train path (next task).

---

### Task 8: Offline training path (single GPU only)

**Files:**
- Create: `tracker/src/tracker/integrations/mjlab/offline_train.py`

**Step 1: Implement minimal training loop without W&B motion download**
- Pattern: copy the structure of `mjlab.scripts.train.run_train` but remove the W&B motion-resolution block.
- Load env cfg from registry (`mjlab.tasks.registry.load_env_cfg`) and deep-copy.
- Set `MotionCommandCfg.motion_file` from local `--motion-file`.
- Instantiate env, wrap with `RslRlVecEnvWrapper`, create runner (use task’s `runner_cls`).
- Use CPU or single GPU; explicitly do not support torchrunx/multi-GPU in phase 1.

**Step 2: Logging**
- Keep behavior consistent with mjlab defaults when possible; avoid requiring wandb for offline motion.

---

### Task 9: Add Adam-SP 29-DoF variant (assets + robot + task)

**Motivation:** Some offline motion datasets are retargeted against a 29-DoF Adam-SP model (adds `wristYaw/Pitch/Roll` L/R). This is incompatible with the current 23-DoF Phase 1 model, so it must be represented as a separate robot variant + separate `task_id` to unblock offline training tests.

**Decision (locked):**
- Action space: **control all 29 joints** (wrists included).
- Motion alignment: **strict** — motion `.npz` must match the 29-DoF MJCF joint/body ordering; no reordering by `joint_names`/`body_names`.
- Wrist dynamics: **placeholders** for Phase 1.5 (calibrate later).

**Reference for joint names/order (retargeting source of truth):**
- `/home/humanoid/Projects/Junsong_WU/ADAM/BrainStorm/holosoma/src/holosoma_retargeting/models/adam_sp/adam_sp_29dof.urdf`
- MJCF used by retargeting (body/joint ordering must match motion `.npz`):
  - `/home/humanoid/Projects/Junsong_WU/ADAM/BrainStorm/holosoma/src/holosoma_retargeting/models/adam_sp/adam_sp_29dof.xml`

**Files (placeholders; exact naming TBD):**
- Add: `tracker/src/tracker/assets/adam_sp/adam_sp_29dof.xml` (copied from holosoma source above)
- Add: `tracker/src/tracker/robots/adam_sp_29_constants.py` (29-DoF spec; wrists enabled; placeholder wrist gains/armature/effort)
- Add: `tracker/src/tracker/robots/adam_sp_29.py` (tracking body set; `validate_motion_npz` expects `(T, 29)` joint arrays)
- Add: `tracker/src/tracker/tasks/tracking/register_adam_sp_29.py` (new `TASK_ID`: `Tracker-Tracking-Flat-Adam-SP-29`)
- Update: `tracker/src/tracker/tasks/tracking/register.py` (rename existing `TASK_ID` to `Tracker-Tracking-Flat-Adam-SP-23`)
- Update: `tracker/src/tracker/tasks/register_all.py` (register both 23-DoF and 29-DoF tasks)
- Update: `tracker/src/tracker/cli/common.py` (motion validation must dispatch by exact task ID, not `"Adam" in task_id`)
- Update tests: add smoke test that both task IDs are registered; add motion schema validation tests for both variants.

**Implementation notes:**
- Ensure the MJCF joint names and *non-free joint order* align with the URDF used by retargeting; motion generation must emit columns in that exact order.
- Renaming the existing Phase 1 task ID to `...-23` is a breaking change for any scripts/logging/W&B sweeps referencing the old ID; Phase 1 uses only the explicit IDs.

---

### Task 9b: Adam-SP-29 collision optimization (G1-style primitives; hard-motion stable)

**Motivation:** With aggressive 29-DoF offline clips (large torso tilt), the sim visits “fallen/scraping” states more often. Mesh-heavy collisions can generate too many contacts/constraints per-step and overflow mjlab’s fixed MJWarp buffers (`nconmax=35`, `njmax=250`). Since buffer sizes must remain unchanged, we must reduce worst-case contacts via a simplified collision model (G1-style).

**Constraints (locked):**
- Do **not** change mjlab tracking defaults, especially `SimulationCfg(nconmax=35, njmax=250)`.
- Keep wrist DOFs enabled for the 29-DoF task, but disable wrist/hand collisions (wrists are controlled, but collisions are off).
- Keep `adam_sp.xml` unchanged as the 23-DoF baseline until the 29-DoF variant is stable on hard clips; only then consider porting the collision approach back to 23-DoF.

**Chosen collision strategy (locked):**
- Foot collisions: **G1-like multi-capsule foot** (few simple contact geoms; no foot/toe mesh collisions).
- Ground contacts: allow **pelvis + torso primitives** to collide with ground (avoid “ghost torso”).
- Self-collision policy (Phase 1): **minimal** — disable broad self-collisions; re-enable selectively later if needed.

**Implementation phases (gradual):**
1) **Phase A — Restructure to match mjlab conventions (no behavior change)**
   - Ensure all visual meshes are `contype=0 conaffinity=0`.
   - Centralize collision defaults (e.g., `default class="collision"` like `g1.xml`).
2) **Phase B — Replace foot collisions with primitives (high impact)**
   - Add ~5–8 capsule geoms per foot (G1-style) and disable/remove toe/foot mesh collisions.
   - Ensure `foot_friction` DR targets the new foot capsule geoms (and no longer targets disabled mesh collisions).
3) **Phase C — Torso/pelvis primitives**
   - Replace pelvis/torso collision meshes with a small set of capsules/boxes.
4) **Phase D — Limbs primitives**
   - Replace thigh/shin/upper-arm/forearm collision meshes with 1–2 capsules per link.
5) **Phase E — Self-collision pruning**
   - Add `<contact><exclude .../></contact>` for adjacent links and common “always-near” pairs (e.g., upper-arm↔torso) to prevent self-contact spam during falls.

**Test/Verify (local):**
- Run 29-DoF offline train on a “hard” clip at scale and confirm no overflow warnings:
  - `uv run tracker-train Tracker-Tracking-Flat-Adam-SP-29 --motion-file <hard_clip.npz> --gpu-ids 0 --env.scene.num-envs 4096`

**Status (2026-01-09):** Implemented a first-pass hard-motion-safe collision configuration for Adam-SP-29:
- `adam_sp_29dof.xml`:
  - Added G1-style multi-capsule feet (`left_foot*_collision`, `right_foot*_collision`).
  - Capsule radius + layout is derived from the toe mesh AABB (`toeLeft.STL`, `toeRight.STL`) to avoid hand-tuned heuristics.
  - Preserved `toeTip/heelPad/midfootPad` bodies for motion compatibility, but removed their collision geoms.
  - Replaced `pelvis_collision` and `torso_collision` mesh collisions with simple box primitives (AABB-derived).
- `adam_sp_29_constants.py`:
  - For Phase 1 hard clips, enabled collisions only for: feet + pelvis + torso; disabled all other `*_collision` geoms.
- Verified: `--env.scene.num-envs 4096` runs without MJWarp overflow warnings on `sub10_largebox_049_mj_fps50.npz` for a short smoke run (`--agent.max-iterations 1`).

**Follow-up (Phase 2 candidate):** Gradually re-enable collisions (e.g., legs first) or replace limb collision meshes with capsules, keeping the no-overflow constraint.

---

### Task 10: Minimal tests (smoke-level)

**Files:**
- Create: `tracker/tests/test_registry_smoke.py`
- Create: `tracker/tests/test_assets_paths.py`

**Tests to include:**
- Importing `tracker` does not crash.
- Bootstrap registers tasks (registry contains expected task IDs).
- `get_asset_path` returns an existing path for Adam-SP XML.

**Run:**
- `uv run pytest -q`

---

### Task 11: Build/package sanity check (asset inclusion)

**Goal:** Ensure installed wheel can locate assets via resolver.

**Run (local):**
- `uv build`
- Create a clean venv and `pip install dist/*.whl`
- Run a short import snippet that resolves `adam_sp.xml` via `get_asset_path`.

**Verified (2026-01-08):**
- `uv build` produced `dist/tracker-0.1.0-py3-none-any.whl`
- Installing the wheel into a clean venv resolved both assets:
  - `get_asset_path("adam_sp", "adam_sp.xml")`
  - `get_asset_path("adam_sp", "adam_sp_29dof.xml")`

## Phase 1 Completion Summary

**Delivered**
- Installable `tracker` package (src-layout) that registers tasks into `mjlab` via `[project.entry-points."mjlab.tasks"]`.
- Wrapper CLIs: `tracker-list-envs`, `tracker-train`, `tracker-play`.
- Motion source selection at runtime:
  - offline/local via `--motion-file`
  - online via `--registry-name` (W&B artifact)
  - mutual exclusion enforced.
- Task IDs (explicit; no legacy alias):
  - `Tracker-Tracking-Flat-Adam-SP-23`
  - `Tracker-Tracking-Flat-Adam-SP-29`
- Offline training path that bypasses `mjlab`’s W&B-only tracking motion resolution.
- Adam-SP assets packaged into the wheel and resolved via `tracker.assets.paths.get_asset_path(...)`.
- 29-DoF variant support (wrists enabled) including strict motion validation and safe qpos/qvel → joint-array conversion for offline datasets.
- Domain randomization baseline recorded above and implemented in `tracker/src/tracker/tasks/tracking/env_cfg.py`.
- Phase 1 is kept minimal: removed the unused placeholder module `tracker.motion` (Phase 2 will introduce the real motion abstraction layer).

**Verified**
- `uv run pytest -q` passes (smoke-level tests).
- Offline training/play works with `--motion-file` for both task IDs (single env smoke run).
- W&B motion artifact path works with `--registry-name` (delegates to `mjlab`).
- `uv build` wheel installs and resolves assets (Task 11 verification above).

## Phase 2 Planning Seed (Draft)

**Primary goals**
- Multi-clip sampling within one run (Phase 1 is “one run per clip”; Phase 2 adds sampling across many clips).
- Offline dataset support as “one long `motion.npz`” with clip indexing/sampling utilities (after Phase 1 stability).
- Multi-GPU training support (torchrunx / distributed) for larger runs.
- Add a viewer/inspection helper (`tracker-view-robot` or equivalent) for fast asset validation and debugging.
- Introduce `tracker.motion` as the central motion abstraction layer (source resolution, caching/normalization, dataset indexing, multi-clip sampling).

**Robotics/dynamics evolution**
- Calibrate Adam-SP wrist dynamics for the 29-DoF variant (armature/effort limits/gains are placeholders in Phase 1).
- Decide whether to introduce nonzero MJCF `dof_damping` / `dof_frictionloss` (currently 0 in assets, so related DR terms have no effect).

**Extensibility / structure**
- Keep task IDs explicit per robot + DoF variant; avoid implicit aliases to reduce ambiguity.
- Continue treating assets as read-only runtime data; evolve dynamics via Python constants/config where possible.

---

## Issues

- **Sandbox network/proxy:** When network is restricted/broken, export proxy env vars (e.g., `http_proxy/https_proxy/all_proxy`) before `uv sync`/W&B artifact download.
  - Example:
    - `export http_proxy=http://127.0.0.1:7897 https_proxy=http://127.0.0.1:7897 all_proxy=http://127.0.0.1:7897`
    - `export HTTP_PROXY=$http_proxy HTTPS_PROXY=$https_proxy ALL_PROXY=$all_proxy`
- **W&B logging (offline `--motion-file`):** `tracker` will fall back to TensorBoard if it cannot detect W&B credentials. After `wandb login`, credentials are typically stored in `~/.netrc` (not `WANDB_API_KEY`), so `tracker` must detect that; this was fixed in `tracker/src/tracker/integrations/mjlab/offline_train.py` (normalize `WANDB_API_HOST` to a URL with scheme before calling `wandb`’s netrc helper).
- **CUDA availability:** Training requires a working CUDA setup for GPU runs; if `torch.cuda.is_available()` is false, offline training will fall back to CPU (much slower).
- **MJWarp buffer overflows (29-DoF, hard clips):** Some aggressive 29-DoF clips can trigger `narrowphase/nefc overflow` warnings at scale due to collision complexity. See Task 9b (collision optimization) — do not “fix” by raising `nconmax/njmax`.

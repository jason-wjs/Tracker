# Teleop Policy Observation History (K=8) Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task-by-task.

**Goal:** Improve multi-clip teleoperation training by giving the teleop policy a short observation history (frame stacking), without changing other tasks.

**Architecture:** Add a tracker-owned `VecEnv` wrapper that stacks the *entire* `policy` observation group over the last `K=8` frames. Enable the wrapper only for `Tracker-Teleop-*` tasks. Do not modify `mjlab/`.

**Tech Stack:** `mjlab` (env), `rsl_rl` (runner), `tensordict`, `torch`, `pytest`.

---

### Task 1: Add failing unit test for history stacking

**Files:**
- Create: `tracker/.worktrees/phase-2.5/tests/test_policy_history_vecenv_wrapper.py`

**Step 1: Write failing test**

Write a unit test with a minimal fake `VecEnv` that returns a `TensorDict` containing:
- `policy`: `(num_envs, obs_dim)` float tensor
- `critic`: any placeholder tensor (unused)

Test cases:
1. First `get_observations()` returns `policy` stacked as `[t, t, ..., t]` (repeat current) with `K=8`.
2. After a `step()`, the stacked tensor shifts: `[t+1, t, ..., t-(K-2)]` for non-done envs.
3. For envs that are `done` on that step, history becomes `[t_reset, t_reset, ..., t_reset]`.

**Step 2: Run test to verify it fails**

Run:
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q tests/test_policy_history_vecenv_wrapper.py`

Expected:
- FAIL because wrapper class does not exist.

---

### Task 2: Implement teleop-only policy history wrapper

**Files:**
- Create: `tracker/.worktrees/phase-2.5/src/tracker/rl/policy_history_vecenv.py`

**Step 1: Write minimal implementation**

Implement `PolicyHistoryVecEnvWrapper`:
- Wraps an `rsl_rl.env.VecEnv` (e.g., `mjlab.rl.RslRlVecEnvWrapper`).
- Stacks only the `policy` group.
- `K=8`, includes current (`[t, t-1, ..., t-7]`), flat concatenation on last dim.
- Reset mode: repeat current frame on reset/done envs.
- Delegates everything else (`num_envs`, `device`, `num_actions`, `unwrapped`, `close`, etc.).

**Step 2: Run test to verify it passes**

Run:
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q tests/test_policy_history_vecenv_wrapper.py`

Expected:
- PASS

**Step 3: Commit**

Run:
```bash
git add tests/test_policy_history_vecenv_wrapper.py src/tracker/rl/policy_history_vecenv.py
git commit -m "feat: add teleop policy observation history wrapper"
```

---

### Task 3: Enable wrapper only for `Tracker-Teleop-*` tasks

**Files:**
- Modify: `tracker/.worktrees/phase-2.5/src/tracker/integrations/mjlab/offline_train.py`
- Modify: `tracker/.worktrees/phase-2.5/src/tracker/cli/eval.py`
- Modify: `tracker/.worktrees/phase-2.5/src/tracker/cli/play.py`

**Step 1: Add a small helper**

Add a helper like:
- `is_teleop_task_id(task_id: str) -> bool`
- returns `task_id.startswith("Tracker-Teleop-")`

**Step 2: Wrap VecEnv**

After creating the `RslRlVecEnvWrapper`, wrap it with:
- `PolicyHistoryVecEnvWrapper(env, k=8, group="policy", reset_mode="repeat")`

Only when `is_teleop_task_id(task_id)` is true.

**Step 3: Sanity check by running the smallest CLI entry points**

Run:
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q`

Expected:
- PASS

**Step 4: Commit**

Run:
```bash
git add src/tracker/integrations/mjlab/offline_train.py src/tracker/cli/eval.py src/tracker/cli/play.py
git commit -m "feat: enable history-stacked obs for teleop tasks"
```

---

### Task 4: Manual smoke test

**Step 1: Train**

Run:
```bash
uv run tracker-train Tracker-Teleop-Flat-Adam-Pro-29-No-State-Estimation \
  --motion-pack /path/to/pack \
  --motion-split train \
  --gpu-ids 0 \
  --agent.max-iterations 50 \
  --env.scene.num-envs 128
```

Expected:
- The printed `Actor MLP` input feature size increases by ~8× for the policy observation set.

**Step 2: Play**

Run:
```bash
uv run tracker-play Tracker-Teleop-Flat-Adam-Pro-29-No-State-Estimation \
  --checkpoint-file /path/to/model.pt \
  --motion-pack /path/to/pack \
  --motion-split val \
  --device cuda:0 \
  --num-envs 4
```

Expected:
- Runs without shape mismatch errors.


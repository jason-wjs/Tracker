# Teleop HistoryEncoder Implementation Plan (Phase‑2.5)

> **For Codex:** REQUIRED SUB-SKILL: Use `superpowers:test-driven-development` to implement this plan test-first.

**Goal:** Add a TWIST2-style history encoder (Conv1D over raw history) for the teleop task to improve multi-clip universal controller training.

**Architecture:** Keep mjlab untouched. Provide a tracker-owned `ActorCritic` implementation (`TrackerActorCriticTeleop`) that:
1) parses the teleop policy observation into motion-context + proprio-current + proprio-history,
2) encodes proprio-history via `HistoryEncoder` (MLP projection + Conv1D),
3) feeds `[motion_context, proprio_current, history_latent]` into an actor backbone MLP,
4) leaves the critic as a standard MLP over critic observations.

**Tech Stack:** `rsl_rl` PPO, `mjlab` env wrapper, `torch` (Conv1D), tracker task registry.

---

### Task 1: Add failing tests for teleop history encoder policy

**Files:**
- Create: `tests/test_teleop_history_encoder_policy_shapes.py`
- Create: `tests/test_teleop_history_encoder_policy_registration.py`

**Step 1: Write failing shape test**

```python
def test_teleop_history_encoder_policy_shapes():
    # Instantiating TrackerActorCriticTeleop should be possible and .act/.evaluate should return correct shapes.
    ...
```

**Step 2: Run test to verify it fails**

Run:
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q tests/test_teleop_history_encoder_policy_shapes.py -q`

Expected:
- FAIL because `tracker.rl.teleop_actor_critic` (or `TrackerActorCriticTeleop`) does not exist yet.

**Step 3: Write failing registration test**

```python
def test_registers_policy_class_for_rsl_rl_eval():
    # OnPolicyRunner uses eval(class_name) in rsl_rl.runners.on_policy_runner module scope.
    ...
```

**Step 4: Run test to verify it fails**

Run:
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q tests/test_teleop_history_encoder_policy_registration.py -q`

Expected:
- FAIL because registration hook isn’t implemented.

---

### Task 2: Implement tracker-owned HistoryEncoder actor-critic

**Files:**
- Create: `src/tracker/rl/teleop_actor_critic.py`

**Step 1: Minimal implementation**
- Implement `HistoryEncoder` with `K=8` support (MLP projection → Conv1D → `history_latent`).
- Implement `TrackerActorCriticTeleop` with the same public interface as `rsl_rl.modules.ActorCritic`.
- Parse teleop policy observation layout:
  - motion context: `command (58) + motion_anchor_ori_b (6)` = 64 dims
  - history blocks (K frames):
    - `base_ang_vel` (3)
    - `joint_pos` (num_actions)
    - `joint_vel` (num_actions)
    - `actions` (num_actions)

**Step 2: Run tests to verify GREEN**

Run:
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q tests/test_teleop_history_encoder_policy_shapes.py -q`

Expected: PASS.

---

### Task 3: Wire teleop task to use the new class_name + rsl_rl eval injection

**Files:**
- Modify: `src/tracker/tasks/tracking/config/rl.py`
- Modify: `src/tracker/tasks/tracking/register.py`
- Modify: `tests/test_registry_smoke.py` (only if needed)

**Step 1: Add teleop runner cfg**
- Add `adam_pro_29_teleop_ppo_runner_cfg()` that copies `adam_pro_29_tracking_ppo_runner_cfg()` but sets:
  - `cfg.policy.class_name = "TrackerActorCriticTeleop"`

**Step 2: Add rsl_rl runner eval injection**
- In `tracker.rl.teleop_actor_critic`, add `register_rsl_rl_teleop_policy()` that:
  - imports `rsl_rl.runners.on_policy_runner` and sets `TrackerActorCriticTeleop` in that module dict.
- Call `register_rsl_rl_teleop_policy()` from `tracker.tasks.tracking.register` before task registration.

**Step 3: Run tests**

Run:
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q tests/test_teleop_history_encoder_policy_registration.py -q`
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q`

Expected: PASS.

---

### Task 4: Manual validation (non-test)

**Goal:** Ensure training starts and builds the policy with HistoryEncoder.

Run (example):
```bash
uv run tracker-train Tracker-Teleop-Flat-Adam-Pro-29-No-State-Estimation \
  --motion-pack /home/humanoid/Downloads/Data/GMR_test/bvh_test1/pack_adam_pro_29_bvh_test1 \
  --motion-split train \
  --gpu-ids 0 \
  --agent.max-iterations 20 \
  --env.scene.num-envs 128
```

Expected:
- Console prints show a policy class other than the plain `ActorCritic` MLP (i.e., `TrackerActorCriticTeleop` / history encoder layers).


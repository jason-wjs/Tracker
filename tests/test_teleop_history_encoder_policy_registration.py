import importlib

import pytest


def test_registers_policy_class_for_rsl_rl_eval():
  """OnPolicyRunner uses eval(policy_cfg['class_name']) in module globals.

  The tracker teleop HistoryEncoder policy must register itself into
  `rsl_rl.runners.on_policy_runner` globals to make eval succeed.
  """

  try:
    teleop_mod = importlib.import_module("tracker.rl.teleop_actor_critic")
  except ImportError as exc:
    pytest.fail(f"Missing module tracker.rl.teleop_actor_critic: {exc}")

  assert hasattr(
    teleop_mod, "register_rsl_rl_teleop_policy"
  ), "Expected register_rsl_rl_teleop_policy() helper"

  teleop_mod.register_rsl_rl_teleop_policy()

  on_policy_runner = importlib.import_module("rsl_rl.runners.on_policy_runner")
  assert hasattr(
    on_policy_runner, "TrackerActorCriticTeleop"
  ), "TrackerActorCriticTeleop not registered into rsl_rl.runners.on_policy_runner"


from __future__ import annotations

import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import torch
import tyro
from rsl_rl.runners import OnPolicyRunner

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

from tracker.cli.common import split_task_id
from tracker.integrations.mjlab.bootstrap import bootstrap
from tracker.tasks.tracking.config.patch import apply_motion_pack


@dataclass(frozen=True)
class EvalConfig:
  checkpoint: Path
  motion_pack: Path
  motion_split: Literal["val", "test"] = "val"
  num_episodes: int = 100
  num_envs: int | None = None
  device: str | None = None


def parse_eval_args(argv: list[str]) -> tuple[str, EvalConfig]:
  task_id, remaining = split_task_id(argv)
  cfg = tyro.cli(
    EvalConfig,
    args=remaining,
    prog=Path(sys.argv[0]).name + f" {task_id}",
    config=(
      tyro.conf.AvoidSubcommands,
      tyro.conf.FlagConversionOff,
    ),
  )
  return task_id, cfg


def run_eval(task_id: str, cfg: EvalConfig) -> None:
  bootstrap()

  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  env_cfg = load_env_cfg(task_id, play=True)
  agent_cfg = load_rl_cfg(task_id)

  apply_motion_pack(env_cfg, pack_dir=cfg.motion_pack, split=cfg.motion_split)

  if cfg.num_envs is not None:
    env_cfg.scene.num_envs = cfg.num_envs

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
  vec_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

  runner_cls = load_runner_cls(task_id) or OnPolicyRunner
  runner_kwargs = {}
  if runner_cls is not OnPolicyRunner:
    runner_kwargs["registry_name"] = None

  runner = runner_cls(vec_env, asdict(agent_cfg), log_dir=None, device=device, **runner_kwargs)
  runner.load(str(cfg.checkpoint), map_location=device)
  policy = runner.get_inference_policy(device=device)

  obs, _ = vec_env.reset()
  ep_ret = torch.zeros(vec_env.num_envs, device=vec_env.device)
  ep_len = torch.zeros(vec_env.num_envs, device=vec_env.device, dtype=torch.long)

  completed = 0
  returns: list[float] = []
  per_task: dict[int, list[float]] = {}

  while completed < cfg.num_episodes:
    actions = policy(obs)
    obs, rew, dones, _extras = vec_env.step(actions)
    ep_ret += rew
    ep_len += 1

    done_ids = (dones > 0).nonzero().flatten()
    if done_ids.numel() == 0:
      continue

    cmd = vec_env.unwrapped.command_manager.get_term("motion")
    task_ids = getattr(cmd, "task_id", None)
    if task_ids is None:
      task_ids = torch.full((vec_env.num_envs,), -1, device=vec_env.device, dtype=torch.long)

    for env_i in done_ids.tolist():
      if completed >= cfg.num_episodes:
        break
      r = float(ep_ret[env_i].item())
      returns.append(r)
      tid = int(task_ids[env_i].item())
      per_task.setdefault(tid, []).append(r)
      completed += 1

    ep_ret[done_ids] = 0.0
    ep_len[done_ids] = 0

  mean_return = sum(returns) / max(len(returns), 1)
  print(f"[EVAL] episodes={len(returns)} mean_return={mean_return:.4f} device={device}")
  for tid in sorted(per_task.keys()):
    vals = per_task[tid]
    print(f"[EVAL] task_id={tid} episodes={len(vals)} mean_return={sum(vals)/len(vals):.4f}")

  vec_env.close()


def main() -> None:
  task_id, cfg = parse_eval_args(sys.argv[1:])
  run_eval(task_id, cfg)


if __name__ == "__main__":
  main()

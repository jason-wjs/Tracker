from __future__ import annotations

import sys
from pathlib import Path

import tyro

from tracker.cli.common import (
  ensure_motion_source_exclusive,
  get_motion_file,
  prepare_motion_for_task,
  validate_motion_for_task,
  split_task_id,
)
from tracker.integrations.mjlab.bootstrap import bootstrap


def _strip_flag(argv: list[str], *, flags: tuple[str, ...]) -> list[str]:
  stripped: list[str] = []
  skip_next = False
  for arg in argv:
    if skip_next:
      skip_next = False
      continue
    if any(arg == f for f in flags):
      skip_next = True
      continue
    if any(arg.startswith(f + "=") for f in flags):
      continue
    stripped.append(arg)
  return stripped


def resolve_train_mode(task_id: str, argv: list[str]) -> tuple[str, list[str]]:
  bootstrap()
  ensure_motion_source_exclusive(argv)

  motion_file = get_motion_file(argv)
  if motion_file is not None:
    validate_motion_for_task(task_id, motion_file)
    remaining = _strip_flag(argv, flags=("--motion-file", "--motion_file"))
    return "offline", remaining

  return "wandb", argv


def run(task_id: str, argv: list[str]) -> None:
  mode, remaining = resolve_train_mode(task_id, argv)

  from mjlab.scripts.train import TrainConfig, launch_training

  args = tyro.cli(
    TrainConfig,
    args=remaining,
    default=TrainConfig.from_task(task_id),
    prog=Path(sys.argv[0]).name + f" {task_id}",
    config=(
      tyro.conf.AvoidSubcommands,
      tyro.conf.FlagConversionOff,
    ),
  )

  if mode == "wandb":
    launch_training(task_id=task_id, args=args)
    return

  if mode != "offline":
    raise RuntimeError(f"Unknown mode: {mode}")

  motion_file = get_motion_file(argv)
  assert motion_file is not None

  from tracker.integrations.mjlab.offline_train import launch_training_offline

  prepared_motion = prepare_motion_for_task(task_id, motion_file)

  launch_training_offline(
    task_id=task_id,
    env_cfg=args.env,
    agent_cfg=args.agent,
    motion_file=str(prepared_motion),
    gpu_ids=args.gpu_ids,
    wandb_run_path=args.wandb_run_path,
    torchrunx_log_dir=args.torchrunx_log_dir,
    enable_nan_guard=args.enable_nan_guard,
    video=args.video,
    video_length=args.video_length,
    video_interval=args.video_interval,
  )


def main() -> None:
  task_id, remaining = split_task_id(sys.argv[1:])
  run(task_id, remaining)


if __name__ == "__main__":
  main()

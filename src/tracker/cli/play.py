from __future__ import annotations

import sys
from pathlib import Path

import tyro

from tracker.cli.common import (
  ensure_motion_source_exclusive,
  get_flag_value,
  get_motion_file,
  prepare_motion_for_task,
  split_task_id,
  validate_motion_for_task,
)
from tracker.integrations.mjlab.bootstrap import bootstrap


def preflight(task_id: str, argv: list[str]) -> None:
  bootstrap()
  ensure_motion_source_exclusive(argv)

  motion_file = get_motion_file(argv)
  if motion_file is None:
    return

  agent = get_flag_value(argv, ("--agent",))
  if agent in {"zero", "random"}:
    raise ValueError(
      "Dummy agents require W&B motion resolution; use --registry-name (not --motion-file)."
    )

  validate_motion_for_task(task_id, motion_file)


def run(task_id: str, argv: list[str]) -> None:
  preflight(task_id, argv)

  from mjlab.scripts.play import PlayConfig, run_play

  cfg = tyro.cli(
    PlayConfig,
    args=argv,
    default=PlayConfig(),
    prog=Path(sys.argv[0]).name + f" {task_id}",
    config=(
      tyro.conf.AvoidSubcommands,
      tyro.conf.FlagConversionOff,
    ),
  )

  if cfg.motion_file is not None:
    prepared_motion = prepare_motion_for_task(task_id, Path(cfg.motion_file))
    cfg.motion_file = str(prepared_motion)

  run_play(task_id, cfg)


def main() -> None:
  task_id, remaining = split_task_id(sys.argv[1:])
  run(task_id, remaining)


if __name__ == "__main__":
  main()

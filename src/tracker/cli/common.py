from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable

TASK_ID_ADAM_SP_23 = "Tracker-Tracking-Flat-Adam-SP-23"
TASK_ID_ADAM_SP_29 = "Tracker-Tracking-Flat-Adam-SP-29"


def split_task_id(argv: list[str]) -> tuple[str, list[str]]:
  if not argv:
    raise SystemExit(f"Usage: {Path(sys.argv[0]).name} <task_id> [args...]")
  return argv[0], argv[1:]


def _normalize_flags(flags: Iterable[str]) -> tuple[str, ...]:
  out: list[str] = []
  for f in flags:
    if not f.startswith("--"):
      raise ValueError(f"Expected flag like '--name', got {f!r}")
    out.append(f)
    out.append(f.replace("-", "_"))
  return tuple(dict.fromkeys(out))


def get_flag_value(argv: list[str], flags: Iterable[str]) -> str | None:
  flags_tuple = _normalize_flags(flags)
  for idx, arg in enumerate(argv):
    for flag in flags_tuple:
      if arg == flag:
        if idx + 1 >= len(argv):
          raise ValueError(f"Flag {flag} requires a value")
        return argv[idx + 1]
      if arg.startswith(flag + "="):
        return arg.split("=", 1)[1]
  return None


def ensure_motion_source_exclusive(argv: list[str]) -> None:
  motion_file = get_flag_value(argv, ("--motion-file",))
  registry_name = get_flag_value(argv, ("--registry-name",))
  if motion_file is not None and registry_name is not None:
    raise ValueError("Use only one of --motion-file or --registry-name")


def get_motion_file(argv: list[str]) -> Path | None:
  value = get_flag_value(argv, ("--motion-file",))
  return Path(value) if value is not None else None


def get_registry_name(argv: list[str]) -> str | None:
  return get_flag_value(argv, ("--registry-name",))


def validate_motion_for_task(task_id: str, motion_file: Path) -> None:
  if task_id == TASK_ID_ADAM_SP_29:
    from tracker.robots.adam_sp_29 import validate_motion_npz

    validate_motion_npz(motion_file)
    return

  if task_id == TASK_ID_ADAM_SP_23:
    from tracker.robots.adam_sp import validate_motion_npz

    validate_motion_npz(motion_file)
    return

  raise ValueError(f"Unknown robot for task_id={task_id!r}; cannot validate motion")


def prepare_motion_for_task(task_id: str, motion_file: Path) -> Path:
  """Prepare a motion file for consumption by mjlab (may rewrite to a cached file)."""
  if task_id == TASK_ID_ADAM_SP_29:
    from tracker.robots.adam_sp_29 import prepare_motion_npz

    return prepare_motion_npz(motion_file)

  return motion_file

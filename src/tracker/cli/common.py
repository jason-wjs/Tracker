from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable

from tracker.robots.registry import TASK_ID_ADAM_SP_23, TASK_ID_ADAM_SP_29, get_adapter_for_task_id


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
  motion_pack = get_flag_value(argv, ("--motion-pack",))
  motion_split = get_flag_value(argv, ("--motion-split",))
  motion_pack_sampling_mode = get_flag_value(argv, ("--motion-pack-sampling-mode",))
  registry_name = get_flag_value(argv, ("--registry-name",))
  sources = [v is not None for v in (motion_file, motion_pack, registry_name)]
  if sum(sources) > 1:
    raise ValueError("Use only one of --motion-file, --motion-pack, or --registry-name")
  if motion_split is not None and motion_pack is None:
    raise ValueError("--motion-split requires --motion-pack")
  if motion_pack_sampling_mode is not None and motion_pack is None:
    raise ValueError("--motion-pack-sampling-mode requires --motion-pack")


def get_motion_file(argv: list[str]) -> Path | None:
  value = get_flag_value(argv, ("--motion-file",))
  return Path(value) if value is not None else None


def get_motion_pack(argv: list[str]) -> Path | None:
  value = get_flag_value(argv, ("--motion-pack",))
  return Path(value) if value is not None else None


def get_motion_split(argv: list[str]) -> str | None:
  return get_flag_value(argv, ("--motion-split",))


def get_motion_pack_sampling_mode(argv: list[str]) -> str | None:
  return get_flag_value(argv, ("--motion-pack-sampling-mode",))


def get_registry_name(argv: list[str]) -> str | None:
  return get_flag_value(argv, ("--registry-name",))


def validate_motion_for_task(task_id: str, motion_file: Path) -> None:
  try:
    adapter = get_adapter_for_task_id(task_id)
  except ValueError as exc:
    raise ValueError(f"Unknown robot for task_id={task_id!r}; cannot validate motion") from exc

  adapter.validate_motion_npz(motion_file)


def prepare_motion_for_task(task_id: str, motion_file: Path) -> Path:
  """Prepare a motion file for consumption by mjlab (may rewrite to a cached file)."""
  adapter = get_adapter_for_task_id(task_id)
  if adapter.prepare_motion_npz is None:
    return motion_file
  return adapter.prepare_motion_npz(motion_file)

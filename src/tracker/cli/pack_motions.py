from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tyro
import yaml

from tracker.motions.manifest import Manifest, TaskSpec, resolve_clips
from tracker.motions.pack_writer import write_motion_pack
from tracker.motions.splits import SplitConfig, assign_splits


@dataclass(frozen=True)
class PackMotionsArgs:
  manifest: Path
  out: Path


def _as_float(value: Any, *, field_name: str) -> float:
  try:
    return float(value)
  except Exception as exc:
    raise ValueError(f"Invalid {field_name}: expected float, got {value!r}") from exc


def _as_int(value: Any, *, field_name: str) -> int:
  try:
    return int(value)
  except Exception as exc:
    raise ValueError(f"Invalid {field_name}: expected int, got {value!r}") from exc


def _load_manifest(path: Path) -> tuple[Manifest, SplitConfig]:
  raw = yaml.safe_load(path.read_text(encoding="utf-8"))
  if not isinstance(raw, dict):
    raise ValueError("Manifest must be a YAML mapping/object.")

  schema_version = _as_int(raw.get("schema_version"), field_name="schema_version")
  if schema_version != 1:
    raise ValueError(f"Unsupported manifest schema_version={schema_version}")

  root_raw = raw.get("root")
  if not isinstance(root_raw, str) or not root_raw:
    raise ValueError("Manifest field `root` must be a non-empty string.")

  root = Path(root_raw).expanduser()
  if not root.is_absolute():
    root = (path.parent / root).resolve()

  tasks_raw = raw.get("tasks")
  if not isinstance(tasks_raw, list) or not tasks_raw:
    raise ValueError("Manifest field `tasks` must be a non-empty list.")

  tasks: list[TaskSpec] = []
  for idx, task in enumerate(tasks_raw):
    if not isinstance(task, dict):
      raise ValueError(f"Task[{idx}] must be a mapping/object.")
    name = task.get("name")
    if not isinstance(name, str) or not name:
      raise ValueError(f"Task[{idx}].name must be a non-empty string.")
    include = task.get("include")
    if not isinstance(include, list) or not include or not all(isinstance(p, str) for p in include):
      raise ValueError(f"Task[{idx}].include must be a non-empty list[str].")
    exclude = task.get("exclude", [])
    if not isinstance(exclude, list) or not all(isinstance(p, str) for p in exclude):
      raise ValueError(f"Task[{idx}].exclude must be a list[str].")
    weight = _as_float(task.get("weight", 1.0), field_name=f"Task[{idx}].weight")
    tasks.append(TaskSpec(name=name, include=list(include), exclude=list(exclude), weight=weight))

  splits_raw = raw.get("splits", {}) or {}
  if not isinstance(splits_raw, dict):
    raise ValueError("Manifest field `splits` must be a mapping/object.")

  split_cfg = SplitConfig(
    seed=_as_int(splits_raw.get("seed", 0), field_name="splits.seed"),
    val_ratio=_as_float(splits_raw.get("val_ratio", 0.02), field_name="splits.val_ratio"),
    test_ratio=_as_float(splits_raw.get("test_ratio", 0.01), field_name="splits.test_ratio"),
    min_val_per_task=_as_int(
      splits_raw.get("min_val_per_task", 1), field_name="splits.min_val_per_task"
    ),
    min_test_per_task=_as_int(
      splits_raw.get("min_test_per_task", 1), field_name="splits.min_test_per_task"
    ),
  )

  return Manifest(schema_version=schema_version, root=root, tasks=tasks), split_cfg


def main() -> None:
  args = tyro.cli(PackMotionsArgs)
  manifest, split_cfg = _load_manifest(args.manifest)

  clips = resolve_clips(manifest)
  splits = assign_splits(clips, split_cfg)

  print(f"[INFO] Resolved {len(clips)} clips from: {args.manifest}")
  print(f"[INFO] Split sizes: train={len(splits.train)} val={len(splits.val)} test={len(splits.test)}")

  write_motion_pack(manifest, split_cfg, args.out, source_manifest_path=args.manifest.resolve())
  print(f"[INFO] Wrote motion pack: {args.out}")


if __name__ == "__main__":
  main()


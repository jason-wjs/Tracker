from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class TaskSpec:
  name: str
  include: list[str]
  exclude: list[str] = field(default_factory=list)
  weight: float = 1.0


@dataclass(frozen=True)
class Manifest:
  schema_version: int
  root: Path
  tasks: list[TaskSpec]


@dataclass(frozen=True)
class ResolvedClip:
  task_id: int
  task_name: str
  abs_path: Path
  relpath: str


def resolve_clips(manifest: Manifest) -> list[ResolvedClip]:
  if manifest.schema_version != 1:
    raise ValueError(f"Unsupported manifest schema_version={manifest.schema_version}")

  root = manifest.root
  task_specs = sorted(manifest.tasks, key=lambda t: t.name)
  task_id_by_name = {task.name: idx for idx, task in enumerate(task_specs)}

  resolved: list[ResolvedClip] = []
  for task in task_specs:
    included: dict[str, Path] = {}
    for pattern in task.include:
      for path in root.glob(pattern):
        if not path.is_file() or path.suffix != ".npz":
          continue
        relpath = path.relative_to(root).as_posix()
        included[relpath] = path

    excluded: set[str] = set()
    for pattern in task.exclude:
      for path in root.glob(pattern):
        if not path.is_file() or path.suffix != ".npz":
          continue
        excluded.add(path.relative_to(root).as_posix())

    task_id = task_id_by_name[task.name]
    for relpath in sorted(included.keys()):
      if relpath in excluded:
        continue
      resolved.append(
        ResolvedClip(
          task_id=task_id,
          task_name=task.name,
          abs_path=included[relpath],
          relpath=relpath,
        )
      )

  return sorted(resolved, key=lambda c: c.relpath)


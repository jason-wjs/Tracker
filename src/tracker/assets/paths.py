"""Asset path helpers for packaged data."""

from __future__ import annotations

import os
from importlib import resources
from pathlib import Path


def _default_asset_cache_dir() -> Path:
  if env := os.environ.get("TRACKER_ASSET_CACHE_DIR"):
    return Path(env)
  return Path.cwd() / ".tracker-cache" / "assets"


def _extract_tree(src, dst: Path) -> None:
  dst.mkdir(parents=True, exist_ok=True)
  for child in src.iterdir():
    out_path = dst / child.name
    if child.is_dir():
      _extract_tree(child, out_path)
      continue
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(child.read_bytes())


def get_asset_path(robot_id: str, relative_path: str) -> Path:
  """Resolve a packaged asset path for a robot."""
  rel = relative_path.lstrip("/\\")
  robot_root = resources.files("tracker.assets") / robot_id

  # Development mode: package lives on disk, we can return a direct filesystem path.
  try:
    robot_root_fs = Path(robot_root)
  except TypeError:
    robot_root_fs = None
  else:
    if robot_root_fs.exists():
      path = robot_root_fs / rel
      if not path.exists():
        raise FileNotFoundError(f"Asset not found: {robot_id}/{rel}")
      return path

  # Wheel/zip imports: extract the robot asset directory to a local cache so MuJoCo can
  # load sibling mesh files referenced by the MJCF.
  cache_dir = _default_asset_cache_dir() / robot_id
  path = cache_dir / rel
  if not path.exists():
    _extract_tree(robot_root, cache_dir)

  if not path.exists():
    raise FileNotFoundError(f"Asset not found: {robot_id}/{rel}")
  return path

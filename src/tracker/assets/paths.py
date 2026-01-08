"""Asset path helpers for packaged data."""

from importlib import resources
from pathlib import Path


def get_asset_path(robot_id: str, relative_path: str) -> Path:
  """Resolve a packaged asset path for a robot."""
  rel = relative_path.lstrip("/\\")
  base = resources.files("tracker.assets") / robot_id
  path = Path(base / rel)
  if not path.exists():
    raise FileNotFoundError(f"Asset not found: {robot_id}/{rel}")
  return path

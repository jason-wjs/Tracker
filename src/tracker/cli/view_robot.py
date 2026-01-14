from __future__ import annotations

from pathlib import Path
from typing import Literal

import mujoco
import tyro

from tracker.assets.paths import get_asset_path

ViewerBackend = Literal["none", "native"]


def _resolve_robot_xml(robot_id: str, variant: int) -> Path:
  if robot_id == "adam_sp" and variant == 23:
    return get_asset_path("adam_sp", "adam_sp.xml")
  if robot_id == "adam_sp" and variant == 29:
    return get_asset_path("adam_sp", "adam_sp_29dof.xml")
  if robot_id == "adam_pro" and variant == 23:
    return get_asset_path("adam_pro", "adam_pro.xml")
  if robot_id == "adam_pro" and variant == 29:
    return get_asset_path("adam_pro", "adam_pro_29dof.xml")

  raise ValueError(f"Unsupported robot_id/variant: {robot_id!r}/{variant!r}")


def view_robot(
  *,
  robot_id: str,
  variant: int = 23,
  viewer: ViewerBackend = "native",
) -> None:
  """Load a robot MJCF and optionally open an interactive MuJoCo viewer."""
  xml_path = _resolve_robot_xml(robot_id, variant)
  model = mujoco.MjModel.from_xml_path(str(xml_path))
  data = mujoco.MjData(model)

  print(f"[INFO] Loaded MJCF: {xml_path}")
  print(
    "[INFO] Model summary: "
    f"nbody={model.nbody}, njnt={model.njnt}, ngeom={model.ngeom}, "
    f"nactuator={model.nu}, nq={model.nq}, nv={model.nv}"
  )

  if viewer == "none":
    return

  if viewer != "native":
    raise ValueError(f"Unsupported viewer backend: {viewer!r}")

  try:
    from mujoco import viewer as mujoco_viewer

    with mujoco_viewer.launch_passive(model, data) as v:
      while v.is_running():
        mujoco.mj_step(model, data)
        v.sync()
  except Exception as exc:
    raise RuntimeError(
      "Failed to launch MuJoCo viewer. Try `--viewer none` for headless inspection."
    ) from exc


def main() -> None:
  tyro.cli(view_robot)


if __name__ == "__main__":
  main()

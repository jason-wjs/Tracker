from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

TASK_ID_ADAM_SP_23 = "Tracker-Tracking-Flat-Adam-SP-23"
TASK_ID_ADAM_SP_29 = "Tracker-Tracking-Flat-Adam-SP-29"
TASK_ID_ADAM_PRO_23 = "Tracker-Tracking-Flat-Adam-Pro-23"
TASK_ID_ADAM_PRO_29 = "Tracker-Tracking-Flat-Adam-Pro-29"
TASK_ID_ADAM_PRO_23_NO_STATE_ESTIMATION = "Tracker-Tracking-Flat-Adam-Pro-23-No-State-Estimation"
TASK_ID_ADAM_PRO_29_NO_STATE_ESTIMATION = "Tracker-Tracking-Flat-Adam-Pro-29-No-State-Estimation"


ValidateMotionFn = Callable[[Path], None]
PrepareMotionFn = Callable[[Path], Path]


@dataclass(frozen=True)
class RobotAdapter:
  task_id: str
  robot_id: str
  variant: str
  validate_motion_npz: ValidateMotionFn
  prepare_motion_npz: PrepareMotionFn | None = None


def _build_adam_sp_23() -> RobotAdapter:
  from tracker.robots.adam_sp import validate_motion_npz

  return RobotAdapter(
    task_id=TASK_ID_ADAM_SP_23,
    robot_id="adam_sp",
    variant="23",
    validate_motion_npz=validate_motion_npz,
  )


def _build_adam_sp_29() -> RobotAdapter:
  from tracker.robots.adam_sp_29 import prepare_motion_npz, validate_motion_npz

  return RobotAdapter(
    task_id=TASK_ID_ADAM_SP_29,
    robot_id="adam_sp",
    variant="29",
    validate_motion_npz=validate_motion_npz,
    prepare_motion_npz=prepare_motion_npz,
  )


def _adam_pro_not_implemented(_: Path) -> None:
  raise NotImplementedError("adam_pro is not implemented yet")


def _build_adam_pro_23() -> RobotAdapter:
  from tracker.robots.adam_pro import validate_motion_npz

  return RobotAdapter(
    task_id=TASK_ID_ADAM_PRO_23,
    robot_id="adam_pro",
    variant="23",
    validate_motion_npz=validate_motion_npz,
  )


def _build_adam_pro_29() -> RobotAdapter:
  from tracker.robots.adam_pro_29 import prepare_motion_npz, validate_motion_npz

  return RobotAdapter(
    task_id=TASK_ID_ADAM_PRO_29,
    robot_id="adam_pro",
    variant="29",
    validate_motion_npz=validate_motion_npz,
    prepare_motion_npz=prepare_motion_npz,
  )


_ADAPTER_BUILDERS: dict[str, Callable[[], RobotAdapter]] = {
  TASK_ID_ADAM_SP_23: _build_adam_sp_23,
  TASK_ID_ADAM_SP_29: _build_adam_sp_29,
  TASK_ID_ADAM_PRO_23: _build_adam_pro_23,
  TASK_ID_ADAM_PRO_29: _build_adam_pro_29,
  TASK_ID_ADAM_PRO_23_NO_STATE_ESTIMATION: _build_adam_pro_23,
  TASK_ID_ADAM_PRO_29_NO_STATE_ESTIMATION: _build_adam_pro_29,
}


def list_known_task_ids() -> list[str]:
  return sorted(_ADAPTER_BUILDERS.keys())


def get_adapter_for_task_id(task_id: str) -> RobotAdapter:
  builder = _ADAPTER_BUILDERS.get(task_id)
  if builder is None:
    raise ValueError(f"Unknown robot for task_id={task_id!r}")
  return builder()

def _list_joint_names(xml_path: str) -> list[str]:
  import mujoco

  model = mujoco.MjModel.from_xml_path(xml_path)
  names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)]
  return [n for n in names if n]


def test_adam_sp_29_has_expected_wrist_joints():
  from tracker.assets.paths import get_asset_path

  xml = str(get_asset_path("adam_sp", "adam_sp_29dof.xml"))
  joint_names = _list_joint_names(xml)
  wrists = [n for n in joint_names if n.startswith(("wristYaw_", "wristPitch_", "wristRoll_"))]
  assert set(wrists) == {
    "wristYaw_Left",
    "wristPitch_Left",
    "wristRoll_Left",
    "wristYaw_Right",
    "wristPitch_Right",
    "wristRoll_Right",
  }


def test_adam_pro_29_has_expected_wrist_joints():
  from tracker.assets.paths import get_asset_path

  xml = str(get_asset_path("adam_pro", "adam_pro_29dof.xml"))
  joint_names = _list_joint_names(xml)
  wrists = [n for n in joint_names if n.startswith(("wristYaw_", "wristPitch_", "wristRoll_"))]
  assert set(wrists) == {
    "wristYaw_Left",
    "wristPitch_Left",
    "wristRoll_Left",
    "wristYaw_Right",
    "wristPitch_Right",
    "wristRoll_Right",
  }


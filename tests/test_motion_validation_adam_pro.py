import numpy as np
import pytest


def _load_model(xml_name: str):
  import mujoco

  from tracker.assets.paths import get_asset_path

  xml_path = get_asset_path("adam_pro", xml_name)
  return mujoco.MjModel.from_xml_path(str(xml_path))


def _nonfree_joint_names(model) -> list[str]:
  import mujoco

  joint_names = [
    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)
  ]
  return [n for n in joint_names if n and n != "floating_base"]


def _body_names(model) -> list[str]:
  import mujoco

  return [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(model.nbody)]


def test_adam_pro_23_validate_motion_rejects_missing_keys(tmp_path):
  from tracker.robots.adam_pro import validate_motion_npz

  path = tmp_path / "bad.npz"
  np.savez(path, joint_pos=np.zeros((1, 1), dtype=np.float32))
  with pytest.raises(ValueError, match="missing keys"):
    validate_motion_npz(path)


def test_adam_pro_23_validate_motion_accepts_tracking_subset_bodies(tmp_path):
  from tracker.robots.adam_pro import TRACKING_BODY_NAMES, validate_motion_npz

  model = _load_model("adam_pro.xml")
  joint_names = _nonfree_joint_names(model)

  t = 2
  path = tmp_path / "ok_subset.npz"
  np.savez(
    path,
    joint_pos=np.zeros((t, len(joint_names)), dtype=np.float32),
    joint_vel=np.zeros((t, len(joint_names)), dtype=np.float32),
    body_pos_w=np.zeros((t, len(TRACKING_BODY_NAMES), 3), dtype=np.float32),
    body_quat_w=np.zeros((t, len(TRACKING_BODY_NAMES), 4), dtype=np.float32),
    body_lin_vel_w=np.zeros((t, len(TRACKING_BODY_NAMES), 3), dtype=np.float32),
    body_ang_vel_w=np.zeros((t, len(TRACKING_BODY_NAMES), 3), dtype=np.float32),
  )
  validate_motion_npz(path)


def test_adam_pro_23_validate_motion_enforces_joint_names_if_present(tmp_path):
  from tracker.robots.adam_pro import TRACKING_BODY_NAMES, validate_motion_npz

  model = _load_model("adam_pro.xml")
  joint_names = _nonfree_joint_names(model)

  bad_joint_names = list(joint_names)
  bad_joint_names[0] = "not_a_real_joint"

  t = 1
  path = tmp_path / "bad_joint_names.npz"
  np.savez(
    path,
    joint_pos=np.zeros((t, len(joint_names)), dtype=np.float32),
    joint_vel=np.zeros((t, len(joint_names)), dtype=np.float32),
    body_pos_w=np.zeros((t, len(TRACKING_BODY_NAMES), 3), dtype=np.float32),
    body_quat_w=np.zeros((t, len(TRACKING_BODY_NAMES), 4), dtype=np.float32),
    body_lin_vel_w=np.zeros((t, len(TRACKING_BODY_NAMES), 3), dtype=np.float32),
    body_ang_vel_w=np.zeros((t, len(TRACKING_BODY_NAMES), 3), dtype=np.float32),
    joint_names=np.asarray(bad_joint_names),
  )
  with pytest.raises(ValueError, match="joint_names mismatch"):
    validate_motion_npz(path)


def test_adam_pro_23_validate_motion_enforces_body_names_if_present(tmp_path):
  from tracker.robots.adam_pro import TRACKING_BODY_NAMES, validate_motion_npz

  model = _load_model("adam_pro.xml")
  joint_names = _nonfree_joint_names(model)

  bad_body_names = list(TRACKING_BODY_NAMES)
  bad_body_names[0] = "not_a_real_body"

  t = 1
  path = tmp_path / "bad_body_names.npz"
  np.savez(
    path,
    joint_pos=np.zeros((t, len(joint_names)), dtype=np.float32),
    joint_vel=np.zeros((t, len(joint_names)), dtype=np.float32),
    body_pos_w=np.zeros((t, len(TRACKING_BODY_NAMES), 3), dtype=np.float32),
    body_quat_w=np.zeros((t, len(TRACKING_BODY_NAMES), 4), dtype=np.float32),
    body_lin_vel_w=np.zeros((t, len(TRACKING_BODY_NAMES), 3), dtype=np.float32),
    body_ang_vel_w=np.zeros((t, len(TRACKING_BODY_NAMES), 3), dtype=np.float32),
    body_names=np.asarray(bad_body_names),
  )
  with pytest.raises(ValueError, match="body_names mismatch"):
    validate_motion_npz(path)


def test_adam_pro_29_prepare_motion_npz_slices_qpos_qvel(tmp_path):
  from tracker.robots.adam_pro_29 import prepare_motion_npz

  model = _load_model("adam_pro_29dof.xml")
  joint_names = _nonfree_joint_names(model)
  body_names = _body_names(model)

  t = 2
  motion_path = tmp_path / "motion_full_qpos_qvel.npz"
  np.savez(
    motion_path,
    fps=np.array([50], dtype=np.int64),
    joint_pos=np.zeros((t, model.nq), dtype=np.float32),
    joint_vel=np.zeros((t, model.nv), dtype=np.float32),
    body_pos_w=np.zeros((t, model.nbody, 3), dtype=np.float32),
    body_quat_w=np.zeros((t, model.nbody, 4), dtype=np.float32),
    body_lin_vel_w=np.zeros((t, model.nbody, 3), dtype=np.float32),
    body_ang_vel_w=np.zeros((t, model.nbody, 3), dtype=np.float32),
    joint_names=np.asarray(joint_names),
    body_names=np.asarray(body_names),
  )

  prepared = prepare_motion_npz(motion_path)
  assert prepared.exists()
  with np.load(prepared, allow_pickle=False) as z:
    assert z["joint_pos"].shape == (t, len(joint_names))
    assert z["joint_vel"].shape == (t, len(joint_names))
    assert z["body_pos_w"].shape == (t, model.nbody, 3)


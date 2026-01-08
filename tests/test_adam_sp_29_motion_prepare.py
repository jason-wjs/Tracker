import numpy as np


def test_prepare_motion_npz_slices_qpos_qvel(tmp_path):
  import mujoco

  from tracker.assets.paths import get_asset_path
  from tracker.robots.adam_sp_29 import prepare_motion_npz

  xml_path = get_asset_path("adam_sp", "adam_sp_29dof.xml")
  model = mujoco.MjModel.from_xml_path(str(xml_path))

  body_names = [
    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(model.nbody)
  ]
  joint_names = [
    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
    for i in range(model.njnt)
    if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) != "floating_base"
  ]

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


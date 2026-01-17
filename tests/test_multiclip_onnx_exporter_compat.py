from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def _write_clip(path: Path, *, t: int, j: int, b: int) -> None:
  np.savez(
    path,
    joint_pos=np.zeros((t, j), dtype=np.float32),
    joint_vel=np.zeros((t, j), dtype=np.float32),
    body_pos_w=np.zeros((t, b, 3), dtype=np.float32),
    body_quat_w=np.zeros((t, b, 4), dtype=np.float32),
    body_lin_vel_w=np.zeros((t, b, 3), dtype=np.float32),
    body_ang_vel_w=np.zeros((t, b, 3), dtype=np.float32),
  )


def test_multiclip_command_works_with_mjlab_onnx_exporter(tmp_path: Path):
  from mjlab.tasks.tracking.mdp.commands import MotionCommand as MjlabMotionCommand
  from mjlab.tasks.tracking.rl.exporter import _OnnxMotionPolicyExporter

  from tracker.motions.manifest import Manifest, TaskSpec
  from tracker.motions.pack_loader import MotionPack
  from tracker.motions.pack_writer import write_motion_pack
  from tracker.motions.splits import SplitConfig
  from tracker.tasks.tracking.multiclip_command import MultiClipMotionCommand

  assert issubclass(MultiClipMotionCommand, MjlabMotionCommand)

  root = tmp_path / "root"
  (root / "all").mkdir(parents=True)
  _write_clip(root / "all" / "0.npz", t=4, j=1, b=1)

  pack_dir = tmp_path / "pack"
  write_motion_pack(
    Manifest(schema_version=1, root=root, tasks=[TaskSpec(name="all", include=["all/*.npz"])]),
    SplitConfig(seed=0, val_ratio=0.0, test_ratio=0.0),
    pack_dir,
  )

  pack = MotionPack.load(pack_dir, device="cpu", cast_body_to_f32=True)

  cmd = object.__new__(MultiClipMotionCommand)
  cmd._set_export_motion(pack=pack, body_indexes=torch.tensor([0], dtype=torch.long), clip_id=0)

  class _CmdMgr:
    def get_term(self, name: str):
      assert name == "motion"
      return cmd

  class _Env:
    command_manager = _CmdMgr()

  class _Policy:
    is_recurrent = False
    actor = torch.nn.Sequential(torch.nn.Linear(1, 1))

  _OnnxMotionPolicyExporter(_Env(), _Policy(), normalizer=None, verbose=False)


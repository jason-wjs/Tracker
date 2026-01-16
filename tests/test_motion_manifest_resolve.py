from pathlib import Path


def test_motion_manifest_resolve_globs_sorted_and_stable_ids(tmp_path: Path):
  from tracker.motions.manifest import Manifest, TaskSpec, resolve_clips

  root = tmp_path / "root"
  (root / "walk").mkdir(parents=True)
  (root / "run").mkdir(parents=True)
  (root / "bad").mkdir(parents=True)

  (root / "walk" / "a.npz").write_bytes(b"")
  (root / "walk" / "b.npz").write_bytes(b"")
  (root / "run" / "c.npz").write_bytes(b"")
  (root / "bad" / "tmp_bad.npz").write_bytes(b"")

  manifest = Manifest(
    schema_version=1,
    root=root,
    tasks=[
      TaskSpec(name="walk", include=["walk/*.npz"], exclude=["walk/b.npz"]),
      TaskSpec(name="run", include=["run/*.npz"]),
    ],
  )

  resolved = resolve_clips(manifest)

  assert [c.relpath for c in resolved] == ["run/c.npz", "walk/a.npz"]
  assert [c.task_name for c in resolved] == ["run", "walk"]
  assert [c.task_id for c in resolved] == [0, 1]


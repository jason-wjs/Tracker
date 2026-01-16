# Motion Pack Directory Format (schema_version: 1)

A *motion pack* is a deterministic directory produced by `tracker-pack-motions`. It concatenates many per-clip `.npz` motions into a small number of `.npy` arrays (mmap-friendly) plus metadata and split indices.

## Directory layout

```text
pack_dir/
  meta.json
  index.jsonl
  arrays/
    joint_pos.npy
    joint_vel.npy
    body_pos_w.npy
    body_quat_w.npy
    body_lin_vel_w.npy
    body_ang_vel_w.npy
    clip_start.npy
    clip_len.npy
    clip_task_id.npy
  splits/
    train_clip_ids.npy
    val_clip_ids.npy
    test_clip_ids.npy
```

## Files

### `meta.json`
Required JSON file containing:
- `schema_version`: `1`
- `created_at`: ISO-8601 string
- `source_manifest`: absolute path to the manifest used
- `root`: dataset root path
- `tasks`: list of `{ "name": ..., "task_id": ..., "weight": ... }`
- `counts`: global + per-task counts (clips, frames)
- `arrays`: shapes and dtypes for each `.npy`

### `index.jsonl`
One JSON object per clip, in `clip_id` order, containing:
- `clip_id`: `int`
- `task_id`: `int`
- `task_name`: `str`
- `relpath`: `str` (root-relative path to original `.npz`)
- `clip_start`: `int` (start frame index into concatenated arrays)
- `clip_len`: `int` (number of frames)

### `arrays/*.npy`
Concatenated per-frame arrays. `clip_start/clip_len` segment frames back into clips.

**Dtypes (locked):**
- `joint_*`: `float32`
- `body_*`: `float16` (cast to `float32` on GPU when used)

### `splits/*.npy`
Per-split lists of `clip_id` (dtype `int64`), deterministically computed per-task with minimums.

## Versioning

- `schema_version: 1` is required in `meta.json`.
- Future versions must be backward-incompatible only with a schema version bump.


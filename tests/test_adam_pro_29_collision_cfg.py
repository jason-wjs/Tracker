def test_adam_pro_29_collisions_only_enable_feet_and_torso():
  from tracker.robots.adam_pro_29_constants import ADAM_PRO_29_COLLISIONS

  cfg = ADAM_PRO_29_COLLISIONS[0]
  enabled = (
    r"^(left_foot[0-9]+_collision|right_foot[0-9]+_collision|pelvis_collision|torso_collision)$"
  )
  assert cfg.contype[enabled] == 1
  assert cfg.contype[r".*_collision$"] == 0
  assert cfg.conaffinity[enabled] == 1
  assert cfg.conaffinity[r".*_collision$"] == 0


from tracker.assets.paths import get_asset_path


def test_get_asset_path_adam_sp_xml_exists():
  path = get_asset_path("adam_sp", "adam_sp.xml")
  assert path.is_file()


def test_get_asset_path_adam_pro_xml_exists():
  path = get_asset_path("adam_pro", "adam_pro.xml")
  assert path.is_file()


def test_get_asset_path_adam_pro_29_xml_exists():
  path = get_asset_path("adam_pro", "adam_pro_29dof.xml")
  assert path.is_file()

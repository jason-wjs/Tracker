from tracker.assets.paths import get_asset_path


def test_get_asset_path_adam_sp_xml_exists():
  path = get_asset_path("adam_sp", "adam_sp.xml")
  assert path.is_file()


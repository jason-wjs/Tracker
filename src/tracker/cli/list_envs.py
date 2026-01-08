"""Wrapper around mjlab's list-envs with tracker task bootstrap."""

import tyro

from tracker.integrations.mjlab.bootstrap import bootstrap


def list_environments(keyword: str | None = None) -> int:
  bootstrap()
  from mjlab.scripts.list_envs import list_environments as mjlab_list_environments

  return mjlab_list_environments(keyword=keyword)


def main() -> int:
  return tyro.cli(list_environments)


if __name__ == "__main__":
  raise SystemExit(main())


def bootstrap() -> None:
  """Ensure mjlab and tracker tasks are registered in the current process."""

  import mjlab  # noqa: F401
  import tracker.tasks.register_all  # noqa: F401


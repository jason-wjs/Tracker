"""Import tracker tasks so they register with mjlab."""

from tracker.tasks.tracking import register as _tracking_register  # noqa: F401
from tracker.tasks.tracking import register_adam_sp_29 as _tracking_register_29  # noqa: F401

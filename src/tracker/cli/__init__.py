"""Command-line entrypoints for the tracker package.

`tracker.cli` contains importable console-script entrypoints (thin arg parsing + I/O)
and delegates to `mjlab.scripts.*` after calling
`tracker.integrations.mjlab.bootstrap.bootstrap()` to ensure tasks are registered.

We use `cli/` (not `scripts/`) to emphasize these are packaged entrypoints, not ad-hoc
utilities, and to avoid confusion with `mjlab.scripts`.
"""

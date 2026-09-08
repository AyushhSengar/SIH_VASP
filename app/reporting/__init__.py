"""Reporting layer: renderers that turn an `InvestigationReport` into output.

Nothing in `app/investigation/` prints, and nothing here analyses. That split
keeps the pipeline assertable in tests without capturing stdout and lets a
future API reuse the same report object with a different renderer.
"""

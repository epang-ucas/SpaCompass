"""Locate SpaCompass runtime resources (gene vocabularies, platform means).

The resources ship with the repository under ``resources/``.  Set the
``SPACOMPASS_RESOURCES`` environment variable to point at another directory
when the files live outside the checkout (for example a shared download
location).
"""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_RESOURCE_ROOT = Path(__file__).resolve().parent.parent / "resources"


def resource_root() -> Path:
    """Return the directory that holds the packaged resources."""
    override = os.environ.get("SPACOMPASS_RESOURCES")
    if override:
        return Path(override).expanduser().resolve()
    return DEFAULT_RESOURCE_ROOT


def resource_path(*parts: str) -> str:
    """Return the absolute path of a resource file, asserting that it exists."""
    path = resource_root().joinpath(*parts)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing SpaCompass resource: {path}. "
            "Download the released resources or set SPACOMPASS_RESOURCES."
        )
    return str(path)

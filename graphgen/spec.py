"""Spec loader for the standard layout dataset."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DEFAULT_SPEC_PATH = Path("datasets/standard/spec.json")


def load_standard_spec(spec_path: str | Path = DEFAULT_SPEC_PATH) -> dict[str, Any]:
    """Load and return the standard layout spec as a dictionary."""
    path = Path(spec_path)
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)

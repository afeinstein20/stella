from __future__ import annotations

from importlib import resources as _resources
from typing import List, Optional

DEFAULT_MODEL_NAME = "ensemble_s0018_i0350_b0.73_savedmodel.keras"


def list_model_names() -> List[str]:
    """Return the list of packaged model filenames."""
    data = _resources.files("stella.data")
    return sorted([p.name for p in data.iterdir() if p.suffix == ".keras"])  # type: ignore[attr-defined]


def list_model_paths() -> List[str]:
    """Return absolute paths to all packaged models."""
    data = _resources.files("stella.data")
    return sorted([str(p) for p in data.iterdir() if p.suffix == ".keras"])  # type: ignore[attr-defined]


def get_model_path(name: Optional[str] = None) -> str:
    """Return the absolute path to a packaged model by name.

    If `name` is None, returns the default model path.
    """
    if name is None:
        name = DEFAULT_MODEL_NAME
    return str(_resources.files("stella.data") / name)


# Convenience precomputed list of model paths for notebooks and quickstarts
# Equivalent to calling list_model_paths().
models: List[str] = list_model_paths()

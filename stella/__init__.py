import os

PACKAGEDIR = os.path.abspath(os.path.dirname(__file__))

from .version import __version__

# Convenience backend utilities exposed at top-level.
from .backends import (
    check_backend,
    benchmark,
    swap_backend,
    require_backend,
)  # noqa: F401
from . import models  # noqa: F401

# Lazy exports for backward compatibility, avoiding heavy imports at module import time
_LAZY_EXPORTS = {
    "ConvNN": ("neural_network", "ConvNN"),
    "FitFlares": ("mark_flares", "FitFlares"),
    "MeasureProt": ("rotations", "MeasureProt"),
    "neural_network": ("neural_network", None),
    "mark_flares": ("mark_flares", None),
    "pipeline": ("pipeline", None),
    "rotations": ("rotations", None),
}


def __getattr__(name):
    tgt = _LAZY_EXPORTS.get(name)
    if tgt is None:
        raise AttributeError(f"module 'stella' has no attribute '{name}'")
    mod_name, attr = tgt
    mod = __import__(f"{__name__}.{mod_name}", fromlist=[attr] if attr else [])
    return getattr(mod, attr) if attr else mod


def __dir__():
    return sorted(list(globals().keys()) + list(_LAZY_EXPORTS.keys()))


__all__ = [
    "__version__",
    "check_backend",
    "benchmark",
    "swap_backend",
    "require_backend",
    "models",
    # Back-compat lazy exports
    "ConvNN",
    "FitFlares",
    "MeasureProt",
    "neural_network",
    "mark_flares",
    "pipeline",
    "rotations",
]

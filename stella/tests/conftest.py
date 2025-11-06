import os
import warnings
from pathlib import Path
import pytest
import sys


def pytest_addoption(parser):
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="Run integration tests (downloads/models). Off by default.",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "integration: marks tests as integration (requires network/models)"
    )
    config.addinivalue_line(
        "markers",
        "downloads: marks tests that may download small data; allowed by default",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-integration"):
        return
    skip_integration = pytest.mark.skip(
        reason="Integration tests are disabled. Use --run-integration to enable."
    )
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_integration)


def pytest_sessionstart(session):
    # Ensure Lightkurve uses the new cache path to avoid migration warnings
    os.environ.setdefault(
        "LIGHTKURVE_CACHE_DIR", str(Path.home() / ".lightkurve" / "cache")
    )
    # Silence optional dependency warnings from Lightkurve PRF module
    warnings.filterwarnings(
        "ignore", message=r".*tpfmodel submodule is not available.*"
    )
    warnings.filterwarnings("ignore", message=r".*Lightkurve cache directory.*")


@pytest.fixture(autouse=True)
def _debug_backend(request):
    """Print concise backend diagnostics for each test function."""
    be = os.environ.get("KERAS_BACKEND")
    imported = "keras" in sys.modules
    current = None
    if imported:
        try:
            import keras  # type: ignore

            current = keras.backend.backend()
        except Exception:
            current = "<error>"
    print(
        f"[test {request.node.nodeid}] KERAS_BACKEND={be} keras_imported={imported} current={current}"
    )

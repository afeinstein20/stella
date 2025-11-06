import json
import os
import sys
import subprocess
import pytest


def _run_py(code: str, env: dict) -> tuple[int, str, str]:
    p = subprocess.Popen(
        [sys.executable, "-c", code],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    out, err = p.communicate()
    return p.returncode, out.strip(), err.strip()


def test_swap_backend_basic():
    # Discover available backends
    rc, out, err = _run_py(
        "import json, stella; print(json.dumps(stella.check_backend(print_summary=False)))",
        {k: v for k, v in os.environ.items() if k != "KERAS_BACKEND"},
    )
    assert rc == 0, err
    info = json.loads(out)
    cands = info.get("candidates", [])
    if not cands:
        pytest.skip("No Keras backends installed")

    for be in cands:
        code = (
            "import os; from stella.backends import swap_backend; "
            f"swap_backend('{be}'); import keras; print(keras.backend.backend())"
        )
        rc, out, err = _run_py(
            code, {k: v for k, v in os.environ.items() if k != "KERAS_BACKEND"}
        )
        assert rc == 0, err
        assert out == be


def test_swap_backend_accelerator_mps_if_available():
    # Only relevant if Torch with MPS is available
    rc, out, err = _run_py(
        "import json, stella; print(json.dumps(stella.check_backend(print_summary=False)))",
        os.environ,
    )
    assert rc == 0, err
    info = json.loads(out)
    torch = info.get("torch", {})
    details = torch.get("details", {})
    if not (torch.get("installed") and details.get("mps")):
        pytest.skip("Torch MPS not available on this runner")

    code = (
        "from stella.backends import swap_backend; "
        "swap_backend('torch', accelerator='mps'); "
        "import keras, numpy as np; "
        "m = keras.Sequential([keras.layers.Input((8,)), keras.layers.Dense(4, activation='relu')]); "
        "y = m(np.zeros((2,8), dtype='float32')); print(keras.backend.backend(), y.shape)"
    )
    rc, out, err = _run_py(
        code, {k: v for k, v in os.environ.items() if k != "KERAS_BACKEND"}
    )
    assert rc == 0, err
    assert out.startswith("torch ") or out.startswith("torch,")

import json
import os
import sys
import time
import warnings
from typing import Dict, List, Optional


def _keras_current_backend() -> Optional[str]:
    try:
        import keras

        return keras.backend.backend()
    except Exception:
        return os.environ.get("KERAS_BACKEND")


def _jax_info() -> Dict:
    info = {"name": "jax", "installed": False, "devices": [], "details": {}}
    try:
        import jax  # type: ignore

        info["installed"] = True
        devs = jax.devices()
        info["devices"] = [f"{d.platform}:{d.id}" for d in devs]
        kinds = sorted({d.platform for d in devs})
        info["details"]["kinds"] = kinds
    except Exception as e:
        info["error"] = str(e)
    return info


def _torch_info() -> Dict:
    info = {"name": "torch", "installed": False, "devices": [], "details": {}}
    try:
        import torch  # type: ignore

        info["installed"] = True
        cuda = torch.cuda.is_available()
        mps = getattr(getattr(torch, "backends", None), "mps", None)
        mps_avail = bool(mps.is_available()) if mps is not None else False
        info["details"]["cuda"] = bool(cuda)
        info["details"]["mps"] = bool(mps_avail)
        if cuda:
            count = torch.cuda.device_count()
            for i in range(count):
                info["devices"].append(f"cuda:{i}:{torch.cuda.get_device_name(i)}")
        if mps_avail:
            info["devices"].append("mps:0:Apple Metal")
    except Exception as e:
        info["error"] = str(e)
    return info


def check_backend(print_summary: bool = True) -> Dict:
    """
    Report available Keras backends and device acceleration.

    Returns a dict with keys: current, candidates, and per-backend info.
    """
    current = _keras_current_backend()
    jax_info = _jax_info()
    torch_info = _torch_info()
    candidates: List[str] = []
    if jax_info["installed"]:
        candidates.append("jax")
    if torch_info["installed"]:
        candidates.append("torch")

    result = {
        "current": current,
        "candidates": candidates,
        "jax": jax_info,
        "torch": torch_info,
        "env": {"KERAS_BACKEND": os.environ.get("KERAS_BACKEND")},
    }

    if print_summary:
        lines = [
            f"Keras backend (current): {current}",
            f"KERAS_BACKEND env: {os.environ.get('KERAS_BACKEND')}",
            f"Available backends: {', '.join(candidates) if candidates else 'none'}",
        ]
        # JAX summary
        if jax_info["installed"]:
            kinds = (
                ", ".join(jax_info.get("details", {}).get("kinds", []))
                or "(no devices)"
            )
            lines.append(
                f"- jax: installed; devices={jax_info['devices']} kinds={kinds}"
            )
        else:
            lines.append("- jax: not installed")
        # Torch summary
        if torch_info["installed"]:
            det = torch_info.get("details", {})
            acc = []
            if det.get("cuda"):
                acc.append("CUDA")
            if det.get("mps"):
                acc.append("MPS")
            acc_s = ", ".join(acc) if acc else "CPU"
            lines.append(
                f"- torch: installed; accel={acc_s}; devices={torch_info['devices']}"
            )
        else:
            lines.append("- torch: not installed")
        print("\n".join(lines))

    return result


def require_backend(backend: Optional[str] = None) -> None:
    """Validate that the selected Keras backend is installed and usable.

    Parameters
    ----------
    backend : str | None
        Backend to require ('jax' or 'torch'). If None, uses the active
        backend from KERAS_BACKEND or keras.backend.backend().

    Raises
    ------
    RuntimeError
        If the required backend is not installed or has no available devices.
    """
    info = check_backend(print_summary=False)
    be = (
        (backend or info.get("current") or os.environ.get("KERAS_BACKEND") or "")
        .strip()
        .lower()
    )
    # Auto-select a sensible default if none is set: prefer JAX, then Torch
    if be not in ("jax", "torch"):
        candidates = info.get("candidates", [])
        if "jax" in candidates:
            be = "jax"
        elif "torch" in candidates:
            be = "torch"
        else:
            raise RuntimeError(
                "No Keras backend installed. Install one with: "
                "pip install stella[jax]  (or)  pip install stella[torch]"
            )
        os.environ["KERAS_BACKEND"] = be

    ok = be in info.get("candidates", [])
    if not ok:
        hint = "pip install stella[jax]" if be == "jax" else "pip install stella[torch]"
        raise RuntimeError(
            f"Requested backend '{be}' is not installed. Install with: {hint}"
        )

    # Optional: ensure at least CPU device present (mostly for JAX visibility)
    details = info.get(be, {})
    # If nothing is reported, still allow CPU fallback; do not hard-fail here.
    return


def _subprocess_benchmark(
    model_path: str, target: str, sector: int, exptime: int, author: str
) -> Dict:
    """Run the actual timed inference in a fresh process for the active KERAS_BACKEND."""
    # Silence lightkurve noisy warnings
    warnings.filterwarnings(
        "ignore", message=r".*tpfmodel submodule is not available.*"
    )
    warnings.filterwarnings("ignore", message=r".*Lightkurve cache directory.*")

    import keras  # type: ignore
    import numpy as np  # type: ignore
    from lightkurve.search import search_lightcurvefile  # type: ignore
    from stella.neural_network import ConvNN  # type: ignore

    # Load model once
    m = keras.models.load_model(model_path, compile=False)
    cadences = int(m.input_shape[1])

    # Download + preprocess light curve
    lcf = search_lightcurvefile(target=target, mission="TESS", sector=sector)
    lc = lcf.download().normalize().remove_nans()
    try:
        lc = lc[lc.quality == 0]
    except Exception:
        pass

    t = lc.time.value
    f = lc.flux.value
    e = getattr(lc, "flux_err", None)
    if e is None:
        e = np.zeros_like(f) + np.nanmedian(f) * 1e-3
    else:
        e = e.value

    # Warmup predict on a tiny slice to trigger JIT/graph build
    cnn = ConvNN(output_dir=".")
    n_warm = min(5 * cadences, len(t))
    _t0 = time.perf_counter()
    cnn.predict(
        modelname=model_path, times=t[:n_warm], fluxes=f[:n_warm], errs=e[:n_warm]
    )

    # Timed full predict
    t0 = time.perf_counter()
    cnn.predict(modelname=model_path, times=t, fluxes=f, errs=e)
    dt = time.perf_counter() - t0

    return {
        "seconds": dt,
        "points": int(len(t)),
        "cadences": cadences,
        "pred_shape": tuple(np.array(cnn.predictions, dtype=object).shape),
    }


def benchmark(
    model_path: str,
    target: str = "tic62124646",
    sector: int = 13,
    exptime: int = 120,
    author: str = "SPOC",
    backends: Optional[List[str]] = None,
) -> Dict:
    """
    Benchmark inference speed across available backends on a standard light curve.

    Parameters
    ----------
    model_path : str
        Path to a `.keras` model file.
    target : str
        TIC identifier (default: 'tic62124646').
    sector : int
        TESS sector to download (default: 13).
    exptime : int
        Cadence in seconds (default: 120).
    author : str
        Lightkurve author (default: 'SPOC').
    backends : list[str] | None
        Specific backends to test; if None, use available ones.
    """
    if not model_path or not os.path.exists(os.path.expanduser(model_path)):
        raise FileNotFoundError(f"Model not found: {model_path}")
    model_path = os.path.expanduser(model_path)

    info = check_backend(print_summary=False)
    candidates = backends or info.get("candidates", [])
    if not candidates:
        raise RuntimeError("No Keras backends are installed to benchmark.")

    results: Dict[str, Dict] = {}
    for be in candidates:
        env = os.environ.copy()
        env["KERAS_BACKEND"] = be
        # Build inline script to run in a fresh interpreter
        code = (
            "import json, os; "
            "from stella.backends import _subprocess_benchmark; "
            f"res=_subprocess_benchmark({model_path!r}, {target!r}, {sector}, {exptime}, {author!r}); "
            "print(json.dumps(res))"
        )
        t0 = time.perf_counter()
        from subprocess import Popen, PIPE

        p = Popen(
            [sys.executable, "-c", code], env=env, stdout=PIPE, stderr=PIPE, text=True
        )
        out, err = p.communicate()
        elapsed = time.perf_counter() - t0
        if p.returncode != 0:
            results[be] = {"error": err.strip(), "elapsed": elapsed}
            continue
        try:
            # Attempt robust parse: use the last JSON-looking line
            line = out.strip().splitlines()[-1] if out.strip().splitlines() else ""
            if line.startswith("{") and line.endswith("}"):
                payload = json.loads(line)
            else:
                payload = json.loads(out.strip())
        except Exception as e:
            payload = {"parse_error": str(e), "raw": out}
        payload["elapsed_wall"] = elapsed
        results[be] = payload

    # Pretty print summary
    print("Benchmark results (lower is better):")
    for be, r in results.items():
        if "seconds" in r:
            print(
                f"- {be}: {r['seconds']:.3f}s predict (wall {r['elapsed_wall']:.3f}s), points={r.get('points')} cadences={r.get('cadences')}"
            )
        else:
            print(f"- {be}: ERROR {r.get('error') or r}")

    return {"backends": candidates, "results": results}


def _apply_accelerator_env(
    backend: str, accelerator: Optional[str], env: Dict[str, str]
) -> None:
    acc = (accelerator or "").lower().strip()
    if backend == "torch":
        if acc == "cpu":
            env["CUDA_VISIBLE_DEVICES"] = ""
            env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        elif acc in ("cuda", "gpu"):
            env.pop("CUDA_VISIBLE_DEVICES", None)  # allow default
            env["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
        elif acc == "mps":
            env["CUDA_VISIBLE_DEVICES"] = ""
            env["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
    elif backend == "jax":
        if acc == "cpu":
            env["JAX_PLATFORM_NAME"] = "cpu"
        elif acc in ("cuda", "gpu"):
            env["JAX_PLATFORM_NAME"] = "gpu"
        elif acc in ("metal", "mps"):
            # Experimental Apple Metal backend
            env["JAX_PLATFORM_NAME"] = "metal"


def swap_backend(
    backend: str, accelerator: Optional[str] = None, restart: bool = False
) -> Dict:
    """
    Prepare environment for a different Keras backend and accelerator.

    Note: Keras backend is selected at import time. If `keras` is already
    imported in this process, you must restart the interpreter for the change
    to take effect. Setting `restart=True` will perform an in-place re-exec.

    Parameters
    ----------
    backend : str
        One of 'jax' or 'torch'.
    accelerator : str | None
        Optional accelerator hint: 'cpu', 'cuda'/'gpu', or 'mps' (Apple Metal).
    restart : bool
        If True and keras is already imported, re-exec the current process.
    """
    be = backend.strip().lower()
    if be not in ("jax", "torch"):
        raise ValueError("backend must be 'jax' or 'torch'")

    env = os.environ.copy()
    env["KERAS_BACKEND"] = be
    _apply_accelerator_env(be, accelerator, env)

    already = "keras" in sys.modules
    summary = {
        "requested_backend": be,
        "accelerator": accelerator,
        "already_imported": already,
        "env_preview": {
            k: env.get(k)
            for k in (
                "KERAS_BACKEND",
                "JAX_PLATFORM_NAME",
                "CUDA_VISIBLE_DEVICES",
                "PYTORCH_ENABLE_MPS_FALLBACK",
            )
        },
        "action": None,
    }

    if already and not restart:
        warnings.warn(
            "Keras is already imported; backend cannot be swapped without restart. Call swap_backend(..., restart=True) to re-exec."
        )
        # Apply env to current process for any child processes
        os.environ.update(env)
        summary["action"] = "env_set_no_restart"
        return summary

    if already and restart:
        # Re-exec the interpreter with new env
        summary["action"] = "reexec"
        os.execvpe(sys.executable, [sys.executable] + sys.argv, env)

    # Not imported yet: set env and return
    os.environ.update(env)
    summary["action"] = "env_set"
    return summary

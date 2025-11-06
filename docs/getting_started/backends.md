# Backends (JAX and PyTorch)

stella uses Keras 3, which can run on multiple numerical backends. You can use either the JAX or the PyTorch backend with the same stella code.

Quick facts
-----------
- Select backend via the environment variable `KERAS_BACKEND` before importing `keras`.
- The stella code is backend‑agnostic; no changes are required besides selecting the backend and installing the backend packages.

Setup: JAX
-----------

```bash
pip install -U keras jax jaxlib
export KERAS_BACKEND=jax
```

Setup: PyTorch
--------------

```bash
pip install -U keras torch
export KERAS_BACKEND=torch
```

Usage is identical
------------------

```python
import os
os.environ.setdefault("KERAS_BACKEND", "jax")  # or "torch"
import keras
m = keras.models.load_model("/path/to/model.keras", compile=False)
y = m.predict(x)
```

Troubleshooting
---------------
- If you see a backend mismatch, ensure `KERAS_BACKEND` is set before any import of `keras` occurs in your Python process.
- Some ops or layers may have different performance characteristics across backends.

Inspect, swap, and benchmark
----------------------------

```python
import stella
stella.check_backend()
stella.swap_backend('torch', accelerator='mps')  # prepare env for PyTorch Metal
```

.. _backends:

Backends (JAX and PyTorch)
==========================

stella uses Keras Core, which can run on multiple numerical backends. You can use either the JAX or the PyTorch backend with the same stella code.

Quick facts
-----------
- Select backend via the environment variable ``KERAS_BACKEND`` before importing ``keras``.
- Keras JIT-compiles model functions under the hood (via XLA for JAX, TorchScript-like tracing for PyTorch where applicable). You do not need to wrap calls in ``jax.jit`` or Torch decorators.
- The stella code is backend-agnostic; no changes are required besides selecting the backend and installing the backend packages.

Setup: JAX
----------

.. code-block:: bash

   pip install -U keras jax jaxlib
   export KERAS_BACKEND=jax

Notes (Apple Silicon): JAX runs on CPU out of the box. If you need GPU/Metal acceleration, consult the JAX docs for the current macOS acceleration story.

Setup: PyTorch
--------------

.. code-block:: bash

   pip install -U keras torch
   export KERAS_BACKEND=torch

Notes (Apple Silicon): PyTorch supports the ``mps`` device via Metal on Apple Silicon; Keras-Core over Torch can leverage this where supported by the operators used in your model.

Usage is identical

Once the backend is selected, load models and run predictions the same way:

.. code-block:: python

   import os
   os.environ.setdefault("KERAS_BACKEND", "jax")  # or "torch"
   import keras
   m = keras.models.load_model("/path/to/model.keras", compile=False)
   y = m.predict(x)

Troubleshooting
---------------
- If you see a backend mismatch, ensure ``KERAS_BACKEND`` is set before any import of ``keras`` occurs in your Python process.
- Some ops or layers may have different performance characteristics across backends. If you encounter unsupported ops on a given backend, please open an issue.

Inspect, swap, and benchmark
----------------------------
- Inspect what's available and the current selection:

.. code-block:: python

   import stella
   stella.check_backend()

- Prepare to swap backends and accelerators (restart your Python session after calling):

.. code-block:: python

   import stella
   # Switch to PyTorch with Apple Metal (MPS)
   stella.swap_backend('torch', accelerator='mps')
   # Or switch to JAX on CPU
   stella.swap_backend('jax', accelerator='cpu')

- Benchmark prediction speed on TIC 62124646 using your model across available backends:

.. code-block:: python

   import stella
   res = stella.benchmark(model_path="/path/to/model.keras")
   print(res)

.. _installation:

Installation
============

To install stella with pip, pick a backend extras group:

.. code-block:: bash

	# JAX (CPU)
	pip install stella[jax]

	# JAX on macOS with Metal (Apple Silicon)
	pip install stella[jax-mps]

	# PyTorch
	pip install stella[torch]


Alternatively you can install the current development version of stella::

    git clone https://github.com/afeinstein20/stella
	cd stella
	python setup.py install

Backends
--------
stella uses Keras Core and works with multiple backends: JAX or PyTorch. You can choose either.

- Select the backend before importing ``keras``:

  .. code-block:: bash

      export KERAS_BACKEND=jax    # or torch

Switching backends (JAX ↔ PyTorch)
----------------------------------

- Install with the matching extras and set the env var before importing ``keras``:

  .. code-block:: bash

      # JAX
      pip install stella[jax]
      export KERAS_BACKEND=jax

      # PyTorch
      pip install stella[torch]
      export KERAS_BACKEND=torch

- Quick sanity check in Python:

	.. code-block:: python

		import os
		os.environ.setdefault("KERAS_BACKEND", "jax")  # or "torch"
		import keras
		print("Backend:", keras.backend.backend())
		m = keras.Sequential([
		    keras.layers.Input((8,)),
		    keras.layers.Dense(4, activation='relu'),
		    keras.layers.Dense(1, activation='sigmoid'),
		])
		print(m.predict([[0]*8]).shape)

Backend validation
------------------
To ensure a backend is available at runtime, call:

.. code-block:: python

	import stella
	stella.require_backend()  # raises with install hint if backend missing

Next steps
----------
- For an end-to-end example using the high-level helpers, see :doc:`pipeline`.

Tip: Swapping at runtime
------------------------
Because the backend is chosen when ``keras`` is first imported, changing backends inside a running Python session is not supported. stella provides helpers to inspect and prepare for a swap:

.. code-block:: python

	import stella
	stella.check_backend()  # shows installed backends and devices
	# Prepare env for torch with Apple Metal (MPS); restart recommended
	stella.swap_backend('torch', accelerator='mps')

Then restart your Python session and re-import your code.

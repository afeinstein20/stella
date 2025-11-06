# Installation

Choose a backend and install dependencies:

```bash
# JAX (CPU)
Install stella and choose a backend.

Via pip extras
--------------

```bash
# JAX (CPU)
pip install "stella[jax]"

# JAX on macOS with Metal (Apple Silicon)
pip install "stella[jax-mps]"

# PyTorch
pip install "stella[torch]"
```

From source (development)
-------------------------

```bash
git clone https://github.com/benjaminpope/stella
cd stella
pip install -e .[dev]
```

Selecting a backend
-------------------

Set the backend before importing `keras`:

```bash
export KERAS_BACKEND=jax   # or torch
```

Quick sanity check
------------------

```python
import os
os.environ.setdefault("KERAS_BACKEND", "jax")  # or "torch"
import keras
print("Backend:", keras.backend.backend())
m = keras.Sequential([
	keras.layers.Input((8,)),
	keras.layers.Dense(4, activation='relu'),
	keras.layers.Dense(1, activation='sigmoid'),
])
print(m([[0]*8]).shape)
```

Backend validation at runtime
-----------------------------

```python
import stella
stella.require_backend()  # raises with install hint if backend missing
```
```

# stella

![stella logo](assets/stella_logo.png)

stella is an open-source Python framework for identifying stellar flares in TESS two-minute cadence data using convolutional neural networks (Keras 3 on JAX or PyTorch).

- Backends: set `KERAS_BACKEND` to `jax` (default) or `torch`.
- Packaged models: `from stella import models as sm; sm.models`.
- Quickstart notebooks live under [Getting Started](getting_started/about.md).

Getting started
---------------

```bash
export KERAS_BACKEND=jax  # or torch
pip install -e .[dev]
```

Citations
---------
- [Feinstein, Montet, & Ansdell (2020), JOSS](https://ui.adsabs.harvard.edu/abs/2020JOSS....5.2347F/abstract)
- [Feinstein et al. (2020, arXiv)](https://ui.adsabs.harvard.edu/abs/2020arXiv200507710F/abstract)

Bug reports and contributions
-----------------------------
stella is MIT-licensed. 

Source and issues on GitHub:

- [Repo](https://github.com/benjaminpope/stella)
- [Issues](https://github.com/benjaminpope/stella/issues)

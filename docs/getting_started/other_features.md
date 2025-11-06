# Other Features

This page showcases a few additional helpers in `stella`: fitting flares from predictions and estimating rotation periods.

Backend
-------
Select a backend before importing `keras`:

```bash
export KERAS_BACKEND=jax   # or torch
```

Imports
-------

```python
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from lightkurve.search import search_lightcurve
from stella.neural_network import ConvNN
from stella.models import get_model_path
from stella.mark_flares import FitFlares
from stella.rotations import MeasureProt
```

Predict with a packaged model
-----------------------------

```python
lc = search_lightcurve(target='TIC 62124646', mission='TESS', sector=13, exptime=120).download().PDCSAP_FLUX
lc = lc.remove_nans().normalize()

cnn = ConvNN(output_dir='.')
cnn.predict(
    modelname=get_model_path(),
    times=lc.time.value,
    fluxes=lc.flux.value,
    errs=(lc.flux_err.value if getattr(lc, 'flux_err', None) is not None else np.zeros_like(lc.time.value)),
    verbose=False,
)
preds = cnn.predictions[0]
```

Fit flares from predictions
---------------------------

```python
ff = FitFlares(
    id=np.array([lc.targetid]),
    time=np.array([cnn.predict_time[0]]),
    flux=np.array([cnn.predict_flux[0]]),
    flux_err=np.array([np.zeros_like(cnn.predict_flux[0])]),
    predictions=np.array([preds]),
)
ff.identify_flare_peaks(threshold=0.5)
flare_table = ff.flare_table
flare_table[:5]
```

Plot a detected flare
---------------------

```python
idx = np.argmax(flare_table['prob']) if len(flare_table) else None
if idx is not None:
    tpeak = flare_table['tpeak'][idx]
    fig, (ax1, ax2) = plt.subplots(figsize=(12,6), nrows=2, sharex=True)
    ax1.plot(cnn.predict_time[0], cnn.predict_flux[0], 'k', lw=1)
    ax1.axvline(tpeak, color='crimson', ls='--', label='flare peak')
    ax1.legend()
    ax2.plot(cnn.predict_time[0], preds, 'orange')
    ax2.set_ylabel('Probability')
    ax2.set_xlabel('Time [BJD-2457000]')
    plt.show()
```

Estimate rotation period (Lomb–Scargle)
--------------------------------------

```python
mProt = MeasureProt(time=lc.time.value, flux=lc.flux.value, flux_err=(lc.flux_err.value if getattr(lc, 'flux_err', None) is not None else None))
per, amp = mProt.run_LS()
print('Best period [days]:', per)
```

Phase‑folded view
-----------------

```python
if per is not None and np.isfinite(per):
    phase = (lc.time.value % per) / per
    plt.figure(figsize=(10,4))
    plt.scatter(phase, lc.flux.value, s=4, alpha=0.5)
    plt.xlabel('Phase')
    plt.ylabel('Normalized Flux')
    plt.title(f'TIC {lc.targetid} — P={per:.3f} d')
    plt.show()
```

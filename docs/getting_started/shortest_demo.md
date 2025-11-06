# Shortest Demo

This is the quickest way to run `stella` on a TESS two‑minute light curve using a prepackaged model.

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

import stella
from stella.neural_network import ConvNN
from stella import models as sm
from lightkurve.search import search_lightcurve
```

List packaged models
--------------------

```python
models = sm.models  # list of installed model filenames
print(models)
```

Download a light curve
----------------------

```python
lc = search_lightcurve(target='TIC 62124646', mission='TESS', sector=13, exptime=120).download().PDCSAP_FLUX
lc = lc.remove_nans().normalize()
```

Predict with a single model
---------------------------

```python
cnn = ConvNN(output_dir='./results')
cnn.predict(
    modelname=models[0],
    times=lc.time.value,
    fluxes=lc.flux.value,
    errs=(lc.flux_err.value if getattr(lc, 'flux_err', None) is not None else np.zeros_like(lc.time.value)),
)
single_pred = cnn.predictions[0]
```

Plot
----

```python
plt.figure(figsize=(14,4))
plt.scatter(cnn.predict_time[0], cnn.predict_flux[0], c=single_pred, vmin=0, vmax=1)
plt.colorbar(label='Probability of Flare')
plt.xlabel('Time [BJD-2457000]')
plt.ylabel('Normalized Flux')
plt.title(f'TIC {lc.targetid}')
plt.show()
```

Ensemble (average multiple models)
----------------------------------

```python
preds = np.zeros((len(models), len(cnn.predictions[0])))
for i, model in enumerate(models):
    cnn.predict(
        modelname=model,
        times=lc.time.value,
        fluxes=lc.flux.value,
        errs=(lc.flux_err.value if getattr(lc, 'flux_err', None) is not None else np.zeros_like(lc.time.value)),
        verbose=False,
    )
    preds[i] = cnn.predictions[0]

avg_pred = np.nanmedian(preds, axis=0)
```

Compare single vs averaged
--------------------------

```python
fig, (ax1, ax2) = plt.subplots(figsize=(14,8), nrows=2, sharex=True, sharey=True)
im = ax1.scatter(cnn.predict_time[0], cnn.predict_flux[0], c=avg_pred, vmin=0, vmax=1)
ax2.scatter(cnn.predict_time[0], cnn.predict_flux[0], c=single_pred, vmin=0, vmax=1)
ax2.set_xlabel('Time [BJD-2457000]')
ax2.set_ylabel('Normalized Flux')
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.81, 0.15, 0.02, 0.7])
fig.colorbar(im, cax=cbar_ax, label='Probability')
ax1.set_title('Averaged Predictions')
ax2.set_title('Single Model Predictions')
plt.subplots_adjust(hspace=0.4)
plt.show()
```

Zoomed comparison
-----------------

```python
fig, (ax1, ax2) = plt.subplots(figsize=(14,8), nrows=2, sharex=True)
ax1.scatter(cnn.predict_time[0], cnn.predict_flux[0], c=avg_pred, vmin=0, vmax=1, cmap='Oranges_r', s=6)
ax1.scatter(cnn.predict_time[0], cnn.predict_flux[0]-0.03, c=single_pred, vmin=0, vmax=1, cmap='Greys_r', s=6)
ax1.set_ylim(0.93,1.05)
ax2.plot(cnn.predict_time[0], single_pred, 'k')
ax2.plot(cnn.predict_time[0], avg_pred, 'orange')
ax1.set_title('Black = Single Model; Orange = Averaged Models')
plt.xlim(1661,1665)
plt.show()
```

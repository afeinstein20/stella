# Tutorial

This tutorial walks through creating a dataset, initializing a CNN, training, inspecting metrics, and predicting on a TESS light curve.

Prerequisites
-------------
- Install dev deps and pick a backend:

```bash
export KERAS_BACKEND=jax  # or torch
pip install -e .[dev]
```

Imports
-------
```python
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

import stella
from stella.download_nn_set import DownloadSets
from stella.preprocessing_flares import FlareDataSet
from stella.neural_network import ConvNN
from stella.models import get_model_path, list_model_paths
from lightkurve.search import search_lightcurve
```

1) Download a small catalog subset (optional)
--------------------------------------------
```python
download = DownloadSets(fn_dir='.')
download.download_catalog()
# use a smaller subset for the tutorial
download.flare_table = download.flare_table[:100]
download.download_lightcurves()
```

2) Build a dataset
------------------
If you already have local files, point to them directly; otherwise pass the `DownloadSets` helper.

```python
# Example local paths; change to your data if needed
FN_DIR = './data/unlabeled'
CATALOG = './data/unlabeled/catalog_per_flare_final.csv'

# Choose one of the following:
# ds = FlareDataSet(downloadSet=download)
ds = FlareDataSet(fn_dir=FN_DIR, catalog=CATALOG)
```

Inspect a few training examples
-------------------------------
```python
ind_pc = np.where(ds.train_labels == 1)[0]
ind_nc = np.where(ds.train_labels == 0)[0]
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10,3), sharex=True, sharey=True)
ax1.plot(ds.train_data[ind_pc[10]], 'r'); ax1.set_title('Flare'); ax1.set_xlabel('Cadences')
ax2.plot(ds.train_data[ind_nc[10]], 'k'); ax2.set_title('No Flare'); ax2.set_xlabel('Cadences')
plt.show()
```

3) Initialize and train a model
-------------------------------
```python
cnn = ConvNN(output_dir='./results', ds=ds)
# Train a (short) run for the tutorial
cnn.train_models(seeds=2, epochs=50)
```

4) Inspect training history
---------------------------
```python
plt.figure(figsize=(7,4))
plt.plot(cnn.history_table['loss_s0002'], 'k', label='Training', lw=3)
plt.plot(cnn.history_table['val_loss_s0002'], 'darkorange', label='Validation', lw=3)
plt.xlabel('Epochs'); plt.ylabel('Loss'); plt.legend(); plt.show()
```

5) Predict on a TESS light curve
--------------------------------
```python
lk = search_lightcurve(target='TIC 62124646', mission='TESS', sector=13, exptime=120).download().PDCSAP_FLUX
lk = lk.remove_nans().normalize()

cnn = ConvNN(output_dir='.')
model_path = get_model_path()  # packaged model
err_arr = (lk.flux_err.value if getattr(lk, 'flux_err', None) is not None else np.zeros_like(lk.time.value))
cnn.predict(modelname=model_path, times=lk.time.value, fluxes=lk.flux.value, errs=err_arr, verbose=False)
```

Plot predictions
----------------
```python
plt.figure(figsize=(14,4))
plt.scatter(cnn.predict_time[0], cnn.predict_flux[0], c=cnn.predictions[0], vmin=0, vmax=1)
plt.colorbar(label='Probability of Flare')
plt.xlabel('Time [BJD-2457000]'); plt.ylabel('Normalized Flux');
plt.title(f'TIC {lk.targetid}')
plt.show()
```

Ensembling (optional)
---------------------
```python
MODELS = list_model_paths()
preds = []
for mp in MODELS:
    cnn.predict(modelname=mp, times=lk.time.value, fluxes=lk.flux.value, errs=err_arr, verbose=False)
    preds.append(cnn.predictions[0])
import numpy as _np
avg_pred = _np.nanmedian(_np.vstack(preds), axis=0)
```

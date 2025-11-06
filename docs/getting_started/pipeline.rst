.. _pipeline:

Pipeline
========

This page shows how to use the high-level pipeline API built on Keras 3 with the JAX backend. It loads a converted ``.keras`` model, runs predictions on a TESS light curve, removes obvious false positives, and fits flares.

Prerequisites
-------------

- Install requirements and select the JAX backend::

   Pipeline
   ========

   Let’s keep things simple. You bring a TESS light curve and a Keras ``.keras`` model; ``stella.pipeline`` takes care of the rest: it normalizes, filters bad cadences, runs the CNN, and helps you mark real flares.

   Setup
   -----

   - Install and pick the JAX backend:

   ::

      pip install -r requirements.txt
      pip install -e .
      export KERAS_BACKEND=jax

   Quick tour with TIC 62124646
   ----------------------------

   We’ll load a two-minute light curve, clean it (remove NaNs, normalize flux, drop non-zero ``quality``), run predictions, then identify and filter flares.

   .. code-block:: python

        from stella.pipeline import (
          predict, predict_ensemble, predict_and_mark,
          mark_flares_from_preds, remove_false_positives,
      )
      from lightkurve.search import search_lightcurve
      import numpy as np
        import os
        from stella import models as stella_models

        MODEL_PATH = stella_models.get_model_path()

      # 1) Download & clean the light curve
      lc = search_lightcurve(target='tic62124646', mission='TESS', sector=13, exptime=120).download().PDCSAP_FLUX
      lc = lc.remove_nans().normalize()
      lc = lc[lc.quality == 0]

      # 2) Predict with a single model (the pipeline also cleans LightCurve inputs internally)
      t, f, e, preds = predict(MODEL_PATH, lc)

      # 3) Turn predictions into flare candidates
      from stella.mark_flares import FitFlares
      fit = FitFlares(id=np.array([lc.targetid]), time=np.array([t]), flux=np.array([f]),
                      flux_err=np.array([e]), predictions=np.array([preds]))
      fit.identify_flare_peaks(threshold=0.5)

      # 4) Filter out obvious false positives (e.g., ultra-short fits)
      filtered = remove_false_positives(fit.flare_table, min_duration_min=4.0)
      print(filtered[:5])

   Ensembling (optional)
   ---------------------

   You can average several models for smoother predictions:

   .. code-block:: python

        from stella import models as stella_models
        MODELS = stella_models.list_model_paths()

      t, f, e, agg, per_model = predict_ensemble(MODELS, lc, aggregate='mean')
      _, flare_tab = mark_flares_from_preds(lc.targetid, t, f, e, agg, threshold=0.5)
      flare_tab = remove_false_positives(flare_tab, min_duration_min=4.0)
      print(flare_tab[:5])

   Tips
   ----

   - Passing a LightCurve to ``predict`` or ``predict_ensemble`` is convenient: the pipeline will automatically call ``remove_nans().normalize()`` and filter to ``quality == 0``.
   - If you have arrays already, pass ``(times, flux, flux_err)`` directly.
   - Tweak ``threshold`` and ``min_duration_min`` to match your science use case.

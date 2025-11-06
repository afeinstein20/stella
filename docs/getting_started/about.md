# About stella

stella identifies flares in TESS short‑cadence data with a convolutional neural network (CNN).

In its simplest form, stella takes a pre‑trained CNN (details in Feinstein et al.) and a light curve `(time, flux, flux_err)` and returns a probability light curve. Values are between 0 and 1, where 1 indicates a likely flare.

You can also train your own customized CNN architecture. See the Quickstart tutorial for a walkthrough.

Links:
- Paper: Feinstein, Montet, & Ansdell (2020, JOSS)
- Issues: https://github.com/benjaminpope/stella/issues

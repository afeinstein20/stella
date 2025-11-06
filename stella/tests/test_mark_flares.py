import numpy as np
import pytest
from numpy.testing import assert_almost_equal

pytestmark = pytest.mark.integration


def _load_lc():
    from lightkurve.search import search_lightcurve

    lk = search_lightcurve(
        target="tic62124646", mission="TESS", exptime=120, sector=13, author="SPOC"
    )
    lk = lk.download(download_dir=".")
    lk = lk.remove_nans().normalize()
    return lk


def test_predictions():
    from stella.neural_network import ConvNN
    from stella import models as sm

    lk = _load_lc()
    modelname = sm.get_model_path()
    cnn = ConvNN(output_dir=".")
    cnn.predict(
        modelname=modelname,
        times=lk.time.value,
        fluxes=lk.flux.value,
        errs=lk.flux_err.value,
    )
    high_flares = np.where(cnn.predictions[0] > 0.99)[0]
    assert len(high_flares) == 0


def test_find_flares_no_candidates():
    from stella.mark_flares import FitFlares

    lk = _load_lc()
    flares = FitFlares(
        id=[lk.targetid],
        time=[lk.time.value],
        flux=[lk.flux.value],
        flux_err=[lk.flux_err.value],
        predictions=[np.zeros_like(lk.flux.value)],
    )

    flares.identify_flare_peaks()
    assert len(flares.flare_table) == 0


def test_identify_flare_peaks_handles_ragged_object_dtype():
    """
    Regression test: previously, ragged/object-dtype inputs caused SciPy medfilt
    to error in identify_flare_peaks. This ensures no exceptions and empty table
    for zero predictions.
    """
    from stella.mark_flares import FitFlares

    lk = _load_lc()

    # Create ragged light curve inputs (different lengths) and wrap in object arrays
    t1 = lk.time.value
    f1 = lk.flux.value
    e1 = lk.flux_err.value
    p1 = np.zeros_like(f1)

    t2 = lk.time.value[:-5]
    f2 = lk.flux.value[:-5]
    e2 = lk.flux_err.value[:-5]
    p2 = np.zeros_like(f2)

    ids = np.array([lk.targetid, lk.targetid], dtype=object)
    time = np.array([t1, t2], dtype=object)
    flux = np.array([f1, f2], dtype=object)
    err = np.array([e1, e2], dtype=object)
    preds = np.array([p1, p2], dtype=object)

    flares = FitFlares(id=ids, time=time, flux=flux, flux_err=err, predictions=preds)

    # Should not raise, and with zero predictions there should be no flares
    flares.identify_flare_peaks(threshold=0.5)
    assert len(flares.flare_table) == 0

    # Spot-check grouping behavior returns object-dtype array for ragged groups
    groups = flares.group_inds(np.array([1, 2, 10, 11, 12, 40]))
    assert groups.dtype == object
    assert any(len(g) == 2 for g in groups) and any(len(g) == 3 for g in groups)

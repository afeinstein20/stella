import pytest
from numpy.testing import assert_almost_equal
from stella import models as sm

pytestmark = pytest.mark.downloads  # allow network, avoid training


def test_keras_backend_is_jax():
    import os
    pytest.importorskip("jax")

    os.environ["KERAS_BACKEND"] = "jax"
    import keras

    assert keras.backend.backend() == "jax"


def test_predict(tmp_path):
    from lightkurve.search import search_lightcurve
    from stella.neural_network import ConvNN

    model_path = sm.get_model_path()
    lk = search_lightcurve(
        target="tic62124646", mission="TESS", sector=13, exptime=120, author="SPOC"
    )
    lk = lk.download(download_dir=".")
    lk = lk.remove_nans().normalize()
    cnn = ConvNN(output_dir=".")
    import numpy as np

    err_arr = (
        lk.flux_err.value
        if getattr(lk, "flux_err", None) is not None
        else np.zeros_like(lk.time.value)
    )
    cnn.predict(
        modelname=model_path,
        times=lk.time.value,
        fluxes=lk.flux.value,
        errs=err_arr,
        verbose=False,
    )
    assert cnn.predictions.shape[0] == 1
    assert cnn.predictions[0].shape[0] == len(lk.time.value)


def _write_dummy_metrics_dir(tmp_path):
    from astropy.table import Table
    import numpy as np

    (tmp_path / "ensemble_s0002_i0010_b0.73.keras").write_text("placeholder")
    n = 10
    tab = Table()
    tab["tic"] = np.arange(n)
    tab["gt"] = np.random.randint(0, 2, size=n)
    tab["tpeak"] = np.linspace(0, 1, n)
    tab["pred_s0002"] = np.random.rand(n)
    tab.write(
        tmp_path / "ensemble_predval_i0010_b0.73.txt", format="ascii", overwrite=True
    )
    h = Table()
    h["precision_s0002"] = np.random.rand(5)
    h.write(
        tmp_path / "ensemble_histories_i0010_b0.73.txt", format="ascii", overwrite=True
    )
    return str(tmp_path)


def test_create_metrics(tmp_path):
    from stella.metrics import ModelMetrics

    fn_dir = _write_dummy_metrics_dir(tmp_path)
    metrics = ModelMetrics(fn_dir=fn_dir)
    assert metrics.mode == "ensemble"
    assert len(metrics.predval_table) > 0
    assert metrics.history_table.colnames[0].startswith("precision")


def test_ensemble(tmp_path):
    from stella.metrics import ModelMetrics

    fn_dir = _write_dummy_metrics_dir(tmp_path)
    metrics = ModelMetrics(fn_dir=fn_dir)
    metrics.calculate_ensemble_metrics()
    assert 0.0 <= metrics.ensemble_accuracy <= 1.0
    assert 0.0 <= metrics.ensemble_avg_precision <= 1.0

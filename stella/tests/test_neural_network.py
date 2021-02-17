from stella.metrics import *
from stella.download_nn_set import *
from stella.preprocessing_flares import *
from stella.neural_network import *
from numpy.testing import assert_almost_equal

download = DownloadSets(fn_dir='.')
download.download_catalog()
download.flare_table = download.flare_table[0:20]
download.download_lightcurves(remove_fits=False)
pre = FlareDataSet(downloadSet=download)

def test_catalog_retrieval():
    assert(download.flare_table['TIC'][0] == 2760232)
    assert_almost_equal(download.flare_table['tpeak'][10], 2458368.8, decimal=1)
    assert(download.flare_table['Flare'][9] == 3)

def test_light_curves():
    download.download_lightcurves(remove_fits=False)

def test_processing():
    assert_almost_equal(pre.frac_balance, 0.7, decimal=1)
    assert(pre.train_data.shape == (48, 200, 1))
    assert(pre.val_data.shape == (6, 200, 1))
    assert(pre.test_data.shape == (7, 200, 1))

def test_tensorflow():
    import tensorflow
    assert(tensorflow.__version__ == '2.4.1')

cnn = ConvNN(output_dir='.', ds=pre)
cnn.train_models(epochs=10, save=True, pred_test=True)

def test_train_model():
    assert(cnn.loss == 'binary_crossentropy')
    assert(cnn.optimizer == 'adam')
    assert(cnn.training_ids[10] == 2760232.0)
    assert(cnn.frac_balance == 0.73)
    assert(len(cnn.val_pred_table) == 6)


def test_predict():
    from lightkurve.search import search_lightcurve

    lk = search_lightcurve(target='tic62124646', mission='TESS',
                           sector=13, exptime=120, author='SPOC')
    lk = lk.download(download_dir='.')#.PDCSAP_FLUX
    lk = lk.remove_nans()

    cnn.predict(modelname='ensemble_s0002_i0010_b0.73.h5',
                times=lk.time,
                fluxes=lk.flux,
                errs=lk.flux_err)
    assert(cnn.predictions.shape == (1,17939))
    assert_almost_equal(cnn.predictions[0][1000], 0.3, decimal=1)

metrics = ModelMetrics(fn_dir='.')

def test_create_metrics():
    assert(metrics.mode == 'ensemble')
    assert(len(metrics.predtest_table)==7)
    assert(metrics.predval_table['gt'][0] == 1)
    assert(metrics.history_table.colnames[2] == 'precision_s0002')

def test_ensemble():
    metrics.calculate_ensemble_metrics()

    assert(metrics.ensemble_accuracy == 0.0)
    assert(metrics.ensemble_avg_precision == 1.0)

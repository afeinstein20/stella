from stella.preprocessing_flares import *
from stella.neural_network import *
from numpy.testing import assert_almost_equal

pre = FlareDataSet(fn_dir='.',
                   catalog='Guenther_2020_flare_catalog.txt')
cnn = ConvNN(output_dir='.',
             ds=pre)

def test_tensorflow():
    import tensorflow
    assert(tensorflow.__version__ == '2.1.0')

def test_build_model():
    cnn.train_models(epochs=10)

    assert(cnn.loss == 'binary_crossentropy')
    assert(cnn.optimizer == 'adam')
    assert(cnn.training_ids[10] == 2760232.0)
    assert(cnn.frac_balance == 0.73)
    assert(len(cnn.val_pred_table) == 6)

    from lightkurve.search import search_lightcurvefile

    lk = search_lightcurvefile(target='tic62124646', mission='TESS')
    lk = lk.download().PDCSAP_FLUX
    lk = lk.remove_nans()
    
    def test_predict():
        cnn.predict(modelname='ensemble_s0002_i0010_b0.73.h5',
                    times=lk.time,
                    fluxes=lk.flux,
                    errs=lk.flux_err)
        assert(cnn.predictions.shape == (1,17939))
        assert_almost_equal(cnn.predictions[0][1000], 0.3, decimal=1)

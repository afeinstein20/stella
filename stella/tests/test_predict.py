from stella.neural_network import *
from numpy.testing import assert_almost_equal
from lightkurve.search import search_lightcurvefile

cnn = ConvNN(output_dir = '.')

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
    

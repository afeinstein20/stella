import numpy as np
from stella import ConvNN
from lightkurve.search import search_lightcurvefile
from numpy.testing import assert_almost_equal

lk = search_lightcurvefile(target='tic62124646', mission='TESS')
lk = lk.download().PDCSAP_FLUX
lk = lk.remove_nans()
modelname = 'ensemble_s0002_i0010_b0.73.h5'

def test_predictions():
    cnn = ConvNN(output_dir='.')
    cnn.predict(modelname=modelname,
                times=lk.time,
                fluxes=lk.flux,
                errs=lk.flux_err)
    high_flares = np.where(cnn.predictions[0]>0.99)[0]
    assert(high_flares[0]==780)
    assert(high_flares[100]==14231)
    assert(len(high_flares)==255)

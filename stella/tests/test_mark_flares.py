import numpy as np
from stella import ConvNN
from stella import FitFlares
from lightkurve.search import search_lightcurvefile
from numpy.testing import assert_almost_equal

lk = search_lightcurvefile(target='tic62124646', mission='TESS')
lk = lk.download().PDCSAP_FLUX
lk = lk.remove_nans()
modelname = 'ensemble_s0002_i0010_b0.73.h5'

cnn = ConvNN(output_dir='.')

def test_predictions():
    cnn.predict(modelname=modelname,
                times=lk.time,
                fluxes=lk.flux,
                errs=lk.flux_err)
    high_flares = np.where(cnn.predictions[0]>0.99)[0]
    assert(len(high_flares) == 0)

def find_flares():
    flares = FitFlares(id=[lk.targetid],
                       time=[lk.time],
                       flux=[lk.flux],
                       flux_err=[lk.flux_err],
                       predictions=[cn.predictions[0]])

    flares.identify_flare_peaks()
    assert(len(flares.flare_table)==0)

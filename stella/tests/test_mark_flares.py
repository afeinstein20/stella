import numpy as np
from stella import ConvNN
from stella import FitFlares
from lightkurve.search import search_lightcurve
from numpy.testing import assert_almost_equal

lk = search_lightcurve(target='tic62124646', mission='TESS',
                       exptime=120, sector=13, author='SPOC')
lk = lk.download(download_dir='.')
lk = lk.remove_nans().normalize()
modelname = 'ensemble_s0002_i0010_b0.73.h5'

cnn = ConvNN(output_dir='.')

def test_predictions():
    cnn.predict(modelname=modelname,
                times=lk.time.value,
                fluxes=lk.flux.value,
                errs=lk.flux_err.value)
    high_flares = np.where(cnn.predictions[0]>0.99)[0]
    assert(len(high_flares) == 0)

def find_flares():
    flares = FitFlares(id=[lk.targetid],
                       time=[lk.time.value],
                       flux=[lk.flux.value],
                       flux_err=[lk.flux_err.value],
                       predictions=[cn.predictions[0]])

    flares.identify_flare_peaks()
    assert(len(flares.flare_table)==0)

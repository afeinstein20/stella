from stella.rotations import MeasureProt
from lightkurve.search import search_lightcurve
from numpy.testing import assert_almost_equal

lk = search_lightcurve(target='tic62124646', mission='TESS',
                       exptime=120, sector=13, author='SPOC')
lk = lk.download(download_dir='.')
lk = lk.remove_nans().normalize()

mProt = MeasureProt([lk.targetid], [lk.time.value], 
                    [lk.flux.value], [lk.flux_err.value])
mProt.run_LS()

def test_measurement():
    assert_almost_equal(mProt.LS_results['period_days'], 3.2, decimal=1)
    assert(mProt.LS_results['Flags']==0)

    

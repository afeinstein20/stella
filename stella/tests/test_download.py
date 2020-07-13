from ..download_nn_set import *
from numpy.testing import assert_almost_equal

download = DownloadSets(fn_dir='.')
download.download_catalog()

def test_catalog_retrieval():
    assert(download.flare_table['TIC'][0] == 2760232)
    assert_almost_equal(download.flare_table['tpeak'][100], 2458379.9, decimal=1)
    assert(download.flare_table['Flare'][4000] == 1)

def test_lightcurves():
    download.flare_table = download.flare_table[0:20]
    download.download_lightcurves()

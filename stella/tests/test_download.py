from stella.download_nn_set import *
from stella.preprocessing_flares import *
from numpy.testing import assert_almost_equal

download = DownloadSets(fn_dir='.')
download.download_catalog()

def test_catalog_retrieval():
    assert(download.flare_table['TIC'][0] == 2760232)
    assert_almost_equal(download.flare_table['tpeak'][100], 2458379.9, decimal=1)
    assert(download.flare_table['Flare'][4000] == 1)
    assert(len(download.flare_table) == 8695)
    

def test_lightcurves():
    download.flare_table = download.flare_table[0:20]
    download.download_lightcurves()


def test_processing():
    pre = FlareDataSet(downloadSet=download)
    assert_almost_equal(pre.frac_balance, 0.7, decimal=1)
    assert(pre.train_data.shape == (48, 200, 1))
    assert(pre.val_data.shape == (6, 200, 1))
    assert(pre.test_data.shape == (7, 200, 1))

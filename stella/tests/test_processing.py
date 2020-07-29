from stella.preprocessing_flares import *
from numpy.testing import assert_almost_equal

pre = FlareDataSet(fn_dir='.',
                   catalog='Guenther_2020_flare_catalog.txt')

def test_processing():
    assert_almost_equal(pre.frac_balance, 0.7, decimal=1)
    assert(pre.train_data.shape == (62, 200, 1))
    assert(pre.val_data.shape == (8, 200, 1))
    assert(pre.test_data.shape == (8, 200, 1))

import pytest
from numpy.testing import assert_almost_equal

pytestmark = pytest.mark.integration


def _make_dataset():
    from stella.preprocessing_flares import FlareDataSet

    return FlareDataSet(fn_dir=".", catalog="Guenther_2020_flare_catalog.txt")


def test_processing():
    pre = _make_dataset()
    assert_almost_equal(pre.frac_balance, 0.7, decimal=1)
    assert pre.train_data.shape == (62, 200, 1)
    assert pre.val_data.shape == (8, 200, 1)
    assert pre.test_data.shape == (8, 200, 1)

from pathlib import Path

import numpy as np
import pytest

sdba = pytest.importorskip("xclim.sdba")  # noqa
from xclim.sdba.detrending import PolyDetrend


def test_poly_detrend(series):
    x = series(np.arange(20 * 365.25), "tas")

    poly = PolyDetrend(degree=1)
    fx = poly.fit(x)
    dx = fx.detrend(x)
    xt = fx.retrend(dx)

    # The precision suffers due to 2 factors:
    # - The date is approximate (middle of the period)
    # - The last period may not be complete.
    np.testing.assert_array_almost_equal(dx, 0)
    np.testing.assert_array_almost_equal(xt, x)


def test_detrend_save_fit(series):
    x = series(np.arange(20 * 365.25), "tas")
    poly = PolyDetrend(degree=1)
    fx = poly.fit(x)
    fx2 = poly.fit(x)
    fx.save_fit()
    fx2.save_fit(filename="test.nc")

    tmpfile = Path(fx.ds.encoding["source"])
    tmpfile2 = Path(fx2.ds.encoding["source"])
    assert tmpfile.is_file()
    del fx
    assert not tmpfile.is_file()

    assert tmpfile2.name == "test.nc"
    del fx2
    assert tmpfile2.is_file()
    tmpfile2.unlink()

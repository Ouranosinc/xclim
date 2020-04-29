import numpy as np
import pytest

sdba = pytest.importorskip("xclim.sdba")  # noqa
from xclim.sdba.detrending import PolyDetrend


@pytest.mark.parametrize("freq", (None, "MS", "YS", "QS", "M", "Y", "Q"))
def test_poly_detrend(series, freq):
    x = series(np.arange(20 * 365.25), "tas")

    poly = PolyDetrend(degree=1, freq=freq)
    fx = poly.fit(x)
    dx = fx.detrend(x)
    xt = fx.retrend(dx)

    # The precision suffers due to 2 factors:
    # - The date is approximate (middle of the period)
    # - The last period may not be complete.
    dec = 6 if freq is None else 0

    np.testing.assert_array_almost_equal(dx, 0, dec)
    np.testing.assert_array_almost_equal(xt, x)

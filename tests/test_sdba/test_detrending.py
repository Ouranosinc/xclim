import numpy as np

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

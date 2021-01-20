import numpy as np
import pytest

from xclim.sdba.detrending import LoessDetrend, PolyDetrend


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


@pytest.mark.slow
def test_loess_detrend(series):
    x = series(np.arange(12 * 365.25), "tas")
    det = LoessDetrend(group="time", d=0, niter=1, f=0.2)
    fx = det.fit(x)
    dx = fx.detrend(x)
    xt = fx.retrend(dx)

    # Strong boundary effects in LOESS, remove ~ f * Nx on each side.
    np.testing.assert_array_almost_equal(dx.isel(time=slice(880, 3500)), 0)
    np.testing.assert_array_almost_equal(xt, x)

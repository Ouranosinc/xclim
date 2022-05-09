from __future__ import annotations

import numpy as np
import pytest
import xarray as xr
from scipy.signal import windows

from xclim.sdba import Grouper
from xclim.sdba.detrending import (
    LoessDetrend,
    MeanDetrend,
    NoDetrend,
    PolyDetrend,
    RollingMeanDetrend,
)


def test_poly_detrend_and_from_ds(series, tmp_path):
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

    file = tmp_path / "test_polydetrend.nc"
    fx.ds.to_netcdf(file)

    ds = xr.open_dataset(file)
    fx2 = PolyDetrend.from_dataset(ds)

    xr.testing.assert_equal(fx.ds, fx2.ds)
    dx2 = fx2.detrend(x)
    np.testing.assert_array_equal(dx, dx2)


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


def test_mean_detrend(series):
    x = series(np.arange(20 * 365.25), "tas")

    md = MeanDetrend().fit(x)
    assert (md.ds.trend == x.mean()).all()

    anomaly = md.detrend(x)
    x2 = md.retrend(anomaly)

    np.testing.assert_array_almost_equal(x, x2)


def test_rollingmean_detrend(series):
    x = series(np.arange(12 * 365.25), "tas")
    det = RollingMeanDetrend(group="time", win=29, min_periods=1)
    fx = det.fit(x)
    dx = fx.detrend(x)
    xt = fx.retrend(dx)

    np.testing.assert_array_almost_equal(dx.isel(time=slice(30, 3500)), 0)
    np.testing.assert_array_almost_equal(xt, x)

    # weights + grouping
    x = xr.DataArray(
        np.sin(2 * np.pi * np.arange(11 * 365) / 365),
        dims=("time",),
        coords={
            "time": xr.cftime_range(
                "2010-01-01", periods=11 * 365, freq="D", calendar="noleap"
            )
        },
    )
    w = windows.get_window("triang", 11, False)
    det = RollingMeanDetrend(
        group=Grouper("time.dayofyear", window=3), win=11, weights=w
    )
    fx = det.fit(x)
    assert fx.ds.trend.notnull().sum() == 365


def test_no_detrend(series):
    x = series(np.arange(12 * 365.25), "tas")

    det = NoDetrend(group="time.dayofyear", kind="+")

    with pytest.raises(ValueError, match="You must call fit()"):
        det.retrend(x)

    with pytest.raises(ValueError, match="You must call fit()"):
        det.detrend(x)

    assert repr(det).endswith("unfitted>")

    fit = det.fit(x)

    np.testing.assert_array_equal(fit.retrend(x), x)
    np.testing.assert_array_equal(fit.detrend(x), x)

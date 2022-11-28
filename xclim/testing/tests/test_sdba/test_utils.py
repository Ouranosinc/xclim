from __future__ import annotations

import numpy as np
import pytest
import xarray as xr
from scipy.stats import norm

from xclim.sdba import nbutils as nbu
from xclim.sdba import utils as u
from xclim.sdba.base import Grouper


def test_ecdf(series):
    dist = norm(5, 2)
    r = dist.rvs(10000)
    q = [0.01, 0.5, 0.99]
    x = xr.DataArray(dist.ppf(q), dims=("q",))
    np.testing.assert_allclose(u.ecdf(series(r, "tas"), x), q, 3)

    # With NaNs
    r[:2000] = np.nan
    np.testing.assert_allclose(u.ecdf(series(r, "tas"), x), q, 3)


def test_map_cdf(series):
    n = 10000
    xd = norm(5, 2)
    yd = norm(7, 3)

    q = [0.1, 0.5, 0.99]
    x_value = u.map_cdf(
        xr.Dataset(dict(x=series(xd.rvs(n), "pr"), y=series(yd.rvs(n), "pr"))),
        y_value=yd.ppf(q),
        dim=["time"],
    )
    np.testing.assert_allclose(x_value, xd.ppf(q), 0.1)

    # Scalar
    q = 0.5
    x_value = u.map_cdf(
        xr.Dataset(dict(x=series(xd.rvs(n), "pr"), y=series(yd.rvs(n), "pr"))),
        y_value=yd.ppf(q),
        dim=["time"],
    )
    np.testing.assert_allclose(x_value, [xd.ppf(q)], 0.1)


def test_equally_spaced_nodes():
    x = u.equally_spaced_nodes(5, eps=1e-4)
    assert len(x) == 7
    d = np.diff(x)
    np.testing.assert_almost_equal(d[0], d[1] / 2, 3)

    x = u.equally_spaced_nodes(1)
    np.testing.assert_almost_equal(x[0], 0.5)


@pytest.mark.parametrize(
    "interp,expi", [("nearest", 2.9), ("linear", 2.95), ("cubic", 2.95)]
)
@pytest.mark.parametrize("extrap,expe", [("constant", 4.4), ("nan", np.NaN)])
def test_interp_on_quantiles_constant(interp, expi, extrap, expe):
    quantiles = np.linspace(0, 1, num=25)
    xq = xr.DataArray(
        np.linspace(205, 229, num=25),
        dims=("quantiles",),
        coords={"quantiles": quantiles},
    )

    yq = xr.DataArray(
        np.linspace(2, 4.4, num=25),
        dims=("quantiles",),
        coords={"quantiles": quantiles},
    )

    newx = xr.DataArray(
        np.linspace(240, 200, num=41) - 0.5,
        dims=("time",),
        coords={"time": xr.cftime_range("1900-03-01", freq="D", periods=41)},
    )
    newx = newx.where(newx > 201)  # Put some NaNs in newx

    xq = xq.expand_dims(lat=[1, 2, 3])
    yq = yq.expand_dims(lat=[1, 2, 3])
    newx = newx.expand_dims(lat=[1, 2, 3])

    out = u.interp_on_quantiles(
        newx, xq, yq, group="time", method=interp, extrapolation=extrap
    )

    if np.isnan(expe):
        assert out.isel(time=0).isnull().all()
    else:
        assert out.isel(lat=1, time=0) == expe
    np.testing.assert_allclose(out.isel(time=25), expi)
    assert out.isel(time=-1).isnull().all()

    xq = xq.where(xq != 220)
    yq = yq.where(yq != 3)
    out = u.interp_on_quantiles(
        newx, xq, yq, group="time", method=interp, extrapolation=extrap
    )

    if np.isnan(expe):
        assert out.isel(time=0).isnull().all()
    else:
        assert out.isel(lat=1, time=0) == expe
    np.testing.assert_allclose(out.isel(time=25), expi)
    assert out.isel(time=-1).isnull().all()


def test_interp_on_quantiles_monthly():
    t = xr.cftime_range("2000-01-01", "2030-12-31", freq="D", calendar="noleap")
    ref = xr.DataArray(
        (
            -20 * np.cos(2 * np.pi * t.dayofyear / 365)
            + 2 * np.random.random_sample((t.size,))
            + 273.15
            + 0.1 * (t - t[0]).days / 365
        ),  # "warming" of 1K per decade,
        dims=("time",),
        coords={"time": t},
        attrs={"units": "K"},
    )
    sim = xr.DataArray(
        (
            -18 * np.cos(2 * np.pi * t.dayofyear / 365)
            + 2 * np.random.random_sample((t.size,))
            + 273.15
            + 0.11 * (t - t[0]).days / 365
        ),  # "warming" of 1.1K per decade
        dims=("time",),
        coords={"time": t},
        attrs={"units": "K"},
    )

    ref = ref.sel(time=slice(None, "2015-01-01"))
    hist = sim.sel(time=slice(None, "2015-01-01"))

    group = Grouper("time.month")
    quantiles = u.equally_spaced_nodes(15, eps=1e-6)
    ref_q = group.apply(nbu.quantile, ref, main_only=True, q=quantiles)
    hist_q = group.apply(nbu.quantile, hist, main_only=True, q=quantiles)
    af = u.get_correction(hist_q, ref_q, "+")

    for interp in ["nearest", "linear", "cubic"]:
        afi = u.interp_on_quantiles(
            sim, hist_q, af, group="time.month", method=interp, extrapolation="constant"
        )
        assert afi.isnull().sum("time") == 0, interp


@pytest.mark.parametrize(
    "interp,expi", [("nearest", 2.9), ("linear", 2.95), ("cubic", 2.95)]
)
@pytest.mark.parametrize("extrap,expe", [("constant", 4.4), ("nan", np.NaN)])
def test_interp_on_quantiles_constant_with_nan(interp, expi, extrap, expe):
    quantiles = np.linspace(0, 1, num=30)
    xq = xr.DataArray(
        np.append(np.linspace(205, 229, num=25), [np.nan] * 5),
        dims=("quantiles",),
        coords={"quantiles": quantiles},
    )

    yq = xr.DataArray(
        np.append(np.linspace(2, 4.4, num=25), [np.nan] * 5),
        dims=("quantiles",),
        coords={"quantiles": quantiles},
    )

    newx = xr.DataArray(
        np.linspace(240, 200, num=41) - 0.5,
        dims=("time",),
        coords={"time": xr.cftime_range("1900-03-01", freq="D", periods=41)},
    )
    newx = newx.where(newx > 201)  # Put some NaNs in newx

    xq = xq.expand_dims(lat=[1, 2, 3])
    yq = yq.expand_dims(lat=[1, 2, 3])
    newx = newx.expand_dims(lat=[1, 2, 3])

    out = u.interp_on_quantiles(
        newx, xq, yq, group="time", method=interp, extrapolation=extrap
    )

    if np.isnan(expe):
        assert out.isel(time=0).isnull().all()
    else:
        assert out.isel(lat=1, time=0) == expe
    np.testing.assert_allclose(out.isel(time=25), expi)
    assert out.isel(time=-1).isnull().all()

    xq = xq.where(xq != 220)
    yq = yq.where(yq != 3)
    out = u.interp_on_quantiles(
        newx, xq, yq, group="time", method=interp, extrapolation=extrap
    )

    if np.isnan(expe):
        assert out.isel(time=0).isnull().all()
    else:
        assert out.isel(lat=1, time=0) == expe
    np.testing.assert_allclose(out.isel(time=25), expi)
    assert out.isel(time=-1).isnull().all()


def test_rank():
    arr = np.random.random_sample(size=(10, 10, 1000))
    da = xr.DataArray(arr, dims=("x", "y", "time"))

    ranks = u.rank(da, dim="time", pct=False)

    exp = arr.argsort().argsort() + 1

    np.testing.assert_array_equal(ranks.values, exp)

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from scipy.stats import norm

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
    x = u.equally_spaced_nodes(5)
    assert len(x) == 7
    d = np.diff(x)
    np.testing.assert_almost_equal(d[0], d[1] / 2, 3)

    x = u.equally_spaced_nodes(1, eps=None)
    np.testing.assert_almost_equal(x[0], 0.5)


@pytest.mark.parametrize(
    "method,exp", [("nan", [np.NaN, -np.inf]), ("constant", [0, -np.inf])]
)
def test_extrapolate_qm(make_qm, method, exp):
    qm = make_qm(np.arange(6).reshape(2, 3))
    xq = make_qm(np.arange(6).reshape(2, 3))

    q, x = u.extrapolate_qm(qm, xq, method=method)

    assert isinstance(q, xr.DataArray)
    assert isinstance(x, xr.DataArray)
    if np.isnan(exp[0]):
        assert q[0, 0].isnull()
    else:
        assert q[0, 0] == exp[0]
    assert x[0, 0] == exp[1]


@pytest.mark.parametrize("group", ["time", "time.month"])
@pytest.mark.parametrize(
    "interp,expi", [("nearest", 2.9), ("linear", 2.95), ("cubic", 2.95)]
)
@pytest.mark.parametrize("extrap,expe", [("constant", 4.4), ("nan", np.NaN)])
def test_interp_on_quantiles_constant(group, interp, expi, extrap, expe):
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

    if group == "time.month":
        xq = xq.expand_dims(month=np.arange(12) + 1)
        yq = yq.expand_dims(month=np.arange(12) + 1)

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
        newx, xq, yq, group=group, method=interp, extrapolation=extrap
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
        newx, xq, yq, group=group, method=interp, extrapolation=extrap
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

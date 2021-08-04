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


@pytest.mark.parametrize("method,exp", [("nan", [0, 0]), ("constant", [0, -np.inf])])
def test_extrapolate_qm(make_qm, method, exp):
    qm = make_qm(np.arange(6).reshape(2, 3))
    xq = make_qm(np.arange(6).reshape(2, 3))

    q, x = u.extrapolate_qm(qm, xq, method=method)

    assert isinstance(q, xr.DataArray)
    assert isinstance(x, xr.DataArray)
    assert q[0, 0] == exp[0]
    assert x[0, 0] == exp[1]


@pytest.mark.parametrize("shape", [(2920,), (2920, 5, 5)])
@pytest.mark.parametrize("group", ["time", "time.month"])
@pytest.mark.parametrize("method", ["nearest", "linear", "cubic"])
def test_interp_on_quantiles(shape, group, method):
    group = Grouper(group)
    raw = np.random.random_sample(shape)  # [0, 1]
    t = pd.date_range("2000-01-01", periods=shape[0], freq="D")
    # obs : [9, 11]
    obs = xr.DataArray(
        raw * 2 + 9, dims=("time", "lat", "lon")[: len(shape)], coords={"time": t}
    )
    # sim [9, 11.4] (x1.2 + 0.2)
    sim = xr.DataArray(
        raw * 2.4 + 9, dims=("time", "lat", "lon")[: len(shape)], coords={"time": t}
    )
    # fut [9.02, 11.38] (x1.18 + 0.2) In order to have every point of fut inside the range of sim
    fut_raw = raw * 2.36 + 9.02
    fut_raw[
        np.array([100, 300, 500, 700])
    ] = 1000  # Points outside the sim range will be NaN
    fut = xr.DataArray(
        fut_raw, dims=("time", "lat", "lon")[: len(shape)], coords={"time": t}
    )

    q = np.linspace(0, 1, 11)
    xq = group.apply("quantile", sim, q=q).rename(quantile="quantiles")
    yq = group.apply("quantile", obs, q=q).rename(quantile="quantiles")

    fut_corr = u.interp_on_quantiles(fut, xq, yq, group=group, method=method).transpose(
        *("time", "lat", "lon")[: len(shape)]
    )

    if method == "nearest":
        np.testing.assert_allclose(fut_corr.values, obs.values, rtol=0.3)
        assert fut_corr.isnull().sum() == 0
    else:
        np.testing.assert_allclose(
            fut_corr.values, obs.where(fut != 1000).values, rtol=2e-3
        )
        xr.testing.assert_equal(fut_corr.isnull(), fut == 1000)


def test_rank():
    arr = np.random.random_sample(size=(10, 10, 1000))
    da = xr.DataArray(arr, dims=("x", "y", "time"))

    ranks = u.rank(da, dim="time", pct=False)

    exp = arr.argsort().argsort() + 1

    np.testing.assert_array_equal(ranks.values, exp)

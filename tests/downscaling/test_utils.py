import numpy as np
import pandas as pd
import pytest
import xarray as xr

from xclim.downscaling import utils as u


def test_jitter_under_thresh():
    da = xr.DataArray([0.5, 2.1, np.nan])
    out = u.jitter_under_thresh(da, 1)

    assert da[0] != out[0]
    assert da[0] < 1
    assert da[0] > 0
    np.testing.assert_allclose(da[1:], out[1:])


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


# class TestReindex:
#     def test_month(self, make_qm):
#         qm = make_qm(np.arange(6).reshape(2, 3))
#         xq = make_qm(np.arange(6).reshape(2, 3))

#         out = u.reindex(qm, xq, extrapolation="nan")
#         np.testing.assert_array_equal(
#             out.T,
#             [
#                 [0, np.nan, np.nan],
#                 [1, 1, np.nan],
#                 [2, 2, 2],
#                 [3, 3, 3],
#                 [np.nan, 4, 4],
#                 [np.nan, np.nan, 5],
#             ],
#         )

#         out = u.reindex(qm, xq, extrapolation="constant")
#         np.testing.assert_array_equal(
#             out.T,
#             [
#                 [0, 1, 2],
#                 [0, 1, 2],
#                 [1, 1, 2],
#                 [2, 2, 2],
#                 [3, 3, 3],
#                 [3, 4, 4],
#                 [3, 4, 5],
#                 [3, 4, 5],
#             ],
#         )

#         assert out.dims == ("month", "x")
#         assert isinstance(out.attrs["quantiles"], np.ndarray)

#     def test_time(self, make_qm):
#         qm = make_qm(np.arange(4), "time")
#         xq = make_qm(np.arange(4), "time")

#         out = u.reindex(qm, xq, extrapolation="nan")
#         assert out.dims == ("x",)
#         assert isinstance(out.attrs["quantiles"], np.ndarray)

#         np.testing.assert_array_equal(out.x, xq)


def test_adjust_freq_1d_simple_nan():
    a = np.array([0, 0, 1, np.nan, 3, 4])
    b = np.array([0, 0, 0, 1, np.nan, 3])

    out = u._adjust_freq_1d(a, b, 1)
    np.testing.assert_equal(out, [0, 0, 0, np.nan, 3, 4])

    out = u._adjust_freq_1d(b, a, 1)
    np.testing.assert_equal(out, [0, 0, 1, 1, np.nan, 3])


def test_adjust_freq_1d_dist():
    v = np.random.randint(1, 100, 1000).astype(float)
    b = np.where(v < 30, v / 30, v)  # sim
    a = np.where(v < 10, v / 30, v)  # obs

    out = u._adjust_freq_1d(b, a, 1)

    # The threshold we should check against is the corresponding threshold for b, not 1.
    np.testing.assert_array_less(1, out[(v >= 10) & (v < 30)])
    np.testing.assert_array_equal(out[v >= 30], a[v >= 30])


def test_adjust_freq_1d_dist_nan():
    v = np.random.randint(1, 100, 1000).astype(float)
    b = np.where(v < 40, v / 40, v)  # sim 40% under thresh
    a = np.where(v < 10, v / 30, v)  # obs 10% under thresh
    # a[-1:] = np.nan  # 20 % under thresh when discounting nans

    out = u._adjust_freq_1d(b, a, 1)

    # Some of these could be lower than tresh because they are randomly generated.
    np.testing.assert_array_less(1, out[(a >= 10) & (a < 40)])

    # Some of these could have been set to 0 due to the nans
    np.testing.assert_array_equal(out[a >= 40], a[a >= 40])


def test_adjust_freq():
    time = pd.date_range("1993-01-01", "2000-12-31", freq="D")
    prvals = np.random.randint(0, 100, size=(time.size, 3))
    pr = xr.DataArray(
        prvals, coords={"time": time, "lat": [0, 1, 2]}, dims=("time", "lat")
    )
    prsim = xr.where(pr < 20, pr / 20, pr)
    probs = xr.where(pr < 10, pr / 20, pr)
    prsim_ad = u.adjust_freq(probs, prsim, 1, "time.month")

    xr.testing.assert_equal(
        (probs < 1).groupby("time.month").sum().T,
        (prsim_ad < 1).groupby("time.month").sum(),
    )

    u.adjust_freq(probs, prsim, 1, "time")


@pytest.mark.parametrize("shape", [(2920,), (2920, 5, 5)])
@pytest.mark.parametrize("group", ["time", "time.month"])
@pytest.mark.parametrize("method", ["nearest", "linear", "cubic"])
def test_interp_on_quantiles(shape, group, method):
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

    dim, prop = u.parse_group(group)
    if prop is not None:
        fut = fut.assign_coords(month=u.get_index(fut, dim, prop, True))

    q = np.linspace(0, 1, 11)
    xq = u.group_apply("quantile", sim, group, q=q).rename(quantile="quantiles")
    yq = u.group_apply("quantile", obs, group, q=q).rename(quantile="quantiles")

    if prop is not None:
        xq = u.add_cyclic_bounds(xq, prop, cyclic_coords=False)
        yq = u.add_cyclic_bounds(yq, prop, cyclic_coords=False)

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
        if prop is not None:
            fut = fut.drop_vars(prop)
        xr.testing.assert_equal(fut_corr.isnull(), fut == 1000)

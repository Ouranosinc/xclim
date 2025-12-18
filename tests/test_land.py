"""Tests for indicators in `land` realm."""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from xclim import land


def test_base_flow_index(ndq_series):
    out = land.base_flow_index(ndq_series, freq="YS")

    assert out.attrs["units"] == "1"
    assert isinstance(out, xr.DataArray)


def test_rb_flashiness_index(ndq_series):
    out = land.base_flow_index(ndq_series, freq="YS")

    assert out.attrs["units"] == "1"
    assert isinstance(out, xr.DataArray)


def test_qdoy_max(ndq_series, q_series):
    out = land.doy_qmax(ndq_series, freq="YS", season="JJA")
    assert out.attrs["units"] == "1"

    a = np.ones(450)
    a[100] = 2
    out = land.doy_qmax(q_series(a), freq="YS")
    assert out[0] == 101


def test_snow_melt_we_max(snw_series):
    a = np.zeros(365)
    a[10] = 5
    snw = snw_series(a)
    out = land.snow_melt_we_max(snw)
    assert out[0] == 5


def test_blowing_snow(snd_series, sfcWind_series):
    a = np.zeros(366)
    a[10:20] = np.arange(10)
    snd = snd_series(a, start="2001-07-1")
    ws = sfcWind_series(a, start="2001-07-1")

    out = land.blowing_snow(snd, ws, snd_thresh="50 cm", sfcWind_thresh="5 km/h")
    np.testing.assert_array_equal(out, [5, np.nan])


def test_snd_storm_days(snd_series):
    a = np.zeros(366)
    a[10:20] = np.arange(10)

    snd = snd_series(a)
    out = land.snd_storm_days(snd, thresh="50 cm")
    np.testing.assert_array_equal(out, [9, np.nan])


def test_snw_storm_days(snw_series):
    a = np.zeros(366)
    a[10:20] = np.arange(10)

    snw = snw_series(a)
    out = land.snw_storm_days(snw, thresh="0.5 kg m-2")
    np.testing.assert_array_equal(out, [9, np.nan])


def test_flow_index(q_series):
    a = np.ones(365 * 2) * 10
    a[10:50] = 50
    q = q_series(a)

    out = land.flow_index(q, p=0.95)
    np.testing.assert_array_equal(out, 5)


def test_high_flow_frequency(q_series):
    a = np.zeros(366 * 2) * 10
    a[50:60] = 10
    a[200:210] = 20
    q = q_series(a)
    out = land.high_flow_frequency(
        q,
        threshold_factor=9,
        freq="YS",
    )
    np.testing.assert_array_equal(out, [20, 0, np.nan])


def test_low_flow_frequency(q_series):
    a = np.ones(366 * 2) * 10
    a[50:60] = 1
    a[200:210] = 1
    q = q_series(a)
    out = land.low_flow_frequency(q, threshold_factor=0.2, freq="YS")
    np.testing.assert_array_equal(out, [20, 0, np.nan])


def test_runoff_ratio(q_series, area_series, pr_series, freq="YS"):
    # 1 years of daily data
    q = np.ones(365, dtype=float) * 10
    pr = np.ones(365, dtype=float) * 20

    # 30 days with low flows, ratio should stay the same
    q[300:330] = 5
    pr[270:300] = 10
    a = 1000
    a = area_series(a)

    q = q_series(q)
    new_start = "2000-07-01"
    q_shifted = q.assign_coords(time=pd.date_range(new_start, periods=q.sizes["time"], freq="D"))

    pr = pr_series(pr, units="mm/hr")
    out = land.runoff_ratio(q_shifted, a, pr, freq="YS")

    assert out.attrs["units"] == "1"
    assert isinstance(out, xr.DataArray)


def test_base_flow_index_seasonal_ratio(q_series):
    # FIXME Results in AttributeError: 'DataArray' object has no attribute 'time'
    #  multiple timestamps to present : seasonal and yearly regarding the ratio
    a = np.ones(364)
    q = q_series(a)
    print(q)
    print(q.dims)
    print(q.coords)

    out = land.base_flow_index_seasonal_ratio(q)

    assert out.attrs["units"] == "1"
    assert isinstance(out, xr.DataArray)


def test_lag_snowpack_flow_peaks(swe_series, q_series):
    a = np.zeros(365)

    # Year 1: 1 day of SWE = 20 mm
    a[50:51] = 20
    # Year 2: 1 day of SWE = 5 mm
    a[300:301] = 5

    # Create a daily time index
    swe = swe_series(a)

    b = np.zeros(365)
    # Year 1: 35 days of high flows directly after max swe
    b[50:85] = 20
    # Year 2: 35 days of high flows 10 days after max swe
    b[310:345] = 5

    # Create a daily time index
    q = q_series(b)

    out = land.lag_snowpack_flow_peaks(swe, q)
    assert out.attrs["units"] == "days"
    assert isinstance(out, xr.DataArray)


def test_ss(q_series):
    # FIXME Results in AttributeError: 'DataArray' object has no attribute 'time'.
    # multiple timestamps : seasonal and yearly.

    # 5 years of increasing data with slope of 1
    q = np.arange(1, 1826)

    # 5 years of increasing data with slope of 2
    qsim = np.arange(1, 1826) * 2

    # Create a daily time index
    q = q_series(q)
    qsim = q_series(qsim)
    print(q)
    print(q.dims)
    print(q.coords)

    out = land.sen_slope(q, qsim)

    assert out.attrs["units"] == "1"
    assert isinstance(out, xr.DataArray)

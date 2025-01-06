"""Tests for indicators in `land` realm."""

from __future__ import annotations

import numpy as np
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

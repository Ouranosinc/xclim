"""Tests for indicators in `land` realm."""
from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

import xclim.core.utils
from xclim import land


def test_base_flow_index(ndq_series):
    out = land.base_flow_index(ndq_series, freq="YS")
    assert out.attrs["units"] == ""
    assert isinstance(out, xr.DataArray)


def test_rb_flashiness_index(ndq_series):
    out = land.base_flow_index(ndq_series, freq="YS")
    assert out.attrs["units"] == ""
    assert isinstance(out, xr.DataArray)


class Test_FA:
    def test_simple(self, ndq_series):
        out = land.freq_analysis(
            ndq_series, mode="max", t=[2, 5], dist="gamma", season="DJF"
        )
        assert out.description in [
            "Streamflow frequency analysis for the maximal winter 1-day flow "
            "estimated using the gamma distribution."
        ]
        assert out.name == "q1maxwinter"
        assert out.shape == (2, 2, 3)  # nrt, nx, ny
        np.testing.assert_array_equal(out.isnull(), False)

    def test_no_indexer(self, ndq_series):
        out = land.freq_analysis(ndq_series, mode="max", t=[2, 5], dist="gamma")
        assert out.description in [
            "Streamflow frequency analysis for the maximal annual 1-day flow "
            "estimated using the gamma distribution."
        ]
        assert out.name == "q1maxannual"
        assert out.shape == (2, 2, 3)  # nrt, nx, ny
        np.testing.assert_array_equal(out.isnull(), False)

    def test_q27(self, ndq_series):
        out = land.freq_analysis(ndq_series, mode="max", t=2, dist="gamma", window=7)
        assert out.shape == (1, 2, 3)

    def test_empty(self, ndq_series):
        q = ndq_series.copy()
        q[:, 0, 0] = np.nan
        out = land.freq_analysis(
            q, mode="max", t=2, dist="genextreme", window=6, freq="YS"
        )
        assert np.isnan(out.values[:, 0, 0]).all()

    def test_wrong_variable(self, pr_series):
        with pytest.raises(xclim.core.utils.ValidationError):
            land.freq_analysis(
                pr_series(np.random.rand(100)), mode="max", t=2, dist="gamma"
            )


class TestStats:
    def test_simple(self, ndq_series):
        out = land.stats(ndq_series, freq="YS", op="min", season="MAM")
        assert out.attrs["units"] == "m^3 s-1"

    def test_missing(self, ndq_series):
        a = ndq_series
        a = ndq_series.where(~((a.time.dt.dayofyear == 5) * (a.time.dt.year == 1902)))
        assert a.shape == (5000, 2, 3)
        out = land.stats(a, op="max", month=1)

        np.testing.assert_array_equal(out.sel(time="1900").isnull(), False)
        np.testing.assert_array_equal(out.sel(time="1902").isnull(), True)


def test_qdoy_max(ndq_series, q_series):
    out = land.doy_qmax(ndq_series, freq="YS", season="JJA")
    assert out.attrs["units"] == ""

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


def test_winter_storm(snd_series):
    a = np.zeros(366)
    a[10:20] = np.arange(10)

    snd = snd_series(a)
    out = land.winter_storm(snd, thresh="50 cm")
    np.testing.assert_array_equal(out, [9, np.nan])

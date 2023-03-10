"""Tests for indicators in `land` realm."""
from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from xclim import land
from xclim.core.utils import ValidationError


def test_base_flow_index(ndq_series):
    out = land.base_flow_index(ndq_series, freq="YS")
    assert out.attrs["units"] == ""
    assert isinstance(out, xr.DataArray)


def test_rb_flashiness_index(ndq_series):
    out = land.base_flow_index(ndq_series, freq="YS")
    assert out.attrs["units"] == ""
    assert isinstance(out, xr.DataArray)


class TestFreqAnalysisDeprecated:
    def test_simple(self, ndq_series):
        with pytest.warns(DeprecationWarning):
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

    def test_wrong_variable(self, pr_series):
        with pytest.raises(ValidationError):
            with pytest.warns(DeprecationWarning):
                land.freq_analysis(
                    pr_series(np.random.rand(100)), mode="max", t=2, dist="gamma"
                )


class TestStatsDeprecated:
    def test_simple(self, ndq_series):
        with pytest.warns(DeprecationWarning):
            out = land.stats(ndq_series, freq="YS", op="min", season="MAM")
        assert out.attrs["units"] == "m^3 s-1"


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

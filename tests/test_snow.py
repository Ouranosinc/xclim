from __future__ import annotations

import numpy as np
import pytest

from xclim import land
from xclim.core.utils import ValidationError


class TestSnowDepth:
    def test_simple(self, snd_series):
        snd = snd_series(np.ones(110), start="2001-01-01")
        out = land.snow_depth(snd, freq="M")
        assert out.units == "cm"
        np.testing.assert_array_equal(out, [100, 100, 100, np.nan])


class TestSnowDepthCoverDuration:
    def test_simple(self, snd_series):
        snd = snd_series(np.ones(110), start="2001-01-01")

        out = land.snd_season_length(snd, freq="M")
        assert out.units == "days"
        np.testing.assert_array_equal(out, [31, 28, 31, np.nan])


class TestSnowWaterCoverDuration:
    def test_simple(self, snw_series):
        snw = snw_series(np.ones(110) * 1000, start="2001-01-01")
        out = land.snw_season_length(snw, freq="M")
        assert out.units == "days"
        np.testing.assert_array_equal(out, [31, 28, 31, np.nan])


class TestContinuousSnowDepthCoverStartEnd:
    def test_simple(self, snd_series):
        a = np.zeros(365)
        # snow depth
        a[100:200] = 0.03
        snd = snd_series(a, start="2001-07-01")
        snd = snd.expand_dims(lat=[0, 1, 2])

        out = land.snd_season_start(snd)
        assert out.units == ""
        np.testing.assert_array_equal(out.isel(lat=0), snd.time.dt.dayofyear[100])

        out = land.snd_season_end(snd)
        assert out.units == ""
        np.testing.assert_array_equal(out.isel(lat=0), snd.time.dt.dayofyear[200])


class TestContinuousSnowWaterCoverStartEnd:
    def test_simple(self, snw_series):
        a = np.zeros(365)
        # snow amount
        a[100:200] = 0.03 * 1000
        snw = snw_series(a, start="2001-07-01")
        snw = snw.expand_dims(lat=[0, 1, 2])

        out = land.snw_season_start(snw)
        assert out.units == ""
        np.testing.assert_array_equal(out.isel(lat=0), snw.time.dt.dayofyear[100])

        out = land.snw_season_end(snw)
        assert out.units == ""
        np.testing.assert_array_equal(out.isel(lat=0), snw.time.dt.dayofyear[200])


class TestSndMaxDoy:
    def test_simple(self, snd_series):
        a = np.zeros(365)
        a[200] = 1
        snd = snd_series(a, start="2001-07-01")
        out = land.snd_max_doy(snd, freq="AS-JUL")
        np.testing.assert_array_equal(out, snd.time.dt.dayofyear[200])

    def test_units(self, tas_series):
        """Check that unit declaration works."""
        tas = tas_series(np.random.rand(365), start="1999-07-01")
        with pytest.raises(ValidationError):
            land.snd_max_doy(tas)

    def test_no_snow(self, atmosds):
        # Put 0 on one row.
        snd = atmosds.snd.where(atmosds.location != "Victoria", 0)
        out = land.snd_max_doy(snd)
        np.testing.assert_array_equal(out.isel(time=1), [16, 13, 91, 29, np.NaN])


class TestSnwMax:
    def test_simple(self, snw_series):
        a = np.zeros(366)
        a[20] = 1
        snw = snw_series(a, start="2001-01-01")
        out = land.snw_max(snw=snw, freq="YS")
        np.testing.assert_array_equal(out, [1, np.nan])


class TestSnwMaxDoy:
    def test_simple(self, snw_series):
        a = np.zeros(366)
        a[20] = 1
        snw = snw_series(a, start="2001-01-01")
        out = land.snw_max_doy(snw, freq="YS")
        np.testing.assert_array_equal(out, [21, np.nan])

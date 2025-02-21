from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from xclim import land
from xclim.core import ValidationError


class TestSnowDepth:
    def test_simple(self, snd_series):
        snd = snd_series(np.ones(110), start="2001-01-01")
        out = land.snow_depth(snd, freq="ME")
        assert out.units == "cm"
        np.testing.assert_array_equal(out, [100, 100, 100, np.nan])


class TestSnowDepthCoverDuration:
    def test_simple(self, snd_series):
        snd = snd_series(np.ones(110), start="2001-01-01")

        out = land.snd_days_above(snd, freq="ME")
        assert out.units == "days"
        np.testing.assert_array_equal(out, [31, 28, 31, np.nan])


class TestSnowWaterCoverDuration:
    @pytest.mark.parametrize("factor,exp", ([1000, [31, 28, 31, np.nan]], [0, [0, 0, 0, np.nan]]))
    def test_simple(self, snw_series, factor, exp):
        snw = snw_series(np.ones(110) * factor, start="2001-01-01")
        out = land.snw_days_above(snw, freq="ME")
        assert out.units == "days"
        np.testing.assert_array_equal(out, exp)


class TestContinuousSnowDepthSeason:
    def test_simple(self, snd_series):
        a = np.zeros(365)
        # snow depth
        a[100:200] = 0.03
        a[150:160] = 0
        snd = snd_series(a, start="2001-07-01")
        snd = snd.expand_dims(lat=[0, 1, 2])

        out = land.snd_season_start(snd)
        assert out.units == "1"

        np.testing.assert_array_equal(out.isel(lat=0), snd.time.dt.dayofyear[100])

        out = land.snd_season_end(snd)
        assert out.units == "1"

        np.testing.assert_array_equal(out.isel(lat=0), snd.time.dt.dayofyear[200])

        out = land.snd_season_length(snd)
        assert out.units == "days"
        np.testing.assert_array_equal(out.isel(lat=0), 100)


class TestContinuousSnowWaterSeason:
    def test_simple(self, snw_series):
        a = np.zeros(365)
        # snow amount
        a[100:200] = 0.03 * 1000
        a[150:160] = 0
        snw = snw_series(a, start="2001-07-01")
        snw = snw.expand_dims(lat=[0, 1, 2])

        out = land.snw_season_start(snw)
        assert out.units == "1"

        np.testing.assert_array_equal(out.isel(lat=0), snw.time.dt.dayofyear[100])

        out = land.snw_season_end(snw)
        assert out.units == "1"

        np.testing.assert_array_equal(out.isel(lat=0), snw.time.dt.dayofyear[200])

        out = land.snw_season_length(snw)
        assert out.units == "days"
        np.testing.assert_array_equal(out.isel(lat=0), 100)


class TestSndMaxDoy:
    def test_simple(self, snd_series):
        a = np.zeros(365)
        a[200] = 1
        snd = snd_series(a, start="2001-07-01")
        out = land.snd_max_doy(snd, freq="YS-JUL")
        np.testing.assert_array_equal(out, snd.time.dt.dayofyear[200])

    def test_units(self, tas_series, random):
        """Check that unit declaration works."""
        tas = tas_series(random.random(365), start="1999-07-01")
        with pytest.raises(ValidationError):
            land.snd_max_doy(tas)

    def test_no_snow(self, atmosds):
        # Put 0 on one row.
        snd = atmosds.snd.where(atmosds.location != "Victoria", 0)
        out = land.snd_max_doy(snd)
        np.testing.assert_array_equal(out.isel(time=1), [16, 13, 91, 29, np.nan])


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


class TestHolidaySnowIndicators:
    def test_xmas_days_simple(self, nimbus):
        ds = xr.open_dataset(nimbus.fetch("cmip6/snw_day_CanESM5_historical_r1i1p1f1_gn_19910101-20101231.nc"))
        snd = land.snw_to_snd(ds.snw)

        out = land.holiday_snow_days(snd)

        assert out.units == "days"
        assert out.long_name == "Number of holiday days with snow"
        np.testing.assert_array_equal(
            out.sum(dim="time"),
            [
                [7.0, 5.0, 2.0, 0.0, 0.0],
                [14.0, 13.0, 9.0, 6.0, 2.0],
                [18.0, 19.0, 19.0, 18.0, 13.0],
                [20.0, 20.0, 20.0, 20.0, 20.0],
                [20.0, 20.0, 20.0, 20.0, 20.0],
                [20.0, 20.0, 20.0, 20.0, 20.0],
            ],
        )

    def test_perfect_xmas_days_simple(self, nimbus):
        ds_snw = xr.open_dataset(nimbus.fetch("cmip6/snw_day_CanESM5_historical_r1i1p1f1_gn_19910101-20101231.nc"))
        ds_prsn = xr.open_dataset(nimbus.fetch("cmip6/prsn_day_CanESM5_historical_r1i1p1f1_gn_19910101-20101231.nc"))

        snd = land.snw_to_snd(ds_snw.snw)
        prsn = ds_prsn.prsn

        out = land.holiday_snow_and_snowfall_days(snd, prsn)

        assert out.units == "days"
        assert out.long_name == "Number of holiday days with snow and snowfall"
        np.testing.assert_array_equal(
            out.sum(dim="time"),
            [
                [3.0, 0.0, 0.0, 0.0, 0.0],
                [5.0, 2.0, 1.0, 1.0, 1.0],
                [6.0, 5.0, 4.0, 4.0, 5.0],
                [7.0, 11.0, 12.0, 9.0, 6.0],
                [10.0, 8.0, 12.0, 10.0, 8.0],
                [9.0, 11.0, 10.0, 7.0, 9.0],
            ],
        )

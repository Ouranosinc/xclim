#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Tests for `xclim` package.
#
# We want to tests multiple things here:
#  - that data results are correct
#  - that metadata is correct and complete
#  - that missing data are handled appropriately
#  - that various calendar formats and supported
#  - that non-valid input frequencies or holes in the time series are detected
#
#
# For correctness, I think it would be useful to use a small dataset and run the original ICCLIM indicators on it,
# saving the results in a reference netcdf dataset. We could then compare the hailstorm output to this reference as
# a first line of defense.
import calendar
import os

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from xclim import indices as xci
from xclim.core.calendar import percentile_doy
from xclim.core.units import ValidationError, convert_units_to
from xclim.testing import open_dataset

K2C = 273.15


# TODO: Obey the line below:
# PLEASE MAINTAIN ALPHABETICAL ORDER


class TestMaxNDayPrecipitationAmount:

    # test 2 day max precip
    def test_single_max(self, pr_series):
        a = pr_series(np.array([3, 4, 20, 20, 0, 6, 9, 25, 0, 0]))
        rxnday = xci.max_n_day_precipitation_amount(a, 2)
        assert rxnday == 40 * 3600 * 24
        assert rxnday.time.dt.year == 2000

    # test whether sum over entire length is resolved
    def test_sumlength_max(self, pr_series):
        a = pr_series(np.array([3, 4, 20, 20, 0, 6, 9, 25, 0, 0]))
        rxnday = xci.max_n_day_precipitation_amount(a, len(a))
        assert rxnday == a.sum("time") * 3600 * 24
        assert rxnday.time.dt.year == 2000

    # test whether non-unique maxes are resolved
    def test_multi_max(self, pr_series):
        a = pr_series(np.array([3, 4, 20, 20, 0, 6, 15, 25, 0, 0]))
        rxnday = xci.max_n_day_precipitation_amount(a, 2)
        assert rxnday == 40 * 3600 * 24
        assert len(rxnday) == 1
        assert rxnday.time.dt.year == 2000


class TestMax1DayPrecipitationAmount:
    @staticmethod
    def time_series(values):
        coords = pd.date_range(
            "7/1/2000", periods=len(values), freq=pd.DateOffset(days=1)
        )
        return xr.DataArray(
            values,
            coords=[coords],
            dims="time",
            attrs={
                "standard_name": "precipitation_flux",
                "cell_methods": "time: sum (interval: 1 day)",
                "units": "mm/day",
            },
        )

    # test max precip
    def test_single_max(self):
        a = self.time_series(np.array([3, 4, 20, 0, 0]))
        rx1day = xci.max_1day_precipitation_amount(a)
        assert rx1day == 20
        assert rx1day.time.dt.year == 2000

    # test whether repeated maxes are resolved
    def test_multi_max(self):
        a = self.time_series(np.array([20, 4, 20, 20, 0]))
        rx1day = xci.max_1day_precipitation_amount(a)
        assert rx1day == 20
        assert rx1day.time.dt.year == 2000
        assert len(rx1day) == 1

    # test whether uniform maxes are resolved
    def test_uniform_max(self):
        a = self.time_series(np.array([20, 20, 20, 20, 20]))
        rx1day = xci.max_1day_precipitation_amount(a)
        assert rx1day == 20
        assert rx1day.time.dt.year == 2000
        assert len(rx1day) == 1


class TestColdSpellDurationIndex:
    def test_simple(self, tasmin_series):
        i = 3650
        A = 10.0
        tn = (
            np.zeros(i)
            + A * np.sin(np.arange(i) / 365.0 * 2 * np.pi)
            + 0.1 * np.random.rand(i)
        )
        tn[10:20] -= 2
        tn = tasmin_series(tn)
        tn10 = percentile_doy(tn, per=10).sel(percentiles=10)

        out = xci.cold_spell_duration_index(tn, tn10, freq="YS")
        assert out[0] == 10
        assert out.units == "d"


class TestColdSpellDays:
    def test_simple(self, tas_series):
        a = np.zeros(365)
        a[10:20] -= 15  # 10 days
        a[40:43] -= 50  # too short -> 0
        a[80:100] -= 30  # at the end and beginning
        da = tas_series(a + K2C)

        out = xci.cold_spell_days(da, thresh="-10. C", freq="M")
        np.testing.assert_array_equal(out, [10, 0, 12, 8, 0, 0, 0, 0, 0, 0, 0, 0])
        assert out.units == "d"


class TestColdSpellFreq:
    def test_simple(self, tas_series):
        a = np.zeros(365)
        a[10:20] -= 15  # 10 days
        a[40:43] -= 50  # too short -> 0
        a[80:86] -= 30
        a[95:101] -= 30
        da = tas_series(a + K2C, start="1971-01-01")

        out = xci.cold_spell_frequency(da, thresh="-10. C", freq="M")
        np.testing.assert_array_equal(out, [1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
        assert out.units == ""

        out = xci.cold_spell_frequency(da, thresh="-10. C", freq="YS")
        np.testing.assert_array_equal(out, 3)
        assert out.units == ""


class TestMaxConsecutiveFrostDays:
    def test_one_freeze_day(self, tasmin_series):
        a = tasmin_series(np.array([3, 4, 5, -1, 3]) + K2C)
        cfd = xci.maximum_consecutive_frost_days(a)
        assert cfd == 1
        assert cfd.time.dt.year == 2000

    def test_no_freeze(self, tasmin_series):
        a = tasmin_series(np.array([3, 4, 5, 1, 3]) + K2C)
        cfd = xci.maximum_consecutive_frost_days(a)
        assert cfd == 0

    def test_all_year_freeze(self, tasmin_series):
        a = tasmin_series(np.zeros(365) - 10 + K2C)
        cfd = xci.maximum_consecutive_frost_days(a)
        assert cfd == 365


class TestMaximumConsecutiveFrostFreeDays:
    def test_one_freeze_day(self, tasmin_series):
        a = tasmin_series(np.array([3, 4, 5, -1, 3]) + K2C)
        ffd = xci.maximum_consecutive_frost_free_days(a)
        assert ffd == 3
        assert ffd.time.dt.year == 2000

    def test_two_freeze_days_with_threshold(self, tasmin_series):
        a = tasmin_series(np.array([3, 4, 5, -0.8, -2, 3]) + K2C)
        ffd = xci.maximum_consecutive_frost_free_days(a, thresh="-1 degC")
        assert ffd == 4

    def test_no_freeze(self, tasmin_series):
        a = tasmin_series(np.array([3, 4, 5, 1, 3]) + K2C)
        ffd = xci.maximum_consecutive_frost_free_days(a)
        assert ffd == 5

    def test_all_year_freeze(self, tasmin_series):
        a = tasmin_series(np.zeros(365) - 10 + K2C)
        ffd = xci.maximum_consecutive_frost_free_days(a)
        assert np.all(ffd) == 0


class TestCoolingDegreeDays:
    def test_no_cdd(self, tas_series):
        a = tas_series(np.array([10, 15, -5, 18]) + K2C)
        cdd = xci.cooling_degree_days(a)
        assert cdd == 0
        assert cdd.units == "K d"

    def test_cdd(self, tas_series):
        a = tas_series(np.array([20, 25, -15, 19]) + K2C)
        cdd = xci.cooling_degree_days(a)
        assert cdd == 10


class TestDailyFreezeThawCycles:
    def test_simple(self, tasmin_series, tasmax_series):
        mn = np.zeros(365)
        mx = np.zeros(365)

        # 5 days in 1st month
        mn[10:20] -= 1
        mx[10:15] += 1

        # 1 day in 2nd month
        mn[40:44] += [1, 1, -1, -1]
        mx[40:44] += [1, -1, 1, -1]

        mn = tasmin_series(mn + K2C)
        mx = tasmax_series(mx + K2C)
        out = xci.daily_freezethaw_cycles(mn, mx, freq="M")
        np.testing.assert_array_equal(out[:2], [5, 1])
        np.testing.assert_array_equal(out[2:], 0)

    def test_zeroed_thresholds(self, tasmin_series, tasmax_series):
        mn = np.zeros(365)
        mx = np.zeros(365)

        # 5 days in 1st month
        mn[10:20] -= 1
        mx[10:15] += 1

        # 1 day in 2nd month
        mn[40:44] += [1, 1, -1, -1]
        mx[40:44] += [1, -1, 1, -1]

        mn = tasmin_series(mn + K2C)
        mx = tasmax_series(mx + K2C)
        out = xci.daily_freezethaw_cycles(
            mn, mx, thresh_tasmax="0 degC", thresh_tasmin="0 degC", freq="M"
        )
        np.testing.assert_array_equal(out[:2], [5, 1])
        np.testing.assert_array_equal(out[2:], 0)


class TestDailyPrIntensity:
    def test_simple(self, pr_series):
        pr = pr_series(np.zeros(365))
        pr[3:8] += [0.5, 1, 2, 3, 4]
        out = xci.daily_pr_intensity(pr, thresh="1 kg/m**2/s")
        np.testing.assert_array_equal(out[0], 2.5 * 3600 * 24)

    def test_mm(self, pr_series):
        pr = pr_series(np.zeros(365))
        pr[3:8] += [0.5, 1, 2, 3, 4]
        pr.attrs["units"] = "mm/d"
        out = xci.daily_pr_intensity(pr, thresh="1 mm/day")
        np.testing.assert_array_almost_equal(out[0], 2.5)


class TestMaxPrIntensity:
    # Hourly indicator
    def test_simple(self, pr_hr_series):
        pr = pr_hr_series(np.zeros(24 * 36))
        pr[10:22] += np.arange(12)  # kg / m2 / s

        out = xci.max_pr_intensity(pr, window=1, freq="Y")
        np.testing.assert_array_almost_equal(out[0], 11)

        out = xci.max_pr_intensity(pr, window=12, freq="Y")
        np.testing.assert_array_almost_equal(out[0], 5.5)

        pr.attrs["units"] = "mm"
        with pytest.raises(ValidationError):
            xci.max_pr_intensity(pr, window=1, freq="Y")


class TestLastSpringFrost:
    def test_simple(self, tas_series):
        a = np.zeros(365)
        a[180:270] = 303.15
        tas = tas_series(a, start="2000/1/1")

        lsf = xci.last_spring_frost(tas)
        assert lsf == 180


class TestFirstDayBelow:
    def test_simple(self, tas_series):
        a = np.zeros(365)
        a[180:270] = 303.15
        tas = tas_series(a, start="2000/1/1")

        fdb = xci.first_day_below(tas)
        assert fdb == 271

        a[:] = 303.15
        tas = tas_series(a, start="2000/1/1")

        fdb = xci.first_day_below(tas)
        assert np.isnan(fdb)


class TestFirstDayAbove:
    def test_simple(self, tas_series):
        a = np.zeros(365) + 307
        a[180:270] = 270
        tas = tas_series(a, start="2000/1/1")

        fdb = xci.first_day_above(tas)
        assert fdb == 1

        fdb = xci.first_day_above(tas, after_date="07-01")
        assert fdb == 271

        a[:] = 270
        tas = tas_series(a, start="2000/1/1")

        fdb = xci.first_day_above(tas)
        assert np.isnan(fdb)


class TestDaysOverPrecipThresh:
    def test_simple(self, pr_series, per_doy):
        a = np.zeros(365)
        a[:8] = np.arange(8)
        pr = pr_series(a, start="1/1/2000")

        per = per_doy(np.zeros(366))
        per[5:] = 5

        out = xci.days_over_precip_thresh(pr, per, thresh="2 kg/m**2/s")
        np.testing.assert_array_almost_equal(out[0], 4)

        out = xci.fraction_over_precip_thresh(pr, per, thresh="2 kg/m**2/s")
        np.testing.assert_array_almost_equal(
            out[0], (3 + 4 + 6 + 7) / (3 + 4 + 5 + 6 + 7)
        )

    def test_quantile(self, pr_series):
        a = np.zeros(365)
        a[:8] = np.arange(8)
        pr = pr_series(a, start="1/1/2000")

        # Create synthetic percentile
        pr0 = pr_series(np.ones(365) * 5, start="1/1/2000")
        per = pr0.quantile(0.5, dim="time", keep_attrs=True)
        per.attrs["units"] = "kg m-2 s-1"  # This won't be needed with xarray 0.13

        out = xci.days_over_precip_thresh(pr, per, thresh="2 kg/m**2/s")
        np.testing.assert_array_almost_equal(
            out[0], 2
        )  # Only days 6 and 7 meet criteria.

    def test_nd(self, pr_ndseries):
        pr = pr_ndseries(np.ones((300, 2, 3)))
        pr0 = pr_ndseries(np.zeros((300, 2, 3)))
        per = pr0.quantile(0.5, dim="time", keep_attrs=True)
        per.attrs["units"] = "kg m-2 s-1"  # This won't be needed with xarray 0.13

        out = xci.days_over_precip_thresh(pr, per, thresh="0.5 kg/m**2/s")
        np.testing.assert_array_almost_equal(out, 300)


class TestFreshetStart:
    def test_simple(self, tas_series):
        tg = np.zeros(365) - 1
        w = 5

        i = 10
        tg[i : i + w - 1] += 6  # too short

        i = 20
        tg[i : i + w] += 6  # ok

        i = 30
        tg[i : i + w + 1] += 6  # Second valid condition, should be ignored.

        tg = tas_series(tg + K2C, start="1/1/2000")
        out = xci.freshet_start(tg, window=w)
        assert out[0] == tg.indexes["time"][20].dayofyear

    def test_no_start(self, tas_series):
        tg = np.zeros(365) - 1
        tg = tas_series(tg, start="1/1/2000")
        out = xci.freshet_start(tg)
        np.testing.assert_equal(out, [np.nan])


class TestGrowingDegreeDays:
    def test_simple(self, tas_series):
        a = np.zeros(365)
        a[0] = 5  # default thresh at 4
        da = tas_series(a + K2C)
        assert xci.growing_degree_days(da)[0] == 1


class TestGrowingSeasonEnd:
    @pytest.mark.parametrize(
        "d1,d2,mid_date,expected",
        [
            ("1950-01-01", "1951-01-01", "07-01", np.nan),  # No growing season
            ("2000-01-01", "2000-12-31", "07-01", 365),  # All year growing season
            ("2000-07-10", "2001-01-01", "07-01", np.nan),  # End happens before start
            ("2000-06-15", "2000-07-15", "07-01", 198),  # Normal case
            ("2000-06-15", "2000-07-25", "07-15", 208),  # PCC Case
            ("2000-06-15", "2000-07-15", "10-01", 275),  # Late mid_date
            ("2000-06-15", "2000-07-15", "01-10", np.nan),  # Early mid_date
        ],
    )
    def test_varying_mid_dates(self, tas_series, d1, d2, mid_date, expected):
        # generate a year of data
        tas = tas_series(np.zeros(365), start="2000/1/1")
        warm_period = tas.sel(time=slice(d1, d2))
        tas = tas.where(~tas.time.isin(warm_period.time), 280)
        gs_end = xci.growing_season_end(tas, mid_date=mid_date)
        np.testing.assert_array_equal(gs_end, expected)


class TestGrowingSeasonLength:
    @pytest.mark.parametrize(
        "d1,d2,expected",
        [
            ("1950-01-01", "1951-01-01", 0),  # No growing season
            ("2000-01-01", "2000-12-31", 365),  # All year growing season
            ("2000-07-10", "2001-01-01", np.nan),  # End happens before start
            ("2000-06-15", "2001-01-01", 199),  # No end
            ("2000-06-15", "2000-07-15", 31),  # Normal case
        ],
    )
    def test_simple(self, tas_series, d1, d2, expected):
        # test for different growing length

        # generate a year of data
        tas = tas_series(np.zeros(365), start="2000/1/1")
        warm_period = tas.sel(time=slice(d1, d2))
        tas = tas.where(~tas.time.isin(warm_period.time), 280)
        gsl = xci.growing_season_length(tas)
        np.testing.assert_array_equal(gsl, expected)

    def test_southhemisphere(self, tas_series):
        tas = tas_series(np.zeros(2 * 365), start="2000/1/1")
        warm_period = tas.sel(time=slice("2000-11-01", "2001-03-01"))
        tas = tas.where(~tas.time.isin(warm_period.time), 280)
        gsl = xci.growing_season_length(tas, mid_date="01-01", freq="AS-Jul")
        np.testing.assert_array_equal(gsl.sel(time="2000-07-01"), 121)


class TestFrostSeasonLength:
    @pytest.mark.parametrize(
        "d1,d2,expected",
        [
            ("1950-01-01", "1951-01-01", 0),  # No frost season
            ("2000-01-01", "2000-12-31", 365),  # All year frost season
            ("2000-06-15", "2001-01-01", 199),  # No end
            ("2000-06-15", "2000-07-15", 31),  # Normal case
        ],
    )
    def test_simple(self, tas_series, d1, d2, expected):
        # test for different growing length

        # generate a year of data
        tas = tas_series(np.zeros(365) + 300, start="2000/1/1")
        cold_period = tas.sel(time=slice(d1, d2))
        tas = tas.where(~tas.time.isin(cold_period.time), 270)
        fsl = xci.frost_season_length(tas, freq="YS", mid_date="07-01")
        np.testing.assert_array_equal(fsl, expected)

    def test_northhemisphere(self, tas_series):
        tas = tas_series(np.zeros(2 * 365) + 300, start="2000/1/1")
        cold_period = tas.sel(time=slice("2000-11-01", "2001-03-01"))
        tas = tas.where(~tas.time.isin(cold_period.time), 270)
        fsl = xci.frost_season_length(tas)
        np.testing.assert_array_equal(fsl.sel(time="2000-07-01"), 121)


class TestHeatingDegreeDays:
    def test_simple(self, tas_series):
        a = np.zeros(365) + 17
        a[:7] += [-3, -2, -1, 0, 1, 2, 3]
        da = tas_series(a + K2C)
        out = xci.heating_degree_days(da)
        np.testing.assert_array_equal(out[:1], 6)
        np.testing.assert_array_equal(out[1:], 0)


class TestHeatWaveIndex:
    def test_simple(self, tasmax_series):
        a = np.zeros(365)
        a[10:20] += 30  # 10 days
        a[40:43] += 50  # too short -> 0
        a[80:100] += 30  # at the end and beginning
        da = tasmax_series(a + K2C)

        out = xci.heat_wave_index(da, thresh="25 C", freq="M")
        np.testing.assert_array_equal(out, [10, 0, 12, 8, 0, 0, 0, 0, 0, 0, 0, 0])


class TestHeatWaveFrequency:
    @pytest.mark.parametrize(
        "thresh_tasmin,thresh_tasmax,window,expected",
        [
            ("22 C", "30 C", 3, 2),  # Some HW
            ("22 C", "30 C", 4, 1),  # No HW
            ("10 C", "10 C", 3, 1),  # One long HW
            ("40 C", "40 C", 3, 0),  # Windowed
        ],
    )
    def test_1d(
        self,
        tasmax_series,
        tasmin_series,
        thresh_tasmin,
        thresh_tasmax,
        window,
        expected,
    ):
        tn = tasmin_series(np.asarray([20, 23, 23, 23, 23, 22, 23, 23, 23, 23]) + K2C)
        tx = tasmax_series(np.asarray([29, 31, 31, 31, 29, 31, 31, 31, 31, 31]) + K2C)

        hwf = xci.heat_wave_frequency(
            tn,
            tx,
            thresh_tasmin=thresh_tasmin,
            thresh_tasmax=thresh_tasmax,
            window=window,
        )
        np.testing.assert_allclose(hwf.values, expected)


class TestHeatWaveMaxLength:
    @pytest.mark.parametrize(
        "thresh_tasmin,thresh_tasmax,window,expected",
        [
            ("22 C", "30 C", 3, 4),  # Some HW
            ("10 C", "10 C", 3, 10),  # One long HW
            ("40 C", "40 C", 3, 0),  # No HW
            ("22 C", "30 C", 5, 0),  # Windowed
        ],
    )
    def test_1d(
        self,
        tasmax_series,
        tasmin_series,
        thresh_tasmin,
        thresh_tasmax,
        window,
        expected,
    ):
        tn = tasmin_series(np.asarray([20, 23, 23, 23, 23, 22, 23, 23, 23, 23]) + K2C)
        tx = tasmax_series(np.asarray([29, 31, 31, 31, 29, 31, 31, 31, 31, 31]) + K2C)

        hwml = xci.heat_wave_max_length(
            tn,
            tx,
            thresh_tasmin=thresh_tasmin,
            thresh_tasmax=thresh_tasmax,
            window=window,
        )
        np.testing.assert_allclose(hwml.values, expected)


class TestHeatWaveTotalLength:
    @pytest.mark.parametrize(
        "thresh_tasmin,thresh_tasmax,window,expected",
        [
            ("22 C", "30 C", 3, 7),  # Some HW
            ("10 C", "10 C", 3, 10),  # One long HW
            ("40 C", "40 C", 3, 0),  # No HW
            ("22 C", "30 C", 5, 0),  # Windowed
        ],
    )
    def test_1d(
        self,
        tasmax_series,
        tasmin_series,
        thresh_tasmin,
        thresh_tasmax,
        window,
        expected,
    ):
        tn = tasmin_series(np.asarray([20, 23, 23, 23, 23, 22, 23, 23, 23, 23]) + K2C)
        tx = tasmax_series(np.asarray([29, 31, 31, 31, 29, 31, 31, 31, 31, 31]) + K2C)

        # some hw
        hwml = xci.heat_wave_total_length(
            tn,
            tx,
            thresh_tasmin=thresh_tasmin,
            thresh_tasmax=thresh_tasmax,
            window=window,
        )
        np.testing.assert_allclose(hwml.values, expected)


class TestHotSpellFrequency:
    @pytest.mark.parametrize(
        "thresh_tasmax,window,expected",
        [
            ("30 C", 3, 2),  # Some HS
            ("30 C", 4, 1),  # One long HS
            ("10 C", 3, 1),  # No HS
            ("40 C", 5, 0),  # Windowed
        ],
    )
    def test_1d(self, tasmax_series, thresh_tasmax, window, expected):
        tx = tasmax_series(np.asarray([29, 31, 31, 31, 29, 31, 31, 31, 31, 31]) + K2C)

        hsf = xci.hot_spell_frequency(tx, thresh_tasmax=thresh_tasmax, window=window)
        np.testing.assert_allclose(hsf.values, expected)


class TestHotSpellMaxLength:
    @pytest.mark.parametrize(
        "thresh_tasmax,window,expected",
        [
            ("30 C", 3, 5),  # Some HS
            ("10 C", 3, 10),  # One long HS
            ("40 C", 3, 0),  # No HS
            ("30 C", 5, 5),  # Windowed
        ],
    )
    def test_1d(self, tasmax_series, thresh_tasmax, window, expected):
        tx = tasmax_series(np.asarray([29, 31, 31, 31, 29, 31, 31, 31, 31, 31]) + K2C)

        hsml = xci.hot_spell_max_length(tx, thresh_tasmax=thresh_tasmax, window=window)
        np.testing.assert_allclose(hsml.values, expected)


class TestTnDaysBelow:
    def test_simple(self, tasmin_series):
        a = np.zeros(365)
        a[:6] -= [27, 28, 29, 30, 31, 32]  # 2 above 30
        mx = tasmin_series(a + K2C)

        out = xci.tn_days_below(mx, thresh="-10 C")
        np.testing.assert_array_equal(out[:1], [6])
        np.testing.assert_array_equal(out[1:], [0])
        out = xci.tn_days_below(mx, thresh="-30 C")
        np.testing.assert_array_equal(out[:1], [2])
        np.testing.assert_array_equal(out[1:], [0])


class TestTxDaysAbove:
    def test_simple(self, tasmax_series):
        a = np.zeros(365)
        a[:6] += [27, 28, 29, 30, 31, 32]  # 2 above 30
        mx = tasmax_series(a + K2C)

        out = xci.tx_days_above(mx, thresh="30 C")
        np.testing.assert_array_equal(out[:1], [2])
        np.testing.assert_array_equal(out[1:], [0])


class TestLiquidPrecipitationRatio:
    def test_simple(self, pr_series, tas_series):
        pr = np.zeros(100)
        pr[10:20] = 1
        pr = pr_series(pr)

        tas = np.zeros(100)
        tas[:14] -= 20
        tas[14:] += 10
        tas = tas_series(tas + K2C)

        out = xci.liquid_precip_ratio(pr, tas=tas, freq="M")
        np.testing.assert_almost_equal(out[:1], [0.6])


class TestMaximumConsecutiveDryDays:
    def test_simple(self, pr_series):
        a = np.zeros(365) + 10
        a[5:15] = 0
        pr = pr_series(a)
        out = xci.maximum_consecutive_dry_days(pr, freq="M")
        assert out[0] == 10

    def test_run_start_at_0(self, pr_series):
        a = np.zeros(365) + 10
        a[:10] = 0
        pr = pr_series(a)
        out = xci.maximum_consecutive_dry_days(pr, freq="M")
        assert out[0] == 10


class TestMaximumConsecutiveTxDays:
    def test_simple(self, tasmax_series):
        a = np.zeros(365) + 273.15
        a[5:15] += 30
        tx = tasmax_series(a, start="1/1/2010")
        out = xci.maximum_consecutive_tx_days(tx, thresh="25 C", freq="M")
        assert out[0] == 10
        np.testing.assert_array_almost_equal(out[1:], 0)


class TestPrecipAccumulation:
    # build test data for different calendar
    time_std = pd.date_range("2000-01-01", "2010-12-31", freq="D")
    da_std = xr.DataArray(
        time_std.year, coords=[time_std], dims="time", attrs={"units": "mm d-1"}
    )

    # calendar 365_day and 360_day not tested for now since xarray.resample
    # does not support other calendars than standard
    #
    # units = 'days since 2000-01-01 00:00'
    # time_365 = cftime.num2date(np.arange(0, 10 * 365), units, '365_day')
    # time_360 = cftime.num2date(np.arange(0, 10 * 360), units, '360_day')
    # da_365 = xr.DataArray(np.arange(time_365.size), coords=[time_365], dims='time')
    # da_360 = xr.DataArray(np.arange(time_360.size), coords=[time_360], dims='time')

    def test_simple(self, pr_series):
        pr = np.zeros(100)
        pr[5:10] = 1
        pr = pr_series(pr)

        out = xci.precip_accumulation(pr, freq="M")
        np.testing.assert_array_equal(out[0], 5 * 3600 * 24)

    def test_yearly(self):
        da_std = self.da_std
        out_std = xci.precip_accumulation(da_std)
        target = [
            (365 + calendar.isleap(y)) * y for y in np.unique(da_std.time.dt.year)
        ]
        np.testing.assert_allclose(out_std.values, target)

    def test_mixed_phases(self, pr_series, tas_series):
        pr = np.zeros(100)
        pr[5:20] = 1
        pr = pr_series(pr)

        tas = np.ones(100) * 280
        tas[5:10] = 270
        tas[10:15] = 268
        tas = tas_series(tas)

        outsn = xci.precip_accumulation(pr, tas=tas, phase="solid", freq="M")
        outsn2 = xci.precip_accumulation(
            pr, tas=tas, phase="solid", thresh="269 K", freq="M"
        )
        outrn = xci.precip_accumulation(pr, tas=tas, phase="liquid", freq="M")

        np.testing.assert_array_equal(outsn[0], 10 * 3600 * 24)
        np.testing.assert_array_equal(outsn2[0], 5 * 3600 * 24)
        np.testing.assert_array_equal(outrn[0], 5 * 3600 * 24)


class TestRainOnFrozenGround:
    def test_simple(self, tas_series, pr_series):
        tas = np.zeros(30) - 1
        pr = np.zeros(30)

        tas[10] += 5
        pr[10] += 2

        tas = tas_series(tas + K2C)
        pr = pr_series(pr / 3600 / 24)

        out = xci.rain_on_frozen_ground_days(pr, tas, freq="MS")
        assert out[0] == 1

    def test_small_rain(self, tas_series, pr_series):
        tas = np.zeros(30) - 1
        pr = np.zeros(30)

        tas[10] += 5
        pr[10] += 0.5

        tas = tas_series(tas + K2C)
        pr = pr_series(pr / 3600 / 24)

        out = xci.rain_on_frozen_ground_days(pr, tas, freq="MS")
        assert out[0] == 0

    def test_consecutive_rain(self, tas_series, pr_series):
        tas = np.zeros(30) - 1
        pr = np.zeros(30)

        tas[10:16] += 5
        pr[10:16] += 5

        tas = tas_series(tas + K2C)
        pr = pr_series(pr)

        out = xci.rain_on_frozen_ground_days(pr, tas, freq="MS")
        assert out[0] == 1


class TestTGXN10p:
    def test_tg10p_simple(self, tas_series):
        i = 366
        tas = np.array(range(i))
        tas = tas_series(tas, start="1/1/2000")
        t10 = percentile_doy(tas, per=10).sel(percentiles=10)

        # create cold spell in june
        tas[175:180] = 1

        out = xci.tg10p(tas, t10, freq="MS")
        assert out[0] == 1
        assert out[5] == 5

        with pytest.raises(AttributeError):
            out = xci.tg10p(tas, tas, freq="MS")

    def test_tx10p_simple(self, tasmax_series):
        i = 366
        tas = np.array(range(i))
        tas = tasmax_series(tas, start="1/1/2000")
        t10 = percentile_doy(tas, per=10).sel(percentiles=10)

        # create cold spell in june
        tas[175:180] = 1

        out = xci.tx10p(tas, t10, freq="MS")
        assert out[0] == 1
        assert out[5] == 5

    def test_tn10p_simple(self, tas_series):
        i = 366
        tas = np.array(range(i))
        tas = tas_series(tas, start="1/1/2000")
        t10 = percentile_doy(tas, per=10).sel(percentiles=10)

        # create cold spell in june
        tas[175:180] = 1

        out = xci.tn10p(tas, t10, freq="MS")
        assert out[0] == 1
        assert out[5] == 5

    def test_doy_interpolation(self):
        # Just a smoke test
        with open_dataset("ERA5/daily_surface_cancities_1990-1993.nc") as ds:
            t10 = percentile_doy(ds.tasmin, per=10).sel(percentiles=10)
            xci.tn10p(ds.tasmin, t10, freq="MS")


class TestTGXN90p:
    def test_tg90p_simple(self, tas_series):
        i = 366
        tas = np.array(range(i))
        tas = tas_series(tas, start="1/1/2000")
        t90 = percentile_doy(tas, per=10).sel(percentiles=10)

        # create cold spell in june
        tas[175:180] = 1

        out = xci.tg90p(tas, t90, freq="MS")
        assert out[0] == 30
        assert out[1] == 29
        assert out[5] == 25

    def test_tx90p_simple(self, tasmax_series):
        i = 366
        tas = np.array(range(i))
        tas = tasmax_series(tas, start="1/1/2000")
        t90 = percentile_doy(tas, per=10).sel(percentiles=10)

        # create cold spell in june
        tas[175:180] = 1

        out = xci.tx90p(tas, t90, freq="MS")
        assert out[0] == 30
        assert out[1] == 29
        assert out[5] == 25

    def test_tn90p_simple(self, tasmin_series):
        i = 366
        tas = np.array(range(i))
        tas = tasmin_series(tas, start="1/1/2000")
        t90 = percentile_doy(tas, per=10).sel(percentiles=10)

        # create cold spell in june
        tas[175:180] = 1

        out = xci.tn90p(tas, t90, freq="MS")
        assert out[0] == 30
        assert out[1] == 29
        assert out[5] == 25


class TestTas:
    @pytest.mark.parametrize("tasmin_units", ["K", "°C"])
    @pytest.mark.parametrize("tasmax_units", ["K", "°C"])
    def test_tas(
        self, tasmin_series, tasmax_series, tas_series, tasmin_units, tasmax_units
    ):
        tas = tas_series(np.ones(10) + (K2C if tasmin_units == "K" else 0))
        tas.attrs["units"] = tasmin_units
        tasmin = tasmin_series(np.zeros(10) + (K2C if tasmin_units == "K" else 0))
        tasmin.attrs["units"] = tasmin_units
        tasmax = tasmax_series(np.ones(10) * 2 + (K2C if tasmax_units == "K" else 0))
        tasmax.attrs["units"] = tasmax_units

        tas_xc = xci.tas(tasmin, tasmax)
        assert tas_xc.attrs["units"] == tasmin_units
        xr.testing.assert_equal(tas, tas_xc)


class TestTxMin:
    def test_simple(self, tasmax_series):
        a = tasmax_series(np.array([20, 25, -15, 19]))
        txm = xci.tx_min(a, freq="YS")
        assert txm == -15


class TestTxMean:
    def test_attrs(self, tasmax_series):
        a = tasmax_series(np.array([320, 321, 322, 323, 324]))
        txm = xci.tx_mean(a, freq="YS")
        assert txm == 322
        assert txm.units == "K"

        a = tasmax_series(np.array([20, 21, 22, 23, 24]))
        a.attrs["units"] = "°C"
        txm = xci.tx_mean(a, freq="YS")

        assert txm == 22
        assert txm.units == "°C"


class TestTxMax:
    def test_simple(self, tasmax_series):
        a = tasmax_series(np.array([20, 25, -15, 19]))
        txm = xci.tx_max(a, freq="YS")
        assert txm == 25


class TestTgMaxTgMinIndices:
    @staticmethod
    def random_tmin_tmax_setup(length, tasmax_series, tasmin_series):
        max_values = np.random.uniform(-20, 40, length)
        min_values = []
        for i in range(length):
            min_values.append(np.random.uniform(-40, max_values[i]))
        tasmax = tasmax_series(np.add(max_values, K2C))
        tasmin = tasmin_series(np.add(min_values, K2C))
        return tasmin, tasmax

    @staticmethod
    def static_tmin_tmax_setup(tasmin_series, tasmax_series):
        max_values = np.add([22, 10, 35.2, 25.1, 18.9, 12, 16], K2C)
        min_values = np.add([17, 3.5, 22.7, 16, 12.4, 7, 12], K2C)
        tasmax = tasmax_series(max_values)
        tasmin = tasmin_series(min_values)
        return tasmin, tasmax

    # def test_random_daily_temperature_range(self, tasmax_series, tasmin_series):
    #     days = 365
    #     tasmin, tasmax = self.random_tmin_tmax_setup(days, tasmin_series, tasmax_series)
    #     dtr = xci.daily_temperature_range(tasmin, tasmax, freq="YS")
    #
    #     np.testing.assert_array_less(-dtr, [0, 0])
    #     np.testing.assert_allclose([dtr.mean()], [20], atol=10)
    @pytest.mark.parametrize(
        "op,expected",
        [
            ("max", 12.5),
            (np.max, 12.5),
            ("min", 4.0),
            (np.min, 4.0),
            ("std", 2.72913233),
            (np.std, 2.72913233),
        ],
    )
    def test_static_reduce_daily_temperature_range(
        self, tasmin_series, tasmax_series, op, expected
    ):
        tasmin, tasmax = self.static_tmin_tmax_setup(tasmin_series, tasmax_series)
        dtr = xci.daily_temperature_range(tasmin, tasmax, freq="YS", op=op)
        assert dtr.units == "K"

        if isinstance(op, str):
            output = getattr(np, op)(tasmax - tasmin)
        else:
            output = op(tasmax - tasmin)
        np.testing.assert_array_almost_equal(dtr, expected)
        np.testing.assert_equal(dtr, output)

    def test_static_daily_temperature_range(self, tasmin_series, tasmax_series):
        tasmin, tasmax = self.static_tmin_tmax_setup(tasmin_series, tasmax_series)
        dtr = xci.daily_temperature_range(tasmin, tasmax, freq="YS")
        assert dtr.units == "K"
        output = np.mean(tasmax - tasmin)

        np.testing.assert_equal(dtr, output)

    # def test_random_variable_daily_temperature_range(self, tasmin_series, tasmax_series):
    #     days = 1095
    #     tasmin, tasmax = self.random_tmin_tmax_setup(days, tasmin_series, tasmax_series)
    #     vdtr = xci.daily_temperature_range_variability(tasmin, tasmax, freq="YS")
    #
    #     np.testing.assert_allclose(vdtr.mean(), 20, atol=10)
    #     np.testing.assert_array_less(-vdtr, [0, 0, 0, 0])

    def test_static_variable_daily_temperature_range(
        self, tasmin_series, tasmax_series
    ):
        tasmin, tasmax = self.static_tmin_tmax_setup(tasmin_series, tasmax_series)
        dtr = xci.daily_temperature_range_variability(tasmin, tasmax, freq="YS")

        np.testing.assert_almost_equal(dtr, 2.667, decimal=3)

    def test_static_extreme_temperature_range(self, tasmin_series, tasmax_series):
        tasmin, tasmax = self.static_tmin_tmax_setup(tasmin_series, tasmax_series)
        etr = xci.extreme_temperature_range(tasmin, tasmax)

        np.testing.assert_array_almost_equal(etr, 31.7)

    def test_uniform_freeze_thaw_cycles(self, tasmin_series, tasmax_series):
        temp_values = np.zeros(365)
        tasmax, tasmin = (
            tasmax_series(temp_values + 5 + K2C),
            tasmin_series(temp_values - 5 + K2C),
        )
        ft = xci.daily_freezethaw_cycles(tasmin, tasmax, freq="YS")

        np.testing.assert_array_equal([np.sum(ft)], [365])

    def test_static_freeze_thaw_cycles(self, tasmin_series, tasmax_series):
        tasmin, tasmax = self.static_tmin_tmax_setup(tasmin_series, tasmax_series)
        tasmin -= 15
        ft = xci.daily_freezethaw_cycles(tasmin, tasmax, freq="YS")

        np.testing.assert_array_equal([np.sum(ft)], [4])

    # TODO: Write a better random_freezethaw_cycles test
    # def test_random_freeze_thaw_cycles(self):
    #     runs = np.array([])
    #     for i in range(10):
    #         temp_values = np.random.uniform(-30, 30, 365)
    #         tasmin, tasmax = self.tmin_tmax_time_series(temp_values + K2C)
    #         ft = xci.daily_freezethaw_cycles(tasmin, tasmax, freq="YS")
    #         runs = np.append(runs, ft)
    #
    #     np.testing.assert_allclose(np.mean(runs), 120, atol=20)


class TestTemperatureSeasonality:
    def test_simple(self, tas_series):
        a = np.zeros(365)
        a = tas_series(a + K2C, start="1971-01-01")

        a[(a.time.dt.season == "DJF")] += -15
        a[(a.time.dt.season == "MAM")] += -5
        a[(a.time.dt.season == "JJA")] += 22
        a[(a.time.dt.season == "SON")] += 2

        out = xci.temperature_seasonality(a)
        np.testing.assert_array_almost_equal(out, 4.940925)

        t_weekly = xci.tg_mean(a, freq="7D")
        out = xci.temperature_seasonality(t_weekly)
        np.testing.assert_array_almost_equal(out, 4.87321337)
        assert out.units == "%"

    def test_celsius(self, tas_series):
        a = np.zeros(365)
        a = tas_series(a, start="1971-01-01")
        a.attrs["units"] = "°C"
        a[(a.time.dt.season == "DJF")] += -15
        a[(a.time.dt.season == "MAM")] += -5
        a[(a.time.dt.season == "JJA")] += 22
        a[(a.time.dt.season == "SON")] += 2

        out = xci.temperature_seasonality(a)
        np.testing.assert_array_almost_equal(out, 4.940925)


class TestPrecipSeasonality:
    def test_simple(self, pr_series):
        a = np.zeros(365)

        a = pr_series(a, start="1971-01-01")

        a[(a.time.dt.month == 12)] += 2 / 3600 / 24
        a[(a.time.dt.month == 8)] += 10 / 3600 / 24
        a[(a.time.dt.month == 1)] += 5 / 3600 / 24

        out = xci.precip_seasonality(a)
        np.testing.assert_array_almost_equal(out, 206.29127187)

        p_weekly = xci.precip_accumulation(a, freq="7D")
        p_weekly.attrs["units"] = "mm week-1"
        out = xci.precip_seasonality(p_weekly)
        np.testing.assert_array_almost_equal(out, 197.25293501)

        p_month = xci.precip_accumulation(a, freq="MS")
        p_month.attrs["units"] = "mm month-1"
        out = xci.precip_seasonality(p_month)
        np.testing.assert_array_almost_equal(out, 208.71994117)


class TestPrecipWettestDriestQuarter:
    def test_exceptions(self, pr_series):
        a = np.zeros(365 * 2)
        a += 1 / 3600 / 24
        a = pr_series(a, start="1971-01-01")
        a[(a.time.dt.month == 9)] += 5 / 3600 / 24
        a[(a.time.dt.month == 3)] += -1 / 3600 / 24

        with pytest.raises(NotImplementedError):
            xci.prcptot_wetdry_quarter(a, op="wettest", src_timestep="toto")
        with pytest.raises(NotImplementedError):
            xci.prcptot_wetdry_quarter(a, op="toto", src_timestep="D")

    def test_simple(self, pr_series):
        a = np.zeros(365 * 2)
        a += 1 / 3600 / 24
        a = pr_series(a, start="1971-01-01")
        a[(a.time.dt.month == 9)] += 5 / 3600 / 24
        a[(a.time.dt.month == 3)] += -1 / 3600 / 24

        out = xci.prcptot_wetdry_quarter(a, op="wettest", src_timestep="D")
        np.testing.assert_array_almost_equal(out, [241, 241])

        out = xci.prcptot_wetdry_quarter(a, op="driest", src_timestep="D")
        np.testing.assert_array_almost_equal(out, [60, 60])

    def test_weekly_monthly(self, pr_series):
        a = np.zeros(365 * 2)
        a += 1 / 3600 / 24
        a = pr_series(a, start="1971-01-01")
        a[(a.time.dt.month == 9)] += 5 / 3600 / 24
        a[(a.time.dt.month == 3)] += -1 / 3600 / 24

        p_weekly = xci.precip_accumulation(a, freq="7D")
        p_weekly.attrs["units"] = "mm week-1"
        out = xci.prcptot_wetdry_quarter(p_weekly, op="wettest", src_timestep="W")
        np.testing.assert_array_almost_equal(out, [241, 241])
        out = xci.prcptot_wetdry_quarter(p_weekly, op="driest", src_timestep="W")
        np.testing.assert_array_almost_equal(out, [60, 60])

        p_month = xci.precip_accumulation(a, freq="MS")
        p_month.attrs["units"] = "mm month-1"
        out = xci.prcptot_wetdry_quarter(p_month, op="wettest", src_timestep="M")
        np.testing.assert_array_almost_equal(out, [242, 242])
        out = xci.prcptot_wetdry_quarter(p_month, op="driest", src_timestep="M")
        np.testing.assert_array_almost_equal(out, [58, 59])

    def test_convertunits_nondaily(self, pr_series):
        a = np.zeros(365 * 2)
        a += 1 / 3600 / 24
        a = pr_series(a, start="1971-01-01")
        a[(a.time.dt.month == 9)] += 5 / 3600 / 24
        a[(a.time.dt.month == 3)] += -1 / 3600 / 24
        p_month = xci.precip_accumulation(a, freq="MS")
        p_month_m = p_month / 10
        p_month_m.attrs["units"] = "cm month-1"
        out = xci.prcptot_wetdry_quarter(p_month_m, op="wettest", src_timestep="M")
        np.testing.assert_array_almost_equal(out, [24.2, 24.2])


class TestTempWetDryPrecipWarmColdQuarter:
    @staticmethod
    def get_data(tas_series, pr_series):
        np.random.seed(123)
        times = pd.date_range("2000-01-01", "2001-12-31", name="time")
        annual_cycle = np.sin(2 * np.pi * (times.dayofyear.values / 365.25 - 0.28))
        base = 10 + 15 * annual_cycle.reshape(-1, 1)
        values = base + 3 * np.random.randn(annual_cycle.size, 1) + K2C
        tas = tas_series(values.squeeze(), start="2001-01-01").sel(
            time=slice("2001", "2002")
        )
        base = 15 * annual_cycle.reshape(-1, 1)
        values = base + 10 + 10 * np.random.randn(annual_cycle.size, 1)
        values = values / 3600 / 24
        values[values < 0] = 0
        pr = pr_series(values.squeeze(), start="2001-01-01").sel(
            time=slice("2001", "2002")
        )
        return tas, pr

    @pytest.mark.parametrize(
        "freq,units,op,expected",
        [
            (("D", "D"), "mm/day", "wettest", [296.22664037, 296.99585849]),
            (("7D", "W"), "mm/week", "wettest", [296.22664037, 296.99585849]),
            (("MS", "M"), "mm/month", "wettest", [296.25598395, 296.98613685]),
            (("D", "D"), "mm/day", "driest", [272.161376, 269.31008671]),
            (("7D", "W"), "mm/week", "driest", [272.161376, 269.31008671]),
            (("MS", "M"), "mm/month", "driest", [272.00644843, 269.04077039]),
        ],
    )
    @pytest.mark.parametrize("use_dask", [True, False])
    def test_tg_wetdry(
        self, tas_series, pr_series, use_dask, freq, units, op, expected
    ):
        tas, pr = self.get_data(tas_series, pr_series)
        freq, src_timestep = freq
        pr = xci.precip_accumulation(pr, freq=freq)
        pr.attrs["units"] = units

        tas = xci.tg_mean(tas, freq=freq)

        if use_dask:
            if freq == "D":
                pytest.skip("Daily input freq and dask arrays not working")
            tas = tas.expand_dims(lat=[0, 1, 2, 3]).chunk({"lat": 1})
            pr = pr.expand_dims(lat=[0, 1, 2, 3]).chunk({"lat": 1})

        out = xci.tg_mean_wetdry_quarter(
            tas=tas, pr=pr, freq="YS", src_timestep=src_timestep, op=op
        )
        if use_dask:
            out = out.isel(lat=0)
        np.testing.assert_array_almost_equal(out, expected)

    @pytest.mark.parametrize(
        "freq,srcts,period,op,expected",
        [
            ("D", "D", "day", "warmest", [2021.82232981, 2237.15117103]),
            ("7D", "W", "week", "warmest", [2021.82232981, 2237.15117103]),
            ("MS", "M", "month", "warmest", [2038.54763205, 2247.47136629]),
            ("D", "D", "day", "coldest", [311.91895223, 264.50013361]),
            ("7D", "W", "week", "coldest", [311.91895223, 264.50013361]),
            ("MS", "M", "month", "coldest", [311.91895223, 259.36682028]),
        ],
    )
    def test_pr_warmcold(
        self, tas_series, pr_series, freq, srcts, period, op, expected
    ):
        tas, pr = self.get_data(tas_series, pr_series)
        pr = convert_units_to(xci.precip_accumulation(pr, freq=freq), "mm")
        pr.attrs["units"] = f"{pr.units} / {period}"

        tas = xci.tg_mean(tas, freq=freq)

        out = xci.prcptot_warmcold_quarter(
            tas=tas, pr=pr, freq="YS", src_timestep=srcts, op=op
        )
        np.testing.assert_array_almost_equal(out, expected)


class TestTempWarmestColdestQuarter:
    def test_simple(self, tas_series):
        a = np.zeros(365 * 2)
        a = tas_series(a + K2C, start="1971-01-01")
        a[(a.time.dt.season == "JJA") & (a.time.dt.year == 1971)] += 22
        a[(a.time.dt.season == "SON") & (a.time.dt.year == 1972)] += 25

        a[(a.time.dt.season == "DJF") & (a.time.dt.year == 1971)] += -15
        a[(a.time.dt.season == "MAM") & (a.time.dt.year == 1972)] += -10

        with pytest.raises(NotImplementedError):
            xci.tg_mean_warmcold_quarter(a, op="warmest", src_timestep="toto")

        with pytest.raises(KeyError):
            xci.tg_mean_warmcold_quarter(a, op="toto", src_timestep="D")

        out = xci.tg_mean_warmcold_quarter(a, op="warmest", src_timestep="D")
        np.testing.assert_array_almost_equal(out, [294.66648352, 298.15])

        out = xci.tg_mean_warmcold_quarter(a, op="coldest", src_timestep="D")
        np.testing.assert_array_almost_equal(out, [263.42472527, 263.25989011])

        t_weekly = xci.tg_mean(a, freq="7D")
        out = xci.tg_mean_warmcold_quarter(t_weekly, op="coldest", src_timestep="W")
        np.testing.assert_array_almost_equal(out, [263.42472527, 263.25989011])

        t_month = xci.tg_mean(a, freq="MS")
        out = xci.tg_mean_warmcold_quarter(t_month, op="coldest", src_timestep="M")
        np.testing.assert_array_almost_equal(out, [263.15, 263.15])

    def test_Celsius(self, tas_series):
        a = np.zeros(365 * 2)
        a = tas_series(a, start="1971-01-01")
        a.attrs["units"] = "°C"
        a[(a.time.dt.season == "JJA") & (a.time.dt.year == 1971)] += 22
        a[(a.time.dt.season == "SON") & (a.time.dt.year == 1972)] += 25

        a[
            (a.time.dt.month >= 1) & (a.time.dt.month <= 3) & (a.time.dt.year == 1971)
        ] += -15
        a[(a.time.dt.season == "MAM") & (a.time.dt.year == 1972)] += -10

        out = xci.tg_mean_warmcold_quarter(a, op="warmest", src_timestep="D")
        np.testing.assert_array_almost_equal(out, [21.51648352, 25])

        out = xci.tg_mean_warmcold_quarter(a, op="coldest", src_timestep="D")
        np.testing.assert_array_almost_equal(out, [-14.835165, -9.89011])


class TestPrcptot:
    @staticmethod
    def get_data(pr_series):
        pr = pr_series(np.zeros(365 * 2), start="1971-01-01")
        pr += 1 / 3600 / 24
        pr[0:7] += 10 / 3600 / 24
        pr[-7:] += 11 / 3600 / 24
        return pr

    @pytest.mark.parametrize(
        "freq,units,expected",
        [
            (("D", "D"), "mm/day", [435.0, 442.0]),
            (("7D", "W"), "mm/week", [441.0, 436.0]),
            (("MS", "M"), "mm/month", [435.0, 442.0]),
        ],
    )
    def test_simple(self, pr_series, freq, units, expected):
        pr = self.get_data(pr_series)
        freq, src_timestep = freq
        pr = xci.precip_accumulation(pr, freq=freq)
        pr.attrs["units"] = units
        out = xci.prcptot(pr=pr, freq="YS", src_timestep=src_timestep)
        np.testing.assert_array_almost_equal(out, expected)


class TestPrecipWettestDriestPeriod:
    @staticmethod
    def get_data(pr_series):
        pr = pr_series(np.zeros(365 * 2), start="1971-01-01")
        pr += 1 / 3600 / 24
        pr[0:7] += 10 / 3600 / 24
        pr[-7:] += 11 / 3600 / 24
        return pr

    @pytest.mark.parametrize(
        "freq,units,op,expected",
        [
            (("D", "D"), "mm/day", "wettest", [11.0, 12.0]),
            (("D", "D"), "mm/day", "driest", [1, 1]),
            (("7D", "W"), "mm/week", "wettest", [77, 62]),
            (("7D", "W"), "mm/week", "driest", [7, 7]),
            (("MS", "M"), "mm/month", "wettest", [101, 107]),
            (("MS", "M"), "mm/month", "driest", [28, 29]),
        ],
    )
    def test_simple(self, pr_series, freq, units, op, expected):
        pr = self.get_data(pr_series)
        freq, src_timestep = freq
        pr = xci.precip_accumulation(pr, freq=freq)
        pr.attrs["units"] = units
        out = xci.prcptot_wetdry_period(
            pr=pr, op=op, freq="YS", src_timestep=src_timestep
        )
        np.testing.assert_array_almost_equal(out, expected)


class TestIsothermality:
    @staticmethod
    def get_data(tasmin_series, tasmax_series):
        np.random.seed(123)
        times = pd.date_range("2000-01-01", "2001-12-31", name="time")
        annual_cycle = np.sin(2 * np.pi * (times.dayofyear.values / 365.25 - 0.28))
        base = 10 + 15 * annual_cycle.reshape(-1, 1)
        values = base + 3 * np.random.randn(annual_cycle.size, 1) + K2C
        tasmin = tasmin_series(values.squeeze(), start="2001-01-01").sel(
            time=slice("2001", "2002")
        )
        values = base + 10 + 3 * np.random.randn(annual_cycle.size, 1) + K2C
        tasmax = tasmax_series(values.squeeze(), start="2001-01-01").sel(
            time=slice("2001", "2002")
        )
        return tasmin, tasmax

    @pytest.mark.parametrize(
        "freq,expected",
        [
            ("D", [18.8700109, 19.40941685]),
            ("7D", [23.29006069, 23.36559839]),
            ("MS", [25.05925319, 25.09443682]),
        ],
    )
    def test_simple(self, tasmax_series, tasmin_series, freq, expected):
        tasmin, tasmax = self.get_data(tasmin_series, tasmax_series)

        # weekly
        tmin = tasmin.resample(time=freq).mean(dim="time", keep_attrs=True)
        tmax = tasmax.resample(time=freq).mean(dim="time", keep_attrs=True)
        out = xci.isothermality(tasmax=tmax, tasmin=tmin, freq="YS")
        np.testing.assert_array_almost_equal(out, expected)
        assert out.units == "%"


class TestWarmDayFrequency:
    def test_1d(self, tasmax_series):
        a = np.zeros(35)
        a[25:] = 31
        da = tasmax_series(a + K2C)
        wdf = xci.warm_day_frequency(da, freq="MS")
        np.testing.assert_allclose(wdf.values, [6, 4])
        wdf = xci.warm_day_frequency(da, freq="YS")
        np.testing.assert_allclose(wdf.values, [10])
        wdf = xci.warm_day_frequency(da, thresh="-1 C")
        np.testing.assert_allclose(wdf.values, [35])
        wdf = xci.warm_day_frequency(da, thresh="50 C")
        np.testing.assert_allclose(wdf.values, [0])


class TestWarmNightFrequency:
    def test_1d(self, tasmin_series):
        a = np.zeros(35)
        a[25:] = 23
        da = tasmin_series(a + K2C)
        wnf = xci.warm_night_frequency(da, freq="MS")
        np.testing.assert_allclose(wnf.values, [6, 4])
        wnf = xci.warm_night_frequency(da, freq="YS")
        np.testing.assert_allclose(wnf.values, [10])
        wnf = xci.warm_night_frequency(da, thresh="-1 C")
        np.testing.assert_allclose(wnf.values, [35])
        wnf = xci.warm_night_frequency(da, thresh="50 C")
        np.testing.assert_allclose(wnf.values, [0])


class TestTxTnDaysAbove:
    def test_1d(self, tasmax_series, tasmin_series):
        tn = tasmin_series(np.asarray([20, 23, 23, 23, 23, 22, 23, 23, 23, 23]) + K2C)
        tx = tasmax_series(np.asarray([29, 31, 31, 31, 29, 31, 31, 31, 31, 31]) + K2C)

        wmmtf = xci.tx_tn_days_above(tn, tx)
        np.testing.assert_allclose(wmmtf.values, [7])
        wmmtf = xci.tx_tn_days_above(tn, tx, thresh_tasmax="50 C")
        np.testing.assert_allclose(wmmtf.values, [0])
        wmmtf = xci.tx_tn_days_above(tn, tx, thresh_tasmax="0 C", thresh_tasmin="0 C")
        np.testing.assert_allclose(wmmtf.values, [10])


class TestWarmSpellDurationIndex:
    def test_simple(self, tasmax_series):
        i = 3650
        A = 10.0
        tx = (
            np.zeros(i)
            + A * np.sin(np.arange(i) / 365.0 * 2 * np.pi)
            + 0.1 * np.random.rand(i)
        )
        tx[10:20] += 2
        tx = tasmax_series(tx)
        tx90 = percentile_doy(tx, per=90).sel(percentiles=90)

        out = xci.warm_spell_duration_index(tx, tx90, freq="YS")
        assert out[0] == 10


class TestWinterRainRatio:
    def test_simple(self, pr_series, tas_series):
        pr = np.ones(450)
        pr = pr_series(pr, start="12/1/2000")
        pr = xr.concat((pr, pr), "dim0")

        tas = np.zeros(450) - 1
        tas[10:20] += 10
        tas = tas_series(tas + K2C, start="12/1/2000")
        tas = xr.concat((tas, tas), "dim0")

        out = xci.winter_rain_ratio(pr=pr, tas=tas)
        np.testing.assert_almost_equal(out.isel(dim0=0), [10.0 / (31 + 31 + 28), 0])


# I'd like to parametrize some of these tests so we don't have to write individual tests for each indicator.
class TestTG:
    @pytest.mark.parametrize(
        "ind,exp",
        [(xci.tg_mean, 283.1391), (xci.tg_min, 266.1117), (xci.tg_max, 292.1250)],
    )
    def test_simple(self, ind, exp):
        ds = open_dataset("ERA5/daily_surface_cancities_1990-1993.nc")
        out = ind(ds.tas.sel(location="Victoria"))
        np.testing.assert_almost_equal(out[0], exp, decimal=4)

    def test_indice_against_icclim(self, cmip3_day_tas):
        from xclim import icclim

        ind = xci.tg_mean(cmip3_day_tas)
        icclim = icclim.TG(cmip3_day_tas)

        np.testing.assert_array_equal(icclim, ind)


class TestFireWeatherIndex:
    nc_gfwed = os.path.join("FWI", "GFWED_sample_2017.nc")

    def test_fire_weather_indexes(self):
        ds = open_dataset(self.nc_gfwed)
        fwis = xci.fire_weather_indexes(
            ds.tas,
            ds.prbc,
            ds.sfcwind,
            ds.rh,
            ds.lat,
            snd=ds.snow_depth,
            ffmc0=ds.FFMC.sel(time="2017-03-02"),
            dmc0=ds.DMC.sel(time="2017-03-02"),
            dc0=ds.DC.sel(time="2017-03-02"),
            start_date="2017-03-03",
            start_up_mode="snow_depth",
            shut_down_mode="snow_depth",
        )
        for ind, name in zip(fwis, ["DC", "DMC", "FFMC", "ISI", "BUI", "FWI"]):
            np.testing.assert_allclose(
                ind.where(ds[name].notnull()).sel(time=slice("2017-06-01", None)),
                ds[name].sel(time=slice("2017-06-01", None)),
                rtol=1e-4,
                atol=1e-4,
            )

    def test_drought_code(self):
        ds = open_dataset(self.nc_gfwed)
        dc = xci.drought_code(
            ds.tas,
            ds.prbc,
            ds.lat,
            snd=ds.snow_depth,
            dc0=ds.DC.sel(time="2017-03-02"),
            start_date="2017-03-03",
            start_up_mode="snow_depth",
            shut_down_mode="snow_depth",
        )
        np.testing.assert_allclose(
            dc.where(ds.DC.notnull()).sel(time=slice("2017-06-01", None)),
            ds.DC.sel(time=slice("2017-06-01", None)),
            rtol=1e-4,
            atol=1e-4,
        )


@pytest.fixture(scope="session")
def cmip3_day_tas():
    # xr.set_options(enable_cftimeindex=False)
    ds = open_dataset(os.path.join("cmip3", "tas.sresb1.giss_model_e_r.run1.atm.da.nc"))
    yield ds.tas
    ds.close()


class TestWindConversion:
    da_uas = xr.DataArray(
        np.array([[3.6, -3.6], [-1, 0]]),
        coords={"lon": [-72, -72], "lat": [55, 55]},
        dims=["lon", "lat"],
    )
    da_uas.attrs["units"] = "km/h"
    da_vas = xr.DataArray(
        np.array([[3.6, 3.6], [-1, -18]]),
        coords={"lon": [-72, -72], "lat": [55, 55]},
        dims=["lon", "lat"],
    )
    da_vas.attrs["units"] = "km/h"
    da_wind = xr.DataArray(
        np.array([[np.hypot(3.6, 3.6), np.hypot(3.6, 3.6)], [np.hypot(1, 1), 18]]),
        coords={"lon": [-72, -72], "lat": [55, 55]},
        dims=["lon", "lat"],
    )
    da_wind.attrs["units"] = "km/h"
    da_windfromdir = xr.DataArray(
        np.array([[225, 135], [0, 360]]),
        coords={"lon": [-72, -72], "lat": [55, 55]},
        dims=["lon", "lat"],
    )
    da_windfromdir.attrs["units"] = "degree"

    def test_uas_vas_2_sfcwind(self):
        wind, windfromdir = xci.uas_vas_2_sfcwind(self.da_uas, self.da_vas)

        assert np.all(
            np.around(wind.values, decimals=10)
            == np.around(self.da_wind.values / 3.6, decimals=10)
        )
        assert np.all(
            np.around(windfromdir.values, decimals=10)
            == np.around(self.da_windfromdir.values, decimals=10)
        )

    def test_sfcwind_2_uas_vas(self):
        uas, vas = xci.sfcwind_2_uas_vas(self.da_wind, self.da_windfromdir)

        assert np.all(np.around(uas.values, decimals=10) == np.array([[1, -1], [0, 0]]))
        assert np.all(
            np.around(vas.values, decimals=10)
            == np.around(np.array([[1, 1], [-(np.hypot(1, 1)) / 3.6, -5]]), decimals=10)
        )


@pytest.mark.parametrize(
    "method", ["bohren98", "tetens30", "sonntag90", "goffgratch46", "wmo08"]
)
@pytest.mark.parametrize(
    "invalid_values,exp0", [("clip", 100), ("mask", np.nan), (None, 151)]
)
def test_relative_humidity_dewpoint(
    tas_series, rh_series, method, invalid_values, exp0
):
    np.testing.assert_allclose(
        xci.relative_humidity(
            tas=tas_series(np.array([-20, -10, -1, 10, 20, 25, 30, 40, 60]) + K2C),
            dtas=tas_series(np.array([-15, -10, -2, 5, 10, 20, 29, 20, 30]) + K2C),
            method=method,
            invalid_values=invalid_values,
        ),
        # Expected values obtained by hand calculation
        rh_series([exp0, 100, 93, 71, 52, 73, 94, 31, 20]),
        rtol=0.02,
        atol=1,
    )


@pytest.mark.parametrize("method", ["tetens30", "sonntag90", "goffgratch46", "wmo08"])
@pytest.mark.parametrize(
    "ice_thresh,exp0", [(None, [125, 286, 568]), ("0 degC", [103, 260, 563])]
)
def test_saturation_vapor_pressure(tas_series, method, ice_thresh, exp0):
    tas = tas_series(np.array([-20, -10, -1, 10, 20, 25, 30, 40, 60]) + K2C)
    # Expected values obtained with the Sonntag90 method
    e_sat_exp = exp0 + [1228, 2339, 3169, 4247, 7385, 19947]

    e_sat = xci.saturation_vapor_pressure(
        tas=tas,
        method=method,
        ice_thresh=ice_thresh,
    )
    np.testing.assert_allclose(e_sat, e_sat_exp, atol=0.5, rtol=0.005)


@pytest.mark.parametrize("method", ["tetens30", "sonntag90", "goffgratch46", "wmo08"])
@pytest.mark.parametrize(
    "invalid_values,exp0", [("clip", 100), ("mask", np.nan), (None, 188)]
)
def test_relative_humidity(
    tas_series, rh_series, huss_series, ps_series, method, invalid_values, exp0
):
    tas = tas_series(np.array([-10, -10, 10, 20, 35, 50, 75, 95]) + K2C)
    # Expected values obtained with the Sonntag90 method
    rh_exp = rh_series([exp0, 63.0, 66.0, 34.0, 14.0, 6.0, 1.0, 0.0])
    ps = ps_series([101325] * 8)
    huss = huss_series([0.003, 0.001] + [0.005] * 7)

    rh = xci.relative_humidity(
        tas=tas,
        huss=huss,
        ps=ps,
        method=method,
        invalid_values=invalid_values,
        ice_thresh="0 degC",
    )
    np.testing.assert_allclose(rh, rh_exp, atol=0.5, rtol=0.005)


@pytest.mark.parametrize("method", ["tetens30", "sonntag90", "goffgratch46", "wmo08"])
@pytest.mark.parametrize(
    "invalid_values,exp0", [("clip", 1.4e-2), ("mask", np.nan), (None, 2.2e-2)]
)
def test_specific_humidity(
    tas_series, rh_series, huss_series, ps_series, method, invalid_values, exp0
):
    tas = tas_series(np.array([20, -10, 10, 20, 35, 50, 75, 95]) + K2C)
    rh = rh_series([150, 10, 90, 20, 80, 50, 70, 40, 30])
    ps = ps_series(1000 * np.array([100] * 4 + [101] * 4))
    # Expected values obtained with the Sonntag90 method
    huss_exp = huss_series(
        [exp0, 1.6e-4, 6.9e-3, 3.0e-3, 2.9e-2, 4.1e-2, 2.1e-1, 5.7e-1]
    )

    huss = xci.specific_humidity(
        tas=tas,
        rh=rh,
        ps=ps,
        method=method,
        invalid_values=invalid_values,
        ice_thresh="0 degC",
    )
    np.testing.assert_allclose(huss, huss_exp, atol=1e-4, rtol=0.05)


def test_degree_days_exceedance_date(tas_series):
    tas = tas_series(np.ones(366) + K2C, start="2000-01-01")

    out = xci.degree_days_exceedance_date(
        tas, thresh="0 degC", op=">", sum_thresh="150 K days"
    )
    assert out[0] == 151

    out = xci.degree_days_exceedance_date(
        tas, thresh="2 degC", op="<", sum_thresh="150 degC days"
    )
    assert out[0] == 151

    out = xci.degree_days_exceedance_date(
        tas, thresh="2 degC", op="<", sum_thresh="150 K days", after_date="04-15"
    )
    assert out[0] == 256


@pytest.mark.parametrize("method,exp", [("binary", [1, 1, 1, 1, 1, 0, 0, 0, 0, 0])])
def test_snowfall_approximation(pr_series, tasmax_series, method, exp):
    pr = pr_series(np.ones(10))
    tasmax = tasmax_series(np.arange(10) + K2C)

    prsn = xci.snowfall_approximation(pr, tas=tasmax, thresh="5 degC", method=method)

    np.testing.assert_allclose(prsn, exp, atol=1e-5, rtol=1e-3)


@pytest.mark.parametrize("method,exp", [("binary", [0, 0, 0, 0, 0, 1, 1, 1, 1, 1])])
def test_rain_approximation(pr_series, tas_series, method, exp):
    pr = pr_series(np.ones(10))
    tas = tas_series(np.arange(10) + K2C)

    prlp = xci.rain_approximation(pr, tas=tas, thresh="5 degC", method=method)

    np.testing.assert_allclose(prlp, exp, atol=1e-5, rtol=1e-3)


def test_first_snowfall(prsn_series):
    prsn = prsn_series(30 - abs(np.arange(366) - 180), start="01-01-2000")
    out = xci.first_snowfall(prsn, thresh="15 kg m-2 s-1", freq="YS")
    assert out[0] == 166


def test_last_snowfall(prsn_series):
    prsn = prsn_series(30 - abs(np.arange(366) - 180), start="01-01-2000")
    out = xci.last_snowfall(prsn, thresh="15 kg m-2 s-1", freq="YS")
    assert out[0] == 196

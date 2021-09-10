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
from xclim.core.calendar import date_range, percentile_doy
from xclim.core.options import set_options
from xclim.core.units import ValidationError, convert_units_to, units
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


class TestAgroclimaticIndices:
    def test_corn_heat_units(self, tasmin_series, tasmax_series):
        tn = tasmin_series(np.array([-10, 5, 4, 3, 10]) + K2C)
        tx = tasmax_series(np.array([-5, 9, 10, 16, 20]) + K2C)

        out = xci.corn_heat_units(
            tn, tx, thresh_tasmin="4.44 degC", thresh_tasmax="10 degC"
        )
        np.testing.assert_allclose(out, [0, 0.504, 0, 8.478, 17.454])

    @pytest.mark.parametrize(
        "method, end_date, deg_days, max_deg_days",
        [
            ("gladstones", "11-01", 1102.1, 1926.0),
            ("jones", "11-01", 1203.4, 2179.1),
            ("icclim", "10-01", 915.0, 1647.0),
        ],
    )
    def test_bedd(self, method, end_date, deg_days, max_deg_days):

        time_data = date_range(
            "1992-01-01", "1995-06-01", freq="D", calendar="standard"
        )
        tn = xr.DataArray(
            np.zeros(time_data.size) + 10 + K2C,
            dims="time",
            coords={"time": time_data},
            attrs=dict(units="K"),
        )

        tx = xr.DataArray(
            np.zeros(time_data.size) + 20 + K2C,
            dims="time",
            coords={"time": time_data},
            attrs=dict(units="K"),
        )

        tx_hot = xr.DataArray(
            np.zeros(time_data.size) + 50 + K2C,
            dims="time",
            coords={"time": time_data},
            attrs=dict(units="K"),
        )

        lat = np.array([45])
        high_lat = np.array([48])

        bedd = xci.biologically_effective_degree_days(
            tasmin=tn,
            tasmax=tx,
            lat=lat,  # noqa
            method=method,
            end_date=end_date,  # noqa
            freq="YS",
        )
        bedd_hot = xci.biologically_effective_degree_days(
            tasmin=tn,
            tasmax=tx_hot,
            lat=lat,  # noqa
            method=method,
            end_date=end_date,  # noqa
            freq="YS",
        )
        bedd_high_lat = xci.biologically_effective_degree_days(
            tasmin=tn,
            tasmax=tx,
            lat=high_lat,  # noqa
            method=method,
            end_date=end_date,  # noqa
            freq="YS",
        )

        if method == "jones":
            np.testing.assert_array_less(
                bedd[1], bedd[0]
            )  # Leap-year has slightly higher values
            np.testing.assert_allclose(
                bedd, np.array([deg_days, deg_days, deg_days, np.NaN]), rtol=3e-4
            )
            np.testing.assert_allclose(
                bedd_hot, [max_deg_days, max_deg_days, max_deg_days, np.NaN], rtol=0.15
            )

        else:
            np.testing.assert_allclose(
                bedd, np.array([deg_days, deg_days, deg_days, np.NaN])
            )
            np.testing.assert_array_equal(
                bedd_hot, [max_deg_days, max_deg_days, max_deg_days, np.NaN]
            )
            if method == "gladstones":
                np.testing.assert_array_less(bedd, bedd_high_lat)
            if method == "icclim":
                np.testing.assert_array_equal(bedd, bedd_high_lat)

    def test_cool_night_index(self):
        ds = open_dataset("cmip5/tas_Amon_CanESM2_rcp85_r1i1p1_200701-200712.nc")
        ds = ds.rename(dict(tas="tasmin"))

        cni = xci.cool_night_index(tasmin=ds.tasmin, lat=ds.lat)
        tasmin = convert_units_to(ds.tasmin, "degC")

        cni_nh = cni.where(cni.lat >= 0, drop=True)
        cni_sh = cni.where(cni.lat < 0, drop=True)

        tn_nh = tasmin.where((tasmin.lat >= 0) & (tasmin.time.dt.month == 9), drop=True)
        tn_sh = tasmin.where((tasmin.lat < 0) & (tasmin.time.dt.month == 3), drop=True)

        np.testing.assert_array_equal(cni_nh, tn_nh)
        np.testing.assert_array_equal(cni_sh, tn_sh)

    @pytest.mark.parametrize(
        "lat_factor, values",
        [
            (60, [135.34, 918.79, 1498.31, 1221.80, 271.72]),
            (75, [55.35, 1058.55, 1895.97, 1472.18, 298.74]),
        ],
    )
    def test_lat_temperature_index(self, lat_factor, values):
        ds = open_dataset("cmip5/tas_Amon_CanESM2_rcp85_r1i1p1_200701-200712.nc")
        ds = ds.drop_isel(time=0)  # drop time=2006/12 for one year of data

        lti = xci.latitude_temperature_index(
            tas=ds.tas, lat=ds.lat, lat_factor=lat_factor
        )
        assert lti.where(abs(lti.lat) > lat_factor).sum() == 0

        lti = lti.where(abs(lti.lat) <= lat_factor, drop=True).where(
            lti.lon <= 35, drop=True
        )
        lti = lti.groupby_bins(lti.lon, 1).mean().groupby_bins(lti.lat, 5).mean()
        np.testing.assert_array_almost_equal(lti[0].transpose(), np.array([values]), 2)

    @pytest.mark.parametrize(
        "method, end_date, values",
        [
            ("smoothed", "10-01", 1702.87),
            ("icclim", "11-01", 1983.53),
            ("jones", "10-01", 1729.12),
            ("jones", "11-01", 2219.51),
        ],
    )
    def test_huglin_index(self, method, end_date, values):
        ds = open_dataset("cmip5/tas_Amon_CanESM2_rcp85_r1i1p1_200701-200712.nc")
        ds = ds.drop_isel(time=0)  # drop time=2006/12 for one year of data

        tasmax, tasmin = ds.tas + 15, ds.tas - 5
        # It would be much better if the index would interpolate to daily from monthly data intelligently.
        tasmax, tasmin = (
            tasmax.resample(time="1D").interpolate("cubic"),
            tasmin.resample(time="1D").interpolate("cubic"),
        )
        tasmax.attrs["units"], tasmin.attrs["units"] = "K", "K"

        hi = xci.huglin_index(
            tasmax=tasmax,
            tasmin=tasmin,
            lat=ds.lat,
            method=method,
            end_date=end_date,  # noqa
        )

        np.testing.assert_almost_equal(np.mean(hi), values, 2)

    def test_qian_weighted_mean_average(self, tas_series):
        mg = np.zeros(365)

        # False start
        mg[10:20] = [1, 2, 5, 6, 1, 2, 4, 5, 4, 1]
        mg[20:40] = np.ones(20)

        # Actual start
        mg[40:50] = np.arange(1, 11)

        mg = tas_series(mg + K2C)
        out = xci.qian_weighted_mean_average(mg, dim="time")
        np.testing.assert_array_equal(
            out[7:12], [273.15, 273.2125, 273.525, 274.3375, 275.775]
        )
        assert out[50].values < (10 + K2C)
        assert out[51].values > K2C
        assert out.attrs["units"] == "K"

    @pytest.mark.parametrize("method,expected", [("bootsma", 2267), ("qian", 2252.0)])
    def test_effective_growing_degree_days(
        self, tasmax_series, tasmin_series, method, expected
    ):
        mg = np.zeros(547)

        # False start
        mg[192:202] = [1, 2, 5, 6, 1, 2, 4, 5, 4, 1]
        mg[202:222] = np.ones(20)
        mg[213] = 20  # An outlier day to test start date (Adds 15 deg days)

        # Actual start
        mg[222:242] = np.arange(1, 21)
        mg[242:382] = np.repeat(20, 140)
        mg[382:392] = np.array([20, 15, 12, 10, 7, 0, -1, 2, 1, -10])

        mx = tasmax_series(mg + K2C + 10)
        mn = tasmin_series(mg + K2C - 10)
        out = xci.effective_growing_degree_days(
            tasmax=mx, tasmin=mn, method=method, freq="YS"
        )
        np.testing.assert_array_equal(out, np.array([np.NaN, expected]))


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
        for attr in ["units", "is_dayofyear", "calendar"]:
            assert attr in lsf.attrs.keys()
        assert lsf.attrs["units"] == ""
        assert lsf.attrs["is_dayofyear"] == 1


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
        for attr in ["units", "is_dayofyear", "calendar"]:
            assert attr in fdb.attrs.keys()
        assert fdb.attrs["units"] == ""
        assert fdb.attrs["is_dayofyear"] == 1


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
        for attr in ["units", "is_dayofyear", "calendar"]:
            assert attr in fdb.attrs.keys()
        assert fdb.attrs["units"] == ""
        assert fdb.attrs["is_dayofyear"] == 1


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
        for attr in ["units", "is_dayofyear", "calendar"]:
            assert attr in out.attrs.keys()
        assert out.attrs["units"] == ""
        assert out.attrs["is_dayofyear"] == 1

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
        for attr in ["units", "is_dayofyear", "calendar"]:
            assert attr in gs_end.attrs.keys()
        assert gs_end.attrs["units"] == ""
        assert gs_end.attrs["is_dayofyear"] == 1


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


class TestTnDays:
    def test_above_simple(self, tasmin_series):
        a = np.zeros(365)
        a[:6] += [27, 28, 29, 30, 31, 32]  # 2 above 30
        mn = tasmin_series(a + K2C)

        out = xci.tn_days_above(mn, thresh="30 C")
        np.testing.assert_array_equal(out[:1], [2])
        np.testing.assert_array_equal(out[1:], [0])

    def test_below_simple(self, tasmin_series):
        a = np.zeros(365)
        a[:6] -= [27, 28, 29, 30, 31, 32]  # 2 above 30
        mn = tasmin_series(a + K2C)

        out = xci.tn_days_below(mn, thresh="-10 C")
        np.testing.assert_array_equal(out[:1], [6])
        np.testing.assert_array_equal(out[1:], [0])
        out = xci.tn_days_below(mn, thresh="-30 C")
        np.testing.assert_array_equal(out[:1], [2])
        np.testing.assert_array_equal(out[1:], [0])


class TestTgDays:
    def test_above_simple(self, tas_series):
        a = np.zeros(365)
        a[:6] += [27, 28, 29, 30, 31, 32]  # 2 above 30
        mg = tas_series(a + K2C)

        out = xci.tg_days_above(mg, thresh="30 C")
        np.testing.assert_array_equal(out[:1], [2])
        np.testing.assert_array_equal(out[1:], [0])

    def test_below_simple(self, tas_series):
        a = np.zeros(365)
        a[:6] -= [27, 28, 29, 30, 31, 32]  # 2 above 30
        mg = tas_series(a + K2C)

        out = xci.tg_days_below(mg, thresh="-10 C")
        np.testing.assert_array_equal(out[:1], [6])
        np.testing.assert_array_equal(out[1:], [0])
        out = xci.tg_days_below(mg, thresh="-30 C")
        np.testing.assert_array_equal(out[:1], [2])
        np.testing.assert_array_equal(out[1:], [0])


class TestTxDays:
    def test_above_simple(self, tasmax_series):
        a = np.zeros(365)
        a[:6] += [27, 28, 29, 30, 31, 32]  # 2 above 30
        mx = tasmax_series(a + K2C)

        out = xci.tx_days_above(mx, thresh="30 C")
        np.testing.assert_array_equal(out[:1], [2])
        np.testing.assert_array_equal(out[1:], [0])

    def test_below_simple(self, tasmax_series):
        a = np.zeros(365)
        a[:6] -= [27, 28, 29, 30, 31, 32]  # 2 above 30
        mx = tasmax_series(a + K2C)

        out = xci.tx_days_below(mx, thresh="-10 C")
        np.testing.assert_array_equal(out[:1], [6])
        np.testing.assert_array_equal(out[1:], [0])
        out = xci.tx_days_below(mx, thresh="-30 C")
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
        assert out[0] == 0
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
        assert out[0] == 0
        assert out[5] == 5

    def test_tn10p_simple(self, tas_series):
        i = 366
        tas = np.array(range(i))
        tas = tas_series(tas, start="1/1/2000")
        t10 = percentile_doy(tas, per=10).sel(percentiles=10)

        # create cold spell in june
        tas[175:180] = 1

        out = xci.tn10p(tas, t10, freq="MS")
        assert out[0] == 0
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
    @staticmethod
    def get_data(pr_series):
        a = np.ones(731)
        a = pr_series(a, start="1971-01-01", units="mm/d")
        a[(a.time.dt.month == 9)] += 5
        a[(a.time.dt.month == 3)] += -1
        return a

    def test_exceptions(self, pr_series):
        a = self.get_data(pr_series)
        with pytest.raises(NotImplementedError):
            xci.prcptot_wetdry_quarter(a, op="toto")

    def test_simple(self, pr_series):
        a = self.get_data(pr_series)

        out = xci.prcptot_wetdry_quarter(a, op="wettest")
        np.testing.assert_array_almost_equal(out, [241, 241])

        out = xci.prcptot_wetdry_quarter(a, op="driest")
        np.testing.assert_array_almost_equal(out, [60, 60])

    def test_weekly_monthly(self, pr_series):
        a = self.get_data(pr_series)

        p_weekly = xci.precip_accumulation(a, freq="7D")
        p_weekly.attrs["units"] = "mm week-1"
        out = xci.prcptot_wetdry_quarter(p_weekly, op="wettest")
        np.testing.assert_array_almost_equal(out, [241, 241])
        out = xci.prcptot_wetdry_quarter(p_weekly, op="driest")
        np.testing.assert_array_almost_equal(out, [60, 60])

        # Can't use precip_accumulation cause "month" is not a constant unit
        p_month = a.resample(time="MS").mean(keep_attrs=True)
        out = xci.prcptot_wetdry_quarter(p_month, op="wettest")
        np.testing.assert_array_almost_equal(out, [242, 242])
        out = xci.prcptot_wetdry_quarter(p_month, op="driest")
        np.testing.assert_array_almost_equal(out, [58, 59])

    def test_convertunits_nondaily(self, pr_series):
        a = self.get_data(pr_series)
        p_month = a.resample(time="MS").mean(keep_attrs=True)
        p_month_m = p_month / 10
        p_month_m.attrs["units"] = "cm day-1"
        out = xci.prcptot_wetdry_quarter(p_month_m, op="wettest")
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
        "freq,op,expected",
        [
            ("D", "wettest", [296.22664037, 296.99585849]),
            ("7D", "wettest", [296.22664037, 296.99585849]),
            ("MS", "wettest", [296.25598395, 296.98613685]),
            ("D", "driest", [272.161376, 269.31008671]),
            ("7D", "driest", [272.161376, 269.31008671]),
            ("MS", "driest", [272.00644843, 269.04077039]),
        ],
    )
    @pytest.mark.parametrize("use_dask", [True, False])
    def test_tg_wetdry(self, tas_series, pr_series, use_dask, freq, op, expected):
        tas, pr = self.get_data(tas_series, pr_series)
        pr = pr.resample(time=freq).mean(keep_attrs=True)

        tas = xci.tg_mean(tas, freq=freq)

        if use_dask:
            if freq == "D":
                pytest.skip("Daily input freq and dask arrays not working")
            tas = tas.expand_dims(lat=[0, 1, 2, 3]).chunk({"lat": 1})
            pr = pr.expand_dims(lat=[0, 1, 2, 3]).chunk({"lat": 1})

        out = xci.tg_mean_wetdry_quarter(tas=tas, pr=pr, freq="YS", op=op)
        if use_dask:
            out = out.isel(lat=0)
        np.testing.assert_array_almost_equal(out, expected)

    @pytest.mark.parametrize(
        "freq,op,expected",
        [
            ("D", "warmest", [2021.82232981, 2237.15117103]),
            ("7D", "warmest", [2021.82232981, 2237.15117103]),
            ("MS", "warmest", [2038.54763205, 2247.47136629]),
            ("D", "coldest", [311.91895223, 264.50013361]),
            ("7D", "coldest", [311.91895223, 264.50013361]),
            ("MS", "coldest", [311.91895223, 259.36682028]),
        ],
    )
    def test_pr_warmcold(self, tas_series, pr_series, freq, op, expected):
        tas, pr = self.get_data(tas_series, pr_series)
        pr = convert_units_to(pr.resample(time=freq).mean(keep_attrs=True), "mm/d")

        tas = xci.tg_mean(tas, freq=freq)

        out = xci.prcptot_warmcold_quarter(tas=tas, pr=pr, freq="YS", op=op)
        np.testing.assert_array_almost_equal(out, expected)


class TestTempWarmestColdestQuarter:
    def test_simple(self, tas_series):
        a = np.zeros(365 * 2)
        a = tas_series(a + K2C, start="1971-01-01")
        a[(a.time.dt.season == "JJA") & (a.time.dt.year == 1971)] += 22
        a[(a.time.dt.season == "SON") & (a.time.dt.year == 1972)] += 25

        a[(a.time.dt.season == "DJF") & (a.time.dt.year == 1971)] += -15
        a[(a.time.dt.season == "MAM") & (a.time.dt.year == 1972)] += -10

        with pytest.raises(KeyError):
            xci.tg_mean_warmcold_quarter(a, op="toto")

        out = xci.tg_mean_warmcold_quarter(a, op="warmest")
        np.testing.assert_array_almost_equal(out, [294.66648352, 298.15])

        out = xci.tg_mean_warmcold_quarter(a, op="coldest")
        np.testing.assert_array_almost_equal(out, [263.42472527, 263.25989011])

        t_weekly = xci.tg_mean(a, freq="7D")
        out = xci.tg_mean_warmcold_quarter(t_weekly, op="coldest")
        np.testing.assert_array_almost_equal(out, [263.42472527, 263.25989011])

        t_month = xci.tg_mean(a, freq="MS")
        out = xci.tg_mean_warmcold_quarter(t_month, op="coldest")
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

        out = xci.tg_mean_warmcold_quarter(a, op="warmest")
        np.testing.assert_array_almost_equal(out, [21.51648352, 25])

        out = xci.tg_mean_warmcold_quarter(a, op="coldest")
        np.testing.assert_array_almost_equal(out, [-14.835165, -9.89011])


class TestPrcptot:
    @staticmethod
    def get_data(pr_series):
        pr = pr_series(np.ones(731), start="1971-01-01", units="mm / d")
        pr[0:7] += 10
        pr[-7:] += 11
        return pr

    @pytest.mark.parametrize(
        "freq,expected",
        [
            ("D", [435.0, 443.0]),
            ("7D", [441.0, 485.0]),
            ("MS", [435.0, 443.0]),
        ],
    )
    def test_simple(self, pr_series, freq, expected):
        pr = self.get_data(pr_series)
        pr = pr.resample(time=freq).mean(keep_attrs=True)
        out = xci.prcptot(pr=pr, freq="YS")
        np.testing.assert_array_almost_equal(out, expected)


class TestPrecipWettestDriestPeriod:
    @staticmethod
    def get_data(pr_series):
        pr = pr_series(np.ones(731), start="1971-01-01", units="mm / d")
        pr[0:7] += 10
        pr[-7:] += 11
        return pr

    @pytest.mark.parametrize(
        "freq,op,expected",
        [
            ("D", "wettest", [11.0, 12.0]),
            ("D", "driest", [1, 1]),
            ("7D", "wettest", [77, 84]),
            ("7D", "driest", [7, 7]),
            ("MS", "wettest", [101, 108]),
            ("MS", "driest", [28, 29]),
        ],
    )
    def test_simple(self, pr_series, freq, op, expected):
        pr = self.get_data(pr_series)
        pr = pr.resample(time=freq).mean(keep_attrs=True)
        out = xci.prcptot_wetdry_period(pr=pr, op=op, freq="YS")
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


class TestWindIndices:
    def test_calm_days(self, sfcWind_series):
        a = np.full(365, 20)  # all non-calm days
        a[10:20] = 2  # non-calm day on default thres, but should count as calm in test
        a[40:50] = 3.1  # non-calm day on test threshold
        da = sfcWind_series(a)
        out = xci.calm_days(da, thresh="3 km h-1", freq="M")
        np.testing.assert_array_equal(out, [10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        assert out.units == "d"

    def test_windy_days(self, sfcWind_series):
        a = np.zeros(365)  # all non-windy days
        a[10:20] = 10.8  # windy day on default threshold, non-windy in test
        a[40:50] = 12  # windy day on test threshold
        a[80:90] = 15  # windy days
        da = sfcWind_series(a)
        out = xci.windy_days(da, thresh="12 km h-1", freq="M")
        np.testing.assert_array_equal(out, [0, 10, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        assert out.units == "d"


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
        from xclim.indicators import icclim  # noqa

        with set_options(cf_compliance="log"):
            ind = xci.tg_mean(cmip3_day_tas)
            icclim = icclim.TG(cmip3_day_tas)

        np.testing.assert_array_equal(icclim, ind)


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
    tas_series, hurs_series, method, invalid_values, exp0
):
    np.testing.assert_allclose(
        xci.relative_humidity(
            tas=tas_series(np.array([-20, -10, -1, 10, 20, 25, 30, 40, 60]) + K2C),
            tdps=tas_series(np.array([-15, -10, -2, 5, 10, 20, 29, 20, 30]) + K2C),
            method=method,
            invalid_values=invalid_values,
        ),
        # Expected values obtained by hand calculation
        hurs_series([exp0, 100, 93, 71, 52, 73, 94, 31, 20]),
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
    tas_series, hurs_series, huss_series, ps_series, method, invalid_values, exp0
):
    tas = tas_series(np.array([-10, -10, 10, 20, 35, 50, 75, 95]) + K2C)
    # Expected values obtained with the Sonntag90 method
    hurs_exp = hurs_series([exp0, 63.0, 66.0, 34.0, 14.0, 6.0, 1.0, 0.0])
    ps = ps_series([101325] * 8)
    huss = huss_series([0.003, 0.001] + [0.005] * 7)

    hurs = xci.relative_humidity(
        tas=tas,
        huss=huss,
        ps=ps,
        method=method,
        invalid_values=invalid_values,
        ice_thresh="0 degC",
    )
    np.testing.assert_allclose(hurs, hurs_exp, atol=0.5, rtol=0.005)


@pytest.mark.parametrize("method", ["tetens30", "sonntag90", "goffgratch46", "wmo08"])
@pytest.mark.parametrize(
    "invalid_values,exp0", [("clip", 1.4e-2), ("mask", np.nan), (None, 2.2e-2)]
)
def test_specific_humidity(
    tas_series, hurs_series, huss_series, ps_series, method, invalid_values, exp0
):
    tas = tas_series(np.array([20, -10, 10, 20, 35, 50, 75, 95]) + K2C)
    hurs = hurs_series([150, 10, 90, 20, 80, 50, 70, 40, 30])
    ps = ps_series(1000 * np.array([100] * 4 + [101] * 4))
    # Expected values obtained with the Sonntag90 method
    huss_exp = huss_series(
        [exp0, 1.6e-4, 6.9e-3, 3.0e-3, 2.9e-2, 4.1e-2, 2.1e-1, 5.7e-1]
    )

    huss = xci.specific_humidity(
        tas=tas,
        hurs=hurs,
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

    for attr in ["units", "is_dayofyear", "calendar"]:
        assert attr in out.attrs.keys()
    assert out.attrs["units"] == ""
    assert out.attrs["is_dayofyear"] == 1


@pytest.mark.parametrize(
    "method,exp",
    [
        ("binary", [1, 1, 1, 0, 0, 0, 0, 0, 0, 0]),
        ("brown", [1, 1, 1, 0.5, 0, 0, 0, 0, 0, 0]),
        ("auer", [1, 1, 1, 0.89805, 0.593292, 0.289366, 0.116624, 0.055821, 0, 0]),
    ],
)
def test_snowfall_approximation(pr_series, tasmax_series, method, exp):
    pr = pr_series(np.ones(10))
    tasmax = tasmax_series(np.arange(10) + K2C)

    prsn = xci.snowfall_approximation(pr, tas=tasmax, thresh="2 degC", method=method)

    np.testing.assert_allclose(prsn, exp, atol=1e-5, rtol=1e-3)


@pytest.mark.parametrize("method,exp", [("binary", [0, 0, 0, 0, 0, 0, 1, 1, 1, 1])])
def test_rain_approximation(pr_series, tas_series, method, exp):
    pr = pr_series(np.ones(10))
    tas = tas_series(np.arange(10) + K2C)

    prlp = xci.rain_approximation(pr, tas=tas, thresh="5 degC", method=method)

    np.testing.assert_allclose(prlp, exp, atol=1e-5, rtol=1e-3)


def test_first_snowfall(prsn_series):
    prsn = prsn_series(30 - abs(np.arange(366) - 180), start="2000-01-01")
    out = xci.first_snowfall(prsn, thresh="15 kg m-2 s-1", freq="YS")
    assert out[0] == 166
    for attr in ["units", "is_dayofyear", "calendar"]:
        assert attr in out.attrs.keys()
    assert out.attrs["units"] == ""
    assert out.attrs["is_dayofyear"] == 1


def test_last_snowfall(prsn_series):
    prsn = prsn_series(30 - abs(np.arange(366) - 180), start="2000-01-01")
    out = xci.last_snowfall(prsn, thresh="15 kg m-2 s-1", freq="YS")
    assert out[0] == 196


def test_days_with_snow(prsn_series):
    prsn = prsn_series(np.arange(365), start="2000-01-01")
    out = xci.days_with_snow(prsn)
    assert len(out) == 2
    # Days with 0 and 1 are not counted, because condition is > thresh, not >=.
    assert sum(out) == 364

    out = xci.days_with_snow(prsn, low="10 kg m-2 s-1", high="20 kg m-2 s-1")
    np.testing.assert_array_equal(out, [10, 0])
    assert out.units == "d"


def test_snow_cover_duration(snd_series):
    a = np.ones(366) / 100.0
    a[10:20] = 0.3
    snd = snd_series(a)
    out = xci.snow_cover_duration(snd)
    assert len(out) == 2
    assert out[0] == 10


def test_continous_snow_cover_start(snd_series):
    snd = snd_series(np.arange(366) / 100.0)
    out = xci.continuous_snow_cover_start(snd)
    assert len(out) == 2
    np.testing.assert_array_equal(out, [snd.time.dt.dayofyear[0].data + 2, np.nan])
    for attr in ["units", "is_dayofyear", "calendar"]:
        assert attr in out.attrs.keys()
    assert out.attrs["units"] == ""
    assert out.attrs["is_dayofyear"] == 1


def test_continuous_snow_cover_end(snd_series):
    a = np.concatenate(
        [
            np.zeros(100),
            np.arange(10),
            10 * np.ones(100),
            10 * np.arange(10)[::-1],
            np.zeros(146),
        ]
    )
    snd = snd_series(a / 100.0)
    out = xci.continuous_snow_cover_end(snd)
    assert len(out) == 2
    doy = snd.time.dt.dayofyear[0].data
    np.testing.assert_array_equal(out, [(doy + 219) % 366, np.nan])
    for attr in ["units", "is_dayofyear", "calendar"]:
        assert attr in out.attrs.keys()
    assert out.attrs["units"] == ""
    assert out.attrs["is_dayofyear"] == 1


def test_high_precip_low_temp(pr_series, tasmin_series):
    pr = pr_series([0, 1, 2, 0, 0])
    tas = tasmin_series(np.array([0, 0, 1, 1]) + K2C)

    out = xci.high_precip_low_temp(pr, tas, pr_thresh="1 kg m-2 s-1", tas_thresh="1 C")
    np.testing.assert_array_equal(out, [1])


def test_blowing_snow(snd_series, sfcWind_series):
    snd = snd_series([0, 0.1, 0.2, 0, 0, 0.1, 0.3, 0.5, 0.7, 0])
    w = sfcWind_series([9, 0, 0, 0, 0, 1, 1, 0, 5, 0])

    out = xci.blowing_snow(snd, w, snd_thresh="50 cm", sfcWind_thresh="4 km/h")
    np.testing.assert_array_equal(out, [1])


def test_winter_storm(snd_series):
    snd = snd_series([0, 0.5, 0.2, 0.7, 0, 0.4])
    out = xci.winter_storm(snd, thresh="30 cm")
    np.testing.assert_array_equal(out, [3])


def test_humidex(tas_series):

    tas = tas_series([15, 25, 35, 40])
    tas.attrs["units"] = "C"

    dtps = tas_series([10, 15, 25, 25])
    dtps.attrs["units"] = "C"

    # expected values from https://en.wikipedia.org/wiki/Humidex
    expected = np.array([16, 29, 47, 52]) * units.degC

    # Celcius
    hc = xci.humidex(tas, dtps)
    np.testing.assert_array_almost_equal(hc, expected, 0)

    # Kelvin
    hk = xci.humidex(convert_units_to(tas, "K"), dtps)
    np.testing.assert_array_almost_equal(hk, expected.to("K"), 0)

    # Fahrenheit
    hf = xci.humidex(convert_units_to(tas, "fahrenheit"), dtps)
    np.testing.assert_array_almost_equal(hf, expected.to("fahrenheit"), 0)

    # With relative humidity
    hurs = xci.relative_humidity(tas, dtps, method="bohren98")
    hr = xci.humidex(tas, hurs=hurs)
    np.testing.assert_array_almost_equal(hr, expected, 0)

    # With relative humidity and Kelvin
    hk = xci.humidex(convert_units_to(tas, "K"), hurs=hurs)
    np.testing.assert_array_almost_equal(hk, expected.to("K"), 0)


@pytest.mark.parametrize(
    "op,exp", [("max", 11), ("sum", 21), ("count", 3), ("mean", 7)]
)
def test_freezethaw_spell(tasmin_series, tasmax_series, op, exp):
    tmin = np.ones(365)
    tmax = np.ones(365)

    tmin[3:5] = -1
    tmin[10:15] = -1
    tmin[20:31] = -1
    tmin[50:55] = -1

    tasmax = tasmax_series(tmax + K2C)
    tasmin = tasmin_series(tmin + K2C)

    out = xci.multiday_temperature_swing(
        tasmin=tasmin, tasmax=tasmax, freq="AS-JUL", window=3, op=op
    )
    np.testing.assert_array_equal(out, exp)


def test_wind_chill(tas_series, sfcWind_series):
    tas = tas_series(np.array([-1, -10, -20, 10, -15]) + K2C)
    sfcWind = sfcWind_series([10, 60, 20, 6, 2])

    out = xci.wind_chill_index(tas=tas, sfcWind=sfcWind)
    # Expected values taken from the online calculator of the ECCC.
    # The calculator was altered to remove the rounding of the output.
    np.testing.assert_allclose(
        out,
        [-4.509267062481955, -22.619869069856854, -30.478945408950928, np.NaN, -16.443],
    )

    out = xci.wind_chill_index(tas=tas, sfcWind=sfcWind, method="US")
    assert out[-1].isnull()


class TestClausiusClapeyronScaledPrecip:
    def test_simple(self):
        pr_baseline = xr.DataArray(
            np.arange(4).reshape(1, 2, 2),
            dims=["time", "lat", "lon"],
            coords={"time": [1], "lat": [-45, 45], "lon": [30, 60]},
            attrs={"units": "mmday"},
        )
        tas_baseline = xr.DataArray(
            np.arange(4).reshape(1, 2, 2),
            dims=["time", "lat", "lon"],
            coords={"time": [1], "lat": [-45, 45], "lon": [30, 60]},
            attrs={"units": "C"},
        )
        tas_future = xr.DataArray(
            np.arange(40).reshape(10, 2, 2),
            dims=["time_fut", "lat", "lon"],
            coords={"time_fut": np.arange(10), "lat": [-45, 45], "lon": [30, 60]},
            attrs={"units": "C"},
        )
        delta_tas = tas_future - tas_baseline
        delta_tas.attrs["units"] = "delta_degC"
        out = xci.clausius_clapeyron_scaled_precipitation(delta_tas, pr_baseline)

        np.testing.assert_allclose(
            out.isel(time=0),
            [
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [
                        1.0,
                        1.31079601,
                        1.71818618,
                        2.25219159,
                        2.95216375,
                        3.86968446,
                        5.07236695,
                        6.64883836,
                        8.7152708,
                        11.42394219,
                    ],
                ],
                [
                    [
                        2.0,
                        2.62159202,
                        3.43637236,
                        4.50438318,
                        5.9043275,
                        7.73936892,
                        10.14473391,
                        13.29767673,
                        17.4305416,
                        22.84788438,
                    ],
                    [
                        3.0,
                        3.93238803,
                        5.15455854,
                        6.75657477,
                        8.85649125,
                        11.60905339,
                        15.21710086,
                        19.94651509,
                        26.1458124,
                        34.27182657,
                    ],
                ],
            ],
        )

    def test_workflow(self, tas_series, pr_series):
        """Test typical workflow."""
        n = int(365.25 * 10)
        tref = tas_series(np.random.rand(n), start="1961-01-01")
        tfut = tas_series(np.random.rand(n) + 2, start="2051-01-01")
        pr = pr_series(np.random.rand(n) * 10, start="1961-01-01")

        # Compute climatologies
        with xr.set_options(keep_attrs=True):
            tref_m = tref.mean(dim="time")
            tfut_m = tfut.mean(dim="time")
            pr_m = pr.mean(dim="time")

        delta_tas = tfut_m - tref_m
        delta_tas.attrs["units"] = "delta_degC"
        pr_m_cc = xci.clausius_clapeyron_scaled_precipitation(delta_tas, pr_m)
        np.testing.assert_array_almost_equal(pr_m_cc, pr_m * 1.07 ** 2, 1)

        # Compute monthly climatologies
        with xr.set_options(keep_attrs=True):
            tref_mm = tref.groupby("time.month").mean()
            tfut_mm = tfut.groupby("time.month").mean()
            pr_mm = pr.groupby("time.month").mean()

        delta_tas_m = tfut_mm - tref_mm
        delta_tas_m.attrs["units"] = "delta_degC"

        pr_mm_cc = xci.clausius_clapeyron_scaled_precipitation(delta_tas_m, pr_mm)
        np.testing.assert_array_almost_equal(pr_mm_cc, pr_mm * 1.07 ** 2, 1)


class TestPotentialEvapotranspiration:
    def test_baier_robertson(self, tasmin_series, tasmax_series):
        tn = tasmin_series(np.array([0, 5, 10]) + 273.15)
        tn = tn.expand_dims(lat=[45])
        tx = tasmax_series(np.array([10, 15, 20]) + 273.15)
        tx = tx.expand_dims(lat=[45])

        out = xci.potential_evapotranspiration(tn, tx, method="BR65")
        np.testing.assert_allclose(out[0, 2], [3.861079 / 86400], rtol=1e-2)

    def test_hargreaves(self, tasmin_series, tasmax_series, tas_series):
        tn = tasmin_series(np.array([0, 5, 10]) + 273.15)
        tn = tn.expand_dims(lat=[45])
        tx = tasmax_series(np.array([10, 15, 20]) + 273.15)
        tx = tx.expand_dims(lat=[45])
        tm = tas_series(np.array([5, 10, 15]) + 273.15)
        tm = tm.expand_dims(lat=[45])

        out = xci.potential_evapotranspiration(tn, tx, tm, method="HG85")
        np.testing.assert_allclose(out[2, 0], [3.962589 / 86400], rtol=1e-2)

    def test_thornthwaite(self, tas_series):
        time_std = date_range(
            "1990-01-01", "1990-12-01", freq="MS", calendar="standard"
        )
        tm = xr.DataArray(
            np.ones((time_std.size, 1)),
            dims=("time", "lat"),
            coords={"time": time_std, "lat": [45]},
            attrs={"units": "degC"},
        )

        out = xci.potential_evapotranspiration(tas=tm, method="TW48")
        np.testing.assert_allclose(out[0, 1], [42.7619242 / (86400 * 30)], rtol=1e-1)


def test_water_budget(pr_series, tasmin_series, tasmax_series):
    pr = pr_series(np.array([10, 10, 10]))
    pr.attrs["units"] = "mm/day"
    pr = pr.expand_dims(lat=[45])
    tn = tasmin_series(np.array([0, 5, 10]) + K2C)
    tn = tn.expand_dims(lat=[45])
    tx = tasmax_series(np.array([10, 15, 20]) + K2C)
    tx = tx.expand_dims(lat=[45])

    out = xci.water_budget(pr, tn, tx, method="BR65")
    np.testing.assert_allclose(out[0, 2], [6.138921 / 86400], rtol=1e-3)

    out = xci.water_budget(pr, tn, tx, method="HG85")
    np.testing.assert_allclose(out[0, 2], [6.037411 / 86400], rtol=1e-3)

    time_std = date_range("1990-01-01", "1990-12-01", freq="MS", calendar="standard")
    tm = xr.DataArray(
        np.ones((time_std.size, 1)),
        dims=("time", "lat"),
        coords={"time": time_std, "lat": [45]},
        attrs={"units": "degC"},
    )
    prm = xr.DataArray(
        np.ones((time_std.size, 1)) * 10,
        dims=("time", "lat"),
        coords={"time": time_std, "lat": [45]},
        attrs={"units": "mm/day"},
    )

    out = xci.water_budget(prm, tas=tm, method="TW48")
    np.testing.assert_allclose(out[1, 0], [8.5746025 / 86400], rtol=1e-1)


def test_dry_spell(pr_series):
    pr = pr_series(
        np.array(
            [
                1.01,
                1.01,
                1.01,
                1.01,
                1.01,
                1.01,
                0.01,
                0.01,
                0.01,
                0.51,
                0.51,
                0.75,
                0.75,
                0.51,
                0.01,
                0.01,
                0.01,
                1.01,
                1.01,
                1.01,
            ]
        )
    )
    pr.attrs["units"] = "mm/day"

    events = xci.dry_spell_frequency(pr, thresh="3 mm", window=7, freq="YS")
    total_d = xci.dry_spell_total_length(pr, thresh="3 mm", window=7, freq="YS")

    np.testing.assert_allclose(events[0], [2], rtol=1e-1)
    np.testing.assert_allclose(total_d[0], [12], rtol=1e-1)

import os

import numpy as np
import pytest
import xarray as xr

from xclim import atmos
from xclim.core.calendar import percentile_doy
from xclim.core.options import set_options
from xclim.core.units import convert_units_to
from xclim.testing import open_dataset

K2C = 273.15


class TestCSDI:
    def test_simple(self, tasmin_series):
        i = 3650
        A = 10.0
        tn = (
            np.zeros(i)
            + A * np.sin(np.arange(i) / 365.0 * 2 * np.pi)
            + 0.1 * np.random.rand(i)
        )
        tn += K2C
        tn[10:20] -= 2
        tn = tasmin_series(tn)
        tn10 = percentile_doy(tn, per=10).sel(percentiles=10)

        out = atmos.cold_spell_duration_index(tn, tn10, freq="AS-JUL")
        assert out[0] == 10

    def test_convert_units(self, tasmin_series):
        i = 3650
        A = 10.0
        tn = (
            np.zeros(i)
            + A * np.sin(np.arange(i) / 365.0 * 2 * np.pi)
            + 0.1 * np.random.rand(i)
        )
        tn[10:20] -= 2
        tn = tasmin_series(tn + K2C)
        tn.attrs["units"] = "C"
        tn10 = percentile_doy(tn, per=10).sel(percentiles=10)

        out = atmos.cold_spell_duration_index(tn, tn10, freq="AS-JUL")
        assert out[0] == 10

    def test_nan_presence(self, tasmin_series):
        i = 3650
        A = 10.0
        tn = (
            np.zeros(i)
            + K2C
            + A * np.sin(np.arange(i) / 365.0 * 2 * np.pi)
            + 0.1 * np.random.rand(i)
        )
        tn[10:20] -= 2
        tn[9] = np.nan
        tn = tasmin_series(tn)
        tn10 = percentile_doy(tn, per=10).sel(percentiles=10)

        out = atmos.cold_spell_duration_index(tn, tn10, freq="AS-JUL")
        assert np.isnan(out[0])


class TestDTR:
    nc_tasmax = os.path.join("NRCANdaily", "nrcan_canada_daily_tasmax_1990.nc")
    nc_tasmin = os.path.join("NRCANdaily", "nrcan_canada_daily_tasmin_1990.nc")

    def test_DTR_3d_data_with_nans(self):
        tasmax = open_dataset(self.nc_tasmax).tasmax
        tasmax_C = open_dataset(self.nc_tasmax).tasmax
        tasmax_C -= K2C
        tasmax_C.attrs["units"] = "C"
        tasmin = open_dataset(self.nc_tasmin).tasmin
        tasmin_C = open_dataset(self.nc_tasmin).tasmin
        tasmin_C -= K2C
        tasmin_C.attrs["units"] = "C"
        # put a nan somewhere
        tasmin.values[32, 1, 0] = np.nan
        tasmin_C.values[32, 1, 0] = np.nan
        dtr = atmos.daily_temperature_range(tasmin, tasmax, freq="MS")
        dtrC = atmos.daily_temperature_range(tasmin_C, tasmax_C, freq="MS")
        min1 = tasmin.values[:, 0, 0]
        max1 = tasmax.values[:, 0, 0]

        dtr1 = max1 - min1

        np.testing.assert_array_equal(dtr, dtrC)
        assert dtr.attrs["units"] == "K"
        assert np.allclose(dtr1[0:31].mean(), dtr.values[0, 0, 0])

        assert np.isnan(dtr.values[1, 1, 0])

        assert np.isnan(dtr.values[0, -1, -1])

        dtr = atmos.max_daily_temperature_range(tasmin, tasmax, freq="MS")
        dtrC = atmos.max_daily_temperature_range(tasmin_C, tasmax_C, freq="MS")
        np.testing.assert_array_equal(dtr, dtrC)
        assert dtr.attrs["units"] == "K"
        assert np.allclose(dtr1[0:31].max(), dtr.values[0, 0, 0])
        assert np.isnan(dtr.values[1, 1, 0])
        assert np.isnan(dtr.values[0, -1, -1])


class TestDTRVar:
    nc_tasmax = os.path.join("NRCANdaily", "nrcan_canada_daily_tasmax_1990.nc")
    nc_tasmin = os.path.join("NRCANdaily", "nrcan_canada_daily_tasmin_1990.nc")

    def test_dtr_var_3d_data_with_nans(self):
        tasmax = open_dataset(self.nc_tasmax).tasmax
        tasmax_C = open_dataset(self.nc_tasmax).tasmax
        tasmax_C -= K2C
        tasmax_C.attrs["units"] = "C"
        tasmin = open_dataset(self.nc_tasmin).tasmin
        tasmin_C = open_dataset(self.nc_tasmin).tasmin
        tasmin_C -= K2C
        tasmin_C.attrs["units"] = "C"
        # put a nan somewhere
        tasmin.values[32, 1, 0] = np.nan
        tasmin_C.values[32, 1, 0] = np.nan
        dtr = atmos.daily_temperature_range_variability(tasmin, tasmax, freq="MS")
        dtrC = atmos.daily_temperature_range_variability(tasmin_C, tasmax_C, freq="MS")
        min1 = tasmin.values[:, 0, 0]
        max1 = tasmax.values[:, 0, 0]
        assert dtr.attrs["units"] == "K"
        dtr1a = max1 - min1
        dtr1 = abs(np.diff(dtr1a))
        np.testing.assert_array_equal(dtr, dtrC)

        # first month jan use 0:30 (n==30) because of day to day diff
        assert np.allclose(dtr1[0:30].mean(), dtr.values[0, 0, 0])

        assert np.isnan(dtr.values[1, 1, 0])

        assert np.isnan(dtr.values[0, -1, -1])


class TestETR:
    nc_tasmax = os.path.join("NRCANdaily", "nrcan_canada_daily_tasmax_1990.nc")
    nc_tasmin = os.path.join("NRCANdaily", "nrcan_canada_daily_tasmin_1990.nc")

    def test_dtr_var_3d_data_with_nans(self):
        tasmax = open_dataset(self.nc_tasmax).tasmax
        tasmax_C = open_dataset(self.nc_tasmax).tasmax
        tasmax_C -= K2C
        tasmax_C.attrs["units"] = "C"
        tasmin = open_dataset(self.nc_tasmin).tasmin
        tasmin_C = open_dataset(self.nc_tasmin).tasmin
        tasmin_C -= K2C
        tasmin_C.attrs["units"] = "C"
        # put a nan somewhere
        tasmin.values[32, 1, 0] = np.nan
        tasmin_C.values[32, 1, 0] = np.nan

        etr = atmos.extreme_temperature_range(tasmin, tasmax, freq="MS")
        etrC = atmos.extreme_temperature_range(tasmin_C, tasmax_C, freq="MS")
        min1 = tasmin.values[:, 0, 0]
        max1 = tasmax.values[:, 0, 0]

        np.testing.assert_array_equal(etr, etrC)

        etr1 = max1[0:31].max() - min1[0:31].min()
        assert np.allclose(etr1, etr.values[0, 0, 0])

        assert np.isnan(etr.values[1, 1, 0])

        assert np.isnan(etr.values[0, -1, -1])


class TestTmean:
    nc_files = (
        os.path.join("NRCANdaily", "nrcan_canada_daily_tasmax_1990.nc"),
        os.path.join("NRCANdaily", "nrcan_canada_daily_tasmin_1990.nc"),
    )

    def test_Tmean_3d_data(self):
        ds_tmax = open_dataset(self.nc_files[0])
        ds_tmin = open_dataset(self.nc_files[1])
        tas = atmos.tg(ds_tmin.tasmin, ds_tmax.tasmax)
        tas_C = atmos.tg(ds_tmin.tasmin, ds_tmax.tasmax)
        tas_C.values -= K2C
        tas_C.attrs["units"] = "C"
        # put a nan somewhere
        tas.values[180, 1, 0] = np.nan
        tas_C.values[180, 1, 0] = np.nan
        tmmean = atmos.tg_mean(tas)
        tmmeanC = atmos.tg_mean(tas_C)
        x1 = tas.values[:, 0, 0]
        tmmean1 = x1.mean()

        # TODO: Investigate the differences between the two outputs.
        # The conversion to K is done after / before the mean.
        np.testing.assert_array_almost_equal(tmmeanC, tmmean, 3)
        # test single point vs manual
        assert np.allclose(tmmean1, tmmean.values[0, 0, 0], tmmeanC.values[0, 0, 0])
        # test single nan point
        assert np.isnan(tmmean.values[0, 1, 0])
        # test all nan point
        assert np.isnan(tmmean.values[0, -1, -1])


class TestTx:
    nc_file = os.path.join("NRCANdaily", "nrcan_canada_daily_tasmax_1990.nc")

    def test_TX_3d_data(self):
        tasmax = open_dataset(self.nc_file).tasmax
        tasmax_C = open_dataset(self.nc_file).tasmax
        tasmax_C.values -= K2C
        tasmax_C.attrs["units"] = "C"
        # put a nan somewhere
        tasmax.values[180, 1, 0] = np.nan
        tasmax_C.values[180, 1, 0] = np.nan
        txmean = atmos.tx_mean(tasmax)
        txmax = atmos.tx_max(tasmax)
        txmin = atmos.tx_min(tasmax)

        txmeanC = atmos.tx_mean(tasmax_C)
        txmaxC = atmos.tx_max(tasmax_C)
        txminC = atmos.tx_min(tasmax_C)

        no_nan = (
            ~np.isnan(txmean).values & ~np.isnan(txmax).values & ~np.isnan(txmin).values
        )

        # test maxes always greater than mean and mean always greater than min (non nan values only)
        assert np.all(txmax.values[no_nan] > txmean.values[no_nan]) & np.all(
            txmean.values[no_nan] > txmin.values[no_nan]
        )

        np.testing.assert_array_almost_equal(txmeanC, txmean, 3)
        np.testing.assert_array_equal(txminC, txmin)
        np.testing.assert_array_equal(txmaxC, txmax)
        x1 = tasmax.values[:, 0, 0]
        txmean1 = x1.mean()
        txmin1 = x1.min()
        txmax1 = x1.max()

        # test single point vs manual
        assert np.allclose(txmean1, txmean.values[0, 0, 0], txmeanC.values[0, 0, 0])
        assert np.allclose(txmax1, txmax.values[0, 0, 0], txmaxC.values[0, 0, 0])
        assert np.allclose(txmin1, txmin.values[0, 0, 0], txminC.values[0, 0, 0])
        # test single nan point
        assert np.isnan(txmean.values[0, 1, 0])
        assert np.isnan(txmin.values[0, 1, 0])
        assert np.isnan(txmax.values[0, 1, 0])
        # test all nan point
        assert np.isnan(txmean.values[0, -1, -1])
        assert np.isnan(txmin.values[0, -1, -1])
        assert np.isnan(txmax.values[0, -1, -1])


class TestTn:
    nc_file = os.path.join("NRCANdaily", "nrcan_canada_daily_tasmin_1990.nc")

    def test_TN_3d_data(self):
        tasmin = open_dataset(self.nc_file).tasmin
        tasmin_C = open_dataset(self.nc_file).tasmin
        tasmin_C.values -= K2C
        tasmin_C.attrs["units"] = "C"
        # put a nan somewhere
        tasmin.values[180, 1, 0] = np.nan
        tasmin_C.values[180, 1, 0] = np.nan
        tnmean = atmos.tn_mean(tasmin)
        tnmax = atmos.tn_max(tasmin)
        tnmin = atmos.tn_min(tasmin)

        tnmeanC = atmos.tn_mean(tasmin_C)
        tnmaxC = atmos.tn_max(tasmin_C)
        tnminC = atmos.tn_min(tasmin_C)

        no_nan = (
            ~np.isnan(tnmean).values & ~np.isnan(tnmax).values & ~np.isnan(tnmin).values
        )

        # test maxes always greater than mean and mean alwyas greater than min (non nan values only)
        assert np.all(tnmax.values[no_nan] > tnmean.values[no_nan]) & np.all(
            tnmean.values[no_nan] > tnmin.values[no_nan]
        )

        np.testing.assert_array_almost_equal(tnmeanC, tnmean, 3)
        np.testing.assert_array_equal(tnminC, tnmin)
        np.testing.assert_array_equal(tnmaxC, tnmax)

        x1 = tasmin.values[:, 0, 0]
        txmean1 = x1.mean()
        txmin1 = x1.min()
        txmax1 = x1.max()

        # test single point vs manual
        assert np.allclose(txmean1, tnmean.values[0, 0, 0], tnmeanC.values[0, 0, 0])
        assert np.allclose(txmax1, tnmax.values[0, 0, 0], tnmaxC.values[0, 0, 0])
        assert np.allclose(txmin1, tnmin.values[0, 0, 0], tnminC.values[0, 0, 0])
        # test single nan point
        assert np.isnan(tnmean.values[0, 1, 0])
        assert np.isnan(tnmin.values[0, 1, 0])
        assert np.isnan(tnmax.values[0, 1, 0])
        # test all nan point
        assert np.isnan(tnmean.values[0, -1, -1])
        assert np.isnan(tnmin.values[0, -1, -1])
        assert np.isnan(tnmax.values[0, -1, -1])


class TestConsecutiveFrostDays:
    def test_one_freeze_day(self, tasmin_series):
        a = np.zeros(365) + K2C + 5.0
        a[2] -= 20

        ts = tasmin_series(a)
        out = atmos.consecutive_frost_days(ts)
        np.testing.assert_array_equal(out, [1])

    def test_three_freeze_day(self, tasmin_series):
        a = np.zeros(365) + K2C + 5.0
        a[2:5] -= 20

        ts = tasmin_series(a)
        out = atmos.consecutive_frost_days(ts)
        np.testing.assert_array_equal(out, [3])

    def test_two_equal_freeze_day(self, tasmin_series):
        a = np.zeros(365) + K2C + 5.0
        a[2:5] -= 20
        a[6:9] -= 20
        ts = tasmin_series(a)
        out = atmos.consecutive_frost_days(ts)
        np.testing.assert_array_equal(out, [3])

    def test_two_events_freeze_day(self, tasmin_series):
        a = np.zeros(365) + K2C + 5.0
        a[2:5] -= 20
        a[6:10] -= 20
        ts = tasmin_series(a)
        out = atmos.consecutive_frost_days(ts)
        np.testing.assert_array_equal(out, [4])

    def test_convert_units_freeze_day(self, tasmin_series):
        a = np.zeros(365) + 5.0
        a[2:5] -= 20
        a[6:10] -= 20
        ts = tasmin_series(a)
        ts.attrs["units"] = "C"
        out = atmos.consecutive_frost_days(ts)
        np.testing.assert_array_equal(out, [4])

    def test_one_nan_day(self, tasmin_series):
        a = np.zeros(365) + K2C + 5.0
        a[2] -= 20
        a[-1] = np.nan

        ts = tasmin_series(a)
        out = atmos.consecutive_frost_days(ts)
        np.testing.assert_array_equal(out, [np.nan])


class TestFrostSeasonLength:
    def test_simple(self, tasmin_series):
        a = np.zeros(730) + K2C + 15
        a[300:400] = K2C - 5
        a[404:407] = K2C - 5
        tasmin = tasmin_series(a, start="2000-01-01")
        # Default, window = 5, mid_date = 07-01, freq= AS-JUL
        out = atmos.frost_season_length(tasmin=tasmin)
        np.testing.assert_array_equal(out, [np.nan, 107, np.nan])

        out = atmos.frost_season_length(tasmin=tasmin, window=3)
        np.testing.assert_array_equal(out, [np.nan, 100, np.nan])

        out = atmos.frost_season_length(tasmin=tasmin, mid_date="07-01", freq="YS")
        np.testing.assert_array_equal(out, [np.nan, np.nan])


class TestColdSpellDays:
    def test_simple(self, tas_series):
        a = np.zeros(365) + K2C
        a[10:20] -= 15  # 10 days
        a[40:43] -= 50  # too short -> 0
        a[80:100] -= 30  # at the end and beginning
        ts = tas_series(a)
        out = atmos.cold_spell_days(ts, thresh="-10 C", freq="MS")
        np.testing.assert_array_equal(out, [10, 0, 12, 8, 0, 0, 0, 0, 0, 0, 0, 0])
        out = atmos.cold_spell_frequency(ts, thresh="-10 C", freq="MS")
        np.testing.assert_array_equal(out, [1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0])

    def test_convert_units(self, tas_series):
        a = np.zeros(365)
        a[10:20] -= 15  # 10 days
        a[40:43] -= 50  # too short -> 0
        a[80:100] -= 30  # at the end and beginning
        ts = tas_series(a)
        ts.attrs["units"] = "C"
        out = atmos.cold_spell_days(ts, thresh="-10 C", freq="MS")
        np.testing.assert_array_equal(out, [10, 0, 12, 8, 0, 0, 0, 0, 0, 0, 0, 0])
        out = atmos.cold_spell_frequency(ts, thresh="-10 C", freq="MS")
        np.testing.assert_array_equal(out, [1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0])

    def test_nan_presence(self, tas_series):
        a = np.zeros(365) + K2C
        a[10:20] -= 15  # 10 days
        a[40:43] -= 50  # too short -> 0
        a[80:100] -= 30  # at the end and beginning
        a[-1] = np.nan
        ts = tas_series(a)

        out = atmos.cold_spell_days(ts, thresh="-10 C", freq="MS")
        np.testing.assert_array_equal(out, [10, 0, 12, 8, 0, 0, 0, 0, 0, 0, 0, np.nan])
        out = atmos.cold_spell_frequency(ts, thresh="-10 C", freq="MS")
        np.testing.assert_array_equal(out, [1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, np.nan])


class TestFrostDays:
    nc_file = os.path.join("NRCANdaily", "nrcan_canada_daily_tasmin_1990.nc")

    def test_3d_data_with_nans(self):
        # test with 3d data
        tasmin = open_dataset(self.nc_file).tasmin
        tasminC = open_dataset(self.nc_file).tasmin
        tasminC -= K2C
        tasminC.attrs["units"] = "C"
        # put a nan somewhere
        tasmin.values[180, 1, 0] = np.nan
        tasminC.values[180, 1, 0] = np.nan
        # compute with both skipna options
        thresh = 273.16
        fd = atmos.frost_days(tasmin, freq="YS")
        fdC = atmos.frost_days(tasminC, freq="YS")
        # fds = xci.frost_days(tasmin, thresh=thresh, freq='YS', skipna=True)

        x1 = tasmin.values[:, 0, 0]

        fd1 = (x1[x1 < thresh]).size

        np.testing.assert_array_equal(fd, fdC)

        assert np.allclose(fd1, fd.values[0, 0, 0])
        # assert (np.allclose(fd1, fds.values[0, 0, 0]))
        assert np.isnan(fd.values[0, 1, 0])
        # assert (np.allclose(fd2, fds.values[0, 1, 0]))
        assert np.isnan(fd.values[0, -1, -1])
        # assert (np.isnan(fds.values[0, -1, -1]))


class TestIceDays:
    nc_file = os.path.join("NRCANdaily", "nrcan_canada_daily_tasmax_1990.nc")

    def test_3d_data_with_nans(self):
        # test with 3d data
        tas = open_dataset(self.nc_file).tasmax
        tasC = open_dataset(self.nc_file).tasmax
        tasC -= K2C
        tasC.attrs["units"] = "C"
        # put a nan somewhere
        tas.values[180, 1, 0] = np.nan
        tasC.values[180, 1, 0] = np.nan
        # compute with both skipna options
        thresh = 273.16
        fd = atmos.ice_days(tas, freq="YS")
        fdC = atmos.ice_days(tasC, freq="YS")

        x1 = tas.values[:, 0, 0]

        fd1 = (x1[x1 < thresh]).size

        np.testing.assert_array_equal(fd, fdC)

        assert np.allclose(fd1, fd.values[0, 0, 0])

        assert np.isnan(fd.values[0, 1, 0])

        assert np.isnan(fd.values[0, -1, -1])


class TestCoolingDegreeDays:
    nc_file = os.path.join("NRCANdaily", "nrcan_canada_daily_tasmax_1990.nc")

    def test_3d_data_with_nans(self):
        # test with 3d data
        tas = open_dataset(self.nc_file).tasmax
        tas.attrs["cell_methods"] = "time: mean within days"
        # put a nan somewhere
        tas.values[180, 1, 0] = np.nan

        # compute with both skipna options
        thresh = 18 + K2C
        cdd = atmos.cooling_degree_days(tas, thresh="18 C", freq="YS")

        x1 = tas.values[:, 0, 0]

        cdd1 = (x1[x1 > thresh] - thresh).sum()

        assert np.allclose(cdd1, cdd.values[0, 0, 0])

        assert np.isnan(cdd.values[0, 1, 0])

        assert np.isnan(cdd.values[0, -1, -1])

    def test_convert_units(self):
        # test with 3d data
        tas = open_dataset(self.nc_file).tasmax
        tas.values -= K2C
        tas.attrs["units"] = "C"
        tas.attrs["cell_methods"] = "time: mean within days"
        # put a nan somewhere
        tas.values[180, 1, 0] = np.nan

        # compute with both skipna options
        thresh = 18
        cdd = atmos.cooling_degree_days(tas, thresh="18 C", freq="YS")

        x1 = tas.values[:, 0, 0]
        # x2 = tas.values[:, 1, 0]

        cdd1 = (x1[x1 > thresh] - thresh).sum()
        # gdd2 = (x2[x2 > thresh] - thresh).sum()

        assert np.allclose(cdd1, cdd.values[0, 0, 0])
        # assert (np.allclose(gdd1, gdds.values[0, 0, 0]))
        assert np.isnan(cdd.values[0, 1, 0])
        # assert (np.allclose(gdd2, gdds.values[0, 1, 0]))
        assert np.isnan(cdd.values[0, -1, -1])
        # assert (np.isnan(gdds.values[0, -1, -1]))


class TestHeatingDegreeDays:
    nc_file = os.path.join("NRCANdaily", "nrcan_canada_daily_tasmax_1990.nc")

    def test_3d_data_with_nans(self):
        # test with 3d data
        tas = open_dataset(self.nc_file).tasmax
        # put a nan somewhere
        tas.values[180, 1, 0] = np.nan
        tas.attrs["cell_methods"] = "time: mean within days"
        # compute with both skipna options
        thresh = 17 + K2C
        hdd = atmos.heating_degree_days(tas, freq="YS")

        x1 = tas.values[:, 0, 0]

        hdd1 = (thresh - x1).clip(min=0).sum()

        assert np.allclose(hdd1, hdd.values[0, 0, 0])

        assert np.isnan(hdd.values[0, 1, 0])

        assert np.isnan(hdd.values[0, -1, -1])

    def test_convert_units(self):
        # test with 3d data
        tas = open_dataset(self.nc_file).tasmax
        # put a nan somewhere
        tas.values[180, 1, 0] = np.nan
        tas.values -= K2C
        tas.attrs["units"] = "C"
        tas.attrs["cell_methods"] = "time: mean within days"
        # compute with both skipna options
        thresh = 17
        hdd = atmos.heating_degree_days(tas, freq="YS")

        x1 = tas.values[:, 0, 0]

        hdd1 = (thresh - x1).clip(min=0).sum()

        assert np.allclose(hdd1, hdd.values[0, 0, 0])

        assert np.isnan(hdd.values[0, 1, 0])

        assert np.isnan(hdd.values[0, -1, -1])


class TestGrowingDegreeDays:
    nc_file = os.path.join("NRCANdaily", "nrcan_canada_daily_tasmax_1990.nc")

    def test_3d_data_with_nans(self):
        # test with 3d data
        tas = open_dataset(self.nc_file).tasmax
        tas.attrs["cell_methods"] = "time: mean within days"
        # put a nan somewhere
        tas.values[180, 1, 0] = np.nan

        # compute with both skipna options
        thresh = K2C + 4
        gdd = atmos.growing_degree_days(tas, freq="YS")
        # gdds = xci.growing_degree_days(tas, thresh=thresh, freq='YS', skipna=True)

        x1 = tas.values[:, 0, 0]
        # x2 = tas.values[:, 1, 0]

        gdd1 = (x1[x1 > thresh] - thresh).sum()
        # gdd2 = (x2[x2 > thresh] - thresh).sum()

        assert np.allclose(gdd1, gdd.values[0, 0, 0])

        assert np.isnan(gdd.values[0, 1, 0])

        assert np.isnan(gdd.values[0, -1, -1])


class TestHeatWaveFrequency:
    def test_1d(self, tasmax_series, tasmin_series):
        tn1 = np.zeros(366)
        tx1 = np.zeros(366)
        tn1[:10] = np.array([20, 23, 23, 23, 23, 21, 23, 23, 23, 23])
        tx1[:10] = np.array([29, 31, 31, 31, 29, 31, 31, 31, 31, 31])

        tn = tasmin_series(tn1 + K2C, start="1/1/2000")
        tx = tasmax_series(tx1 + K2C, start="1/1/2000")
        tnC = tasmin_series(tn1, start="1/1/2000")
        tnC.attrs["units"] = "C"
        txC = tasmax_series(tx1, start="1/1/2000")
        txC.attrs["units"] = "C"

        hwf = atmos.heat_wave_frequency(
            tn, tx, thresh_tasmin="22 C", thresh_tasmax="30 C", freq="YS"
        )
        hwfC = atmos.heat_wave_frequency(
            tnC, txC, thresh_tasmin="22 C", thresh_tasmax="30 C", freq="YS"
        )
        np.testing.assert_array_equal(hwf, hwfC)
        np.testing.assert_allclose(hwf.values[:1], 2)

        hwf = atmos.heat_wave_frequency(
            tn, tx, thresh_tasmin="22 C", thresh_tasmax="30 C", window=4, freq="YS"
        )
        np.testing.assert_allclose(hwf.values[:1], 1)

        # one long hw
        hwf = atmos.heat_wave_frequency(
            tn, tx, thresh_tasmin="10 C", thresh_tasmax="10 C", freq="YS"
        )
        np.testing.assert_allclose(hwf.values[:1], 1)
        # no hw
        hwf = atmos.heat_wave_frequency(
            tn, tx, thresh_tasmin="40 C", thresh_tasmax="40 C", freq="YS"
        )
        np.testing.assert_allclose(hwf.values[:1], 0)


class TestHeatWaveMaxLength:
    def test_1d(self, tasmax_series, tasmin_series):
        tn1 = np.zeros(366)
        tx1 = np.zeros(366)
        tn1[:10] = np.array([20, 23, 23, 23, 23, 21, 23, 23, 23, 23])
        tx1[:10] = np.array([29, 31, 31, 31, 29, 31, 31, 31, 31, 31])

        tn = tasmin_series(tn1 + K2C, start="1/1/2000")
        tx = tasmax_series(tx1 + K2C, start="1/1/2000")
        tnC = tasmin_series(tn1, start="1/1/2000")
        tnC.attrs["units"] = "C"
        txC = tasmax_series(tx1, start="1/1/2000")
        txC.attrs["units"] = "C"

        hwf = atmos.heat_wave_max_length(
            tn, tx, thresh_tasmin="22 C", thresh_tasmax="30 C", freq="YS"
        )
        hwfC = atmos.heat_wave_max_length(
            tnC, txC, thresh_tasmin="22 C", thresh_tasmax="30 C", freq="YS"
        )
        np.testing.assert_array_equal(hwf, hwfC)
        np.testing.assert_allclose(hwf.values[:1], 4)

        hwf = atmos.heat_wave_max_length(
            tn, tx, thresh_tasmin="20 C", thresh_tasmax="30 C", window=4, freq="YS"
        )
        np.testing.assert_allclose(hwf.values[:1], 5)

        # one long hw
        hwf = atmos.heat_wave_max_length(
            tn, tx, thresh_tasmin="10 C", thresh_tasmax="10 C", freq="YS"
        )
        np.testing.assert_allclose(hwf.values[:1], 10)
        # no hw
        hwf = atmos.heat_wave_max_length(
            tn, tx, thresh_tasmin="40 C", thresh_tasmax="40 C", freq="YS"
        )
        np.testing.assert_allclose(hwf.values[:1], 0)


class TestHeatWaveTotalLength:
    def test_1d(self, tasmax_series, tasmin_series):
        tn1 = np.zeros(366)
        tx1 = np.zeros(366)
        tn1[:10] = np.array([20, 23, 23, 23, 23, 21, 23, 23, 23, 23])
        tx1[:10] = np.array([29, 31, 31, 31, 29, 31, 31, 31, 31, 31])

        tn = tasmin_series(tn1 + K2C, start="1/1/2000")
        tx = tasmax_series(tx1 + K2C, start="1/1/2000")
        tnC = tasmin_series(tn1, start="1/1/2000")
        tnC.attrs["units"] = "C"
        txC = tasmax_series(tx1, start="1/1/2000")
        txC.attrs["units"] = "C"

        hwf = atmos.heat_wave_total_length(
            tn, tx, thresh_tasmin="22 C", thresh_tasmax="30 C", freq="YS"
        )
        hwfC = atmos.heat_wave_total_length(
            tnC, txC, thresh_tasmin="22 C", thresh_tasmax="30 C", freq="YS"
        )
        np.testing.assert_array_equal(hwf, hwfC)
        np.testing.assert_allclose(hwf.values[:1], 7)

        hwf = atmos.heat_wave_total_length(
            tn, tx, thresh_tasmin="20 C", thresh_tasmax="30 C", window=4, freq="YS"
        )
        np.testing.assert_allclose(hwf.values[:1], 5)

        # one long hw
        hwf = atmos.heat_wave_total_length(
            tn, tx, thresh_tasmin="10 C", thresh_tasmax="10 C", freq="YS"
        )
        np.testing.assert_allclose(hwf.values[:1], 10)
        # no hw
        hwf = atmos.heat_wave_total_length(
            tn, tx, thresh_tasmin="40 C", thresh_tasmax="40 C", freq="YS"
        )
        np.testing.assert_allclose(hwf.values[:1], 0)


class TestHeatWaveIndex:
    def test_simple(self, tasmax_series):
        tx = np.zeros(366)
        tx[:10] = np.array([29, 31, 31, 31, 29, 31, 31, 31, 31, 31])
        tx = tasmax_series(tx + K2C, start="1/1/2000")
        hwi = atmos.heat_wave_index(tx, freq="YS")
        np.testing.assert_array_equal(hwi, [10])

    def test_convert_units(self, tasmax_series):
        tx = np.zeros(366)
        tx[:10] = np.array([29, 31, 31, 31, 29, 31, 31, 31, 31, 31])
        tx = tasmax_series(tx, start="1/1/2000")
        tx.attrs["units"] = "C"
        hwi = atmos.heat_wave_index(tx, freq="YS")
        np.testing.assert_array_equal(hwi, [10])

    def test_nan_presence(self, tasmax_series):
        tx = np.zeros(366)
        tx[:10] = np.array([29, 31, 31, 31, 29, 31, 31, 31, 31, 31])
        tx[-1] = np.nan
        tx = tasmax_series(tx + K2C, start="1/1/2000")

        hwi = atmos.heat_wave_index(tx, freq="YS")
        np.testing.assert_array_equal(hwi, [np.nan])


class TestDailyFreezeThaw:
    nc_tasmax = os.path.join("NRCANdaily", "nrcan_canada_daily_tasmax_1990.nc")
    nc_tasmin = os.path.join("NRCANdaily", "nrcan_canada_daily_tasmin_1990.nc")

    def test_3d_data_with_nans(self):
        tasmax = open_dataset(self.nc_tasmax).tasmax
        tasmin = open_dataset(self.nc_tasmin).tasmin

        # put a nan somewhere
        tasmin.values[180, 1, 0] = np.nan

        frzthw = atmos.daily_freezethaw_cycles(tasmin, tasmax, freq="YS")

        min1 = tasmin.values[:, 0, 0]
        max1 = tasmax.values[:, 0, 0]

        frzthw1 = ((min1 < K2C) * (max1 > K2C) * 1.0).sum()

        np.testing.assert_allclose(frzthw1, frzthw.values[0, 0, 0])

        assert np.isnan(frzthw.values[0, 1, 0])

        assert np.isnan(frzthw.values[0, -1, -1])

    def test_convert_units(self):
        tasmax = open_dataset(self.nc_tasmax).tasmax
        tasmin = open_dataset(self.nc_tasmin).tasmin
        tasmax.values -= K2C
        tasmax.attrs["units"] = "C"
        tasmin.values -= K2C
        tasmin.attrs["units"] = "C"
        # put a nan somewhere
        tasmin.values[180, 1, 0] = np.nan

        with pytest.warns(None) as record:
            frzthw = atmos.daily_freezethaw_cycles(
                tasmin,
                tasmax,
                thresh_tasmin="0 degC",
                thresh_tasmax="0 degC",
                freq="YS",
            )

        min1 = tasmin.values[:, 0, 0]
        max1 = tasmax.values[:, 0, 0]

        frzthw1 = (((min1 < 0) & (max1 > 0)) * 1.0).sum()

        assert (
            "This index calculation will soon require user-specified thresholds."
            not in [str(q.message) for q in record]
        )

        np.testing.assert_allclose(frzthw1, frzthw.values[0, 0, 0])

        assert np.isnan(frzthw.values[0, 1, 0])

        assert np.isnan(frzthw.values[0, -1, -1])


class TestGrowingSeasonLength:
    @pytest.mark.parametrize("chunks", [None, {"time": 183.0}])
    def test_single_year(self, tas_series, chunks):
        a = np.zeros(366) + K2C
        ts = tas_series(a, start="1/1/2000")
        tt = (ts.time.dt.month >= 5) & (ts.time.dt.month <= 8)
        offset = np.random.uniform(low=5.5, high=23, size=(tt.sum().values,))
        ts[tt] = ts[tt] + offset
        if chunks:
            ts = ts.chunk(chunks)

        out = atmos.growing_season_length(ts)

        np.testing.assert_array_equal(out, tt.sum())

    def test_convert_units(self, tas_series):
        a = np.zeros(366)

        ts = tas_series(a, start="1/1/2000")
        ts.attrs["units"] = "C"
        tt = (ts.time.dt.month >= 5) & (ts.time.dt.month <= 8)
        offset = np.random.uniform(low=5.5, high=23, size=(tt.sum().values,))
        ts[tt] = ts[tt] + offset

        out = atmos.growing_season_length(ts)

        np.testing.assert_array_equal(out, tt.sum())

    def test_nan_presence(self, tas_series):
        a = np.zeros(366)
        a[50] = np.nan
        ts = tas_series(a, start="1/1/2000")
        ts.attrs["units"] = "C"
        tt = (ts.time.dt.month >= 5) & (ts.time.dt.month <= 8)

        offset = np.random.uniform(low=5.5, high=23, size=(tt.sum().values,))
        ts[tt] = ts[tt] + offset

        out = atmos.growing_season_length(ts)

        np.testing.assert_array_equal(out, [np.nan])

    def test_multiyear(self, tas_series):
        a = np.zeros(366 * 10)

        ts = tas_series(a, start="1/1/2000")
        ts.attrs["units"] = "C"
        tt = (ts.time.dt.month >= 5) & (ts.time.dt.month <= 8)

        offset = np.random.uniform(low=5.5, high=23, size=(tt.sum().values,))
        ts[tt] = ts[tt] + offset

        out = atmos.growing_season_length(ts)

        np.testing.assert_array_equal(out[3], tt[0:366].sum().values)


class TestTnDaysBelow:
    nc_file = os.path.join("NRCANdaily", "nrcan_canada_daily_tasmin_1990.nc")

    def test_3d_data_with_nans(self):
        # test with 3d data
        tas = open_dataset(self.nc_file).tasmin
        tasC = open_dataset(self.nc_file).tasmin
        tasC -= K2C
        tasC.attrs["units"] = "C"
        # put a nan somewhere
        tas.values[180, 1, 0] = np.nan
        tasC.values[180, 1, 0] = np.nan
        # compute with both skipna options
        thresh = 273.16 + -10
        fd = atmos.tn_days_below(tas, thresh="-10 degC", freq="YS")
        fdC = atmos.tn_days_below(tasC, thresh="-10 degC", freq="YS")

        x1 = tas.values[:, 0, 0]

        fd1 = (x1[x1 < thresh]).size

        np.testing.assert_array_equal(fd, fdC)

        assert np.allclose(fd1, fd.values[0, 0, 0])

        assert np.isnan(fd.values[0, 1, 0])

        assert np.isnan(fd.values[0, -1, -1])


class TestTxDaysAbove:
    nc_file = os.path.join("NRCANdaily", "nrcan_canada_daily_tasmax_1990.nc")

    def test_3d_data_with_nans(self):
        # test with 3d data
        tas = open_dataset(self.nc_file).tasmax
        tasC = open_dataset(self.nc_file).tasmax
        tasC -= K2C
        tasC.attrs["units"] = "C"
        # put a nan somewhere
        tas.values[180, 1, 0] = np.nan
        tasC.values[180, 1, 0] = np.nan
        # compute with both skipna options
        thresh = 273.16 + 25
        fd = atmos.tx_days_above(tas, freq="YS")
        fdC = atmos.tx_days_above(tasC, freq="YS")

        x1 = tas.values[:, 0, 0]

        fd1 = (x1[x1 > thresh]).size

        np.testing.assert_array_equal(fd, fdC)

        assert np.allclose(fd1, fd.values[0, 0, 0])

        assert np.isnan(fd.values[0, 1, 0])

        assert np.isnan(fd.values[0, -1, -1])


class TestTnDaysAbove:
    nc_file = os.path.join("NRCANdaily", "nrcan_canada_daily_tasmin_1990.nc")

    @pytest.mark.parametrize(
        "tn_indice, kwargs",
        [("tn_days_above", dict(thresh="20 degC")), ("tropical_nights", dict())],
    )
    def test_3d_data_with_nans(self, tn_indice, kwargs):
        # test with 3d data
        tas = open_dataset(self.nc_file).tasmin
        tasC = open_dataset(self.nc_file).tasmin
        tasC -= K2C
        tasC.attrs["units"] = "C"
        # put a nan somewhere
        tas.values[180, 1, 0] = np.nan
        tasC.values[180, 1, 0] = np.nan
        # compute with both skipna options
        thresh = 273.16 + 20

        out = getattr(atmos, tn_indice)(tas, **kwargs, freq="YS")
        outC = getattr(atmos, tn_indice)(tasC, **kwargs, freq="YS")
        # fds = xci.frost_days(tasmin, thresh=thresh, freq='YS', skipna=True)

        x1 = tas.values[:, 0, 0]

        out1 = (x1[x1 > thresh]).size

        np.testing.assert_array_equal(out, outC)

        assert np.allclose(out1, out.values[0, 0, 0])
        assert np.isnan(out.values[0, 1, 0])
        assert np.isnan(out.values[0, -1, -1])


class TestTxTnDaysAbove:
    nc_tasmax = os.path.join("NRCANdaily", "nrcan_canada_daily_tasmax_1990.nc")
    nc_tasmin = os.path.join("NRCANdaily", "nrcan_canada_daily_tasmin_1990.nc")

    def test_3d_data_with_nans(self):
        tasmax = open_dataset(self.nc_tasmax).tasmax
        tasmin = open_dataset(self.nc_tasmin).tasmin

        tasmaxC = open_dataset(self.nc_tasmax).tasmax
        tasminC = open_dataset(self.nc_tasmin).tasmin
        tasmaxC -= K2C
        tasmaxC.attrs["units"] = "C"
        tasminC -= K2C
        tasminC.attrs["units"] = "C"

        # put a nan somewhere
        tasmin.values[180, 1, 0] = np.nan
        tasminC.values[180, 1, 0] = np.nan

        out = atmos.tx_tn_days_above(
            tasmin, tasmax, thresh_tasmax="25 C", thresh_tasmin="18 C"
        )
        outC = atmos.tx_tn_days_above(
            tasminC, tasmaxC, thresh_tasmax="25 C", thresh_tasmin="18 C"
        )
        np.testing.assert_array_equal(out, outC)

        min1 = tasmin.values[:, 53, 76]
        max1 = tasmax.values[:, 53, 76]

        out1 = ((min1 > (K2C + 18)) * (max1 > (K2C + 25)) * 1.0).sum()

        assert np.allclose(out1, out.values[0, 53, 76])

        assert np.isnan(out.values[0, 1, 0])

        assert np.isnan(out.values[0, -1, -1])


class TestT90p:
    def test_tg90p_simple(self, tas_series):
        i = 366
        arr = np.asarray(np.arange(i), "float")
        tas = tas_series(arr, start="1/1/2000")
        tasC = tas.copy()
        tasC -= K2C
        tasC.attrs["units"] = "C"
        t90 = percentile_doy(tas, window=1, per=90).sel(percentiles=90)

        # create cold spell in june
        tas[175:180] = 1
        tasC[175:180] = 1 - K2C
        out = atmos.tg90p(tas, t90, freq="MS")
        outC = atmos.tg90p(tasC, t90, freq="MS")

        np.testing.assert_array_equal(out, outC)
        assert out[0] == 30
        assert out[1] == 29
        assert out[5] == 25

        # nan treatment
        tas[33] = np.nan
        tasC[33] = np.nan
        out = atmos.tg90p(tas, t90, freq="MS")
        outC = atmos.tg90p(tasC, t90, freq="MS")

        np.testing.assert_array_equal(out, outC)
        assert out[0] == 30
        assert np.isnan(out[1])
        assert out[5] == 25

    def test_tn90p_simple(self, tasmin_series):
        i = 366
        arr = np.asarray(np.arange(i), "float")
        tas = tasmin_series(arr, start="1/1/2000")
        tasC = tas.copy()
        tasC -= K2C
        tasC.attrs["units"] = "C"
        t90 = percentile_doy(tas, window=1, per=90).sel(percentiles=90)

        # create cold spell in june
        tas[175:180] = 1
        tasC[175:180] = 1 - K2C
        out = atmos.tn90p(tas, t90, freq="MS")
        outC = atmos.tn90p(tasC, t90, freq="MS")

        np.testing.assert_array_equal(out, outC)
        assert out[0] == 30
        assert out[1] == 29
        assert out[5] == 25

        # nan treatment
        tas[33] = np.nan
        tasC[33] = np.nan
        out = atmos.tn90p(tas, t90, freq="MS")
        outC = atmos.tn90p(tasC, t90, freq="MS")

        np.testing.assert_array_equal(out, outC)
        assert out[0] == 30
        assert np.isnan(out[1])
        assert out[5] == 25

    def test_tx90p_simple(self, tasmax_series):
        i = 366
        arr = np.asarray(np.arange(i), "float")
        tas = tasmax_series(arr, start="1/1/2000")
        tasC = tas.copy()
        tasC -= K2C
        tasC.attrs["units"] = "C"
        t90 = percentile_doy(tas, window=1, per=90).sel(percentiles=90)

        # create cold spell in june
        tas[175:180] = 1
        tasC[175:180] = 1 - K2C
        out = atmos.tx90p(tas, t90, freq="MS")
        outC = atmos.tx90p(tasC, t90, freq="MS")

        np.testing.assert_array_equal(out, outC)
        assert out[0] == 30
        assert out[1] == 29
        assert out[5] == 25

        # nan treatment
        tas[33] = np.nan
        tasC[33] = np.nan
        out = atmos.tx90p(tas, t90, freq="MS")
        outC = atmos.tx90p(tasC, t90, freq="MS")

        np.testing.assert_array_equal(out, outC)
        assert out[0] == 30
        assert np.isnan(out[1])
        assert out[5] == 25


class TestT10p:
    def test_tg10p_simple(self, tas_series):
        i = 366
        arr = np.asarray(np.arange(i), "float")
        tas = tas_series(arr, start="1/1/2000")
        tasC = tas.copy()
        tasC -= K2C
        tasC.attrs["units"] = "C"
        t10 = percentile_doy(tas, per=10).sel(percentiles=10)

        # create cold spell in june
        tas[175:180] = 1
        tasC[175:180] = 1 - K2C
        out = atmos.tg10p(tas, t10, freq="MS")
        outC = atmos.tg10p(tasC, t10, freq="MS")

        np.testing.assert_array_equal(out, outC)

        assert out[0] == 0
        assert out[5] == 5

        # nan treatment
        tas[33] = np.nan
        tasC[33] = np.nan
        out = atmos.tg10p(tas, t10, freq="MS")
        outC = atmos.tg10p(tasC, t10, freq="MS")

        np.testing.assert_array_equal(out, outC)
        assert out[0] == 0
        assert np.isnan(out[1])
        assert out[5] == 5

    def test_tn10p_simple(self, tasmin_series):
        i = 366
        arr = np.asarray(np.arange(i), "float")
        tas = tasmin_series(arr, start="1/1/2000")
        tasC = tas.copy()
        tasC -= K2C
        tasC.attrs["units"] = "C"
        t10 = percentile_doy(tas, per=10).sel(percentiles=10)

        # create cold spell in june
        tas[175:180] = 1
        tasC[175:180] = 1 - K2C
        out = atmos.tn10p(tas, t10, freq="MS")
        outC = atmos.tn10p(tasC, t10, freq="MS")

        np.testing.assert_array_equal(out, outC)
        assert out[0] == 0
        assert out[5] == 5

        # nan treatment
        tas[33] = np.nan
        tasC[33] = np.nan
        out = atmos.tn10p(tas, t10, freq="MS")
        outC = atmos.tn10p(tasC, t10, freq="MS")

        np.testing.assert_array_equal(out, outC)
        assert out[0] == 0
        assert np.isnan(out[1])
        assert out[5] == 5

    def test_tx10p_simple(self, tasmax_series):
        i = 366
        arr = np.asarray(np.arange(i), "float")
        tas = tasmax_series(arr, start="1/1/2000")
        tasC = tas.copy()
        tasC -= K2C
        tasC.attrs["units"] = "C"
        t10 = percentile_doy(tas, per=10).sel(percentiles=10)

        # create cold spell in june
        tas[175:180] = 1
        tasC[175:180] = 1 - K2C
        out = atmos.tx10p(tas, t10, freq="MS")
        outC = atmos.tx10p(tasC, t10, freq="MS")

        np.testing.assert_array_equal(out, outC)
        assert out[0] == 0
        assert out[5] == 5

        # nan treatment
        tas[33] = np.nan
        tasC[33] = np.nan
        out = atmos.tx10p(tas, t10, freq="MS")
        outC = atmos.tx10p(tasC, t10, freq="MS")

        np.testing.assert_array_equal(out, outC)
        assert out[0] == 0
        assert np.isnan(out[1])
        assert out[5] == 5


def test_freshet_start(tas_series):
    out = atmos.freshet_start(
        tas_series(np.arange(-50, 350) + 274, start="1/1/2000"), freq="YS"
    )
    assert out[0] == 51


def test_degree_days_exceedance_date():
    tas = open_dataset("FWI/GFWED_sample_2017.nc").tas
    tas.attrs.update(
        cell_methods="time: mean within days", standard_name="air_temperature"
    )

    out = atmos.degree_days_exceedance_date(
        tas=tas,
        thresh="4 degC",
        op=">",
        sum_thresh="200 K days",
    )
    np.testing.assert_array_equal(out, np.array([[153, 136, 9, 6]]).T)
    assert "tmean > 4 degc" in out.attrs["description"]

    with set_options(check_missing="skip"):
        out = atmos.degree_days_exceedance_date(
            tas=tas,
            thresh="4 degC",
            op=">",
            sum_thresh="1500 K days",
            after_date="07-02",
            freq="YS",
        )
        np.testing.assert_array_equal(out, np.array([[np.nan, 280, 241, 244]]).T)


def test_warm_spell_duration_index():
    tasmax = open_dataset("ERA5/daily_surface_cancities_1990-1993.nc").tasmax
    tx90 = percentile_doy(tasmax, window=5, per=90)

    out = atmos.warm_spell_duration_index(
        tasmax=tasmax, tx90=tx90, window=3, freq="AS-JUL"
    )
    np.testing.assert_array_equal(out[0, :, 0], np.array([np.nan, 3, 0, 0, np.nan]))
    assert (
        "Annual total number of days within spells of at least 3 days"
        in out.description
    )


def test_maximum_consecutive_warm_days():
    tasmax = open_dataset("ERA5/daily_surface_cancities_1990-1993.nc").tasmax
    out = atmos.maximum_consecutive_warm_days(tasmax)
    np.testing.assert_array_equal(out[1, :], np.array([13, 21, 6, 10]))
    assert (
        "Annual longest spell of consecutive days with tmax above 25 degc."
        in out.description
    )


def test_corn_heat_units():
    tn = open_dataset("ERA5/daily_surface_cancities_1990-1993.nc").tasmin
    tx = open_dataset("ERA5/daily_surface_cancities_1990-1993.nc").tasmax

    with xr.set_options(keep_attrs=True):
        tnC = tn - K2C
        tnC.attrs["units"] = "C"

    chu = atmos.corn_heat_units(
        tasmin=tn, tasmax=tx, thresh_tasmin="4.44 degC", thresh_tasmax="10 degC"
    )
    chuC = atmos.corn_heat_units(
        tasmin=tnC, tasmax=tx, thresh_tasmin="4.44 degC", thresh_tasmax="10 degC"
    )

    np.testing.assert_allclose(chu, chuC, rtol=1e-3)

    np.testing.assert_allclose(
        chu[0, 180:185], np.array([13.777, 12.368, 11.966, 14.674, 16.797]), rtol=1e-4
    )

    assert (
        "specific thresholds : tmin > 4.44 degc and tmax > 10 degc." in chu.description
    )


def test_freezethaw_spell_frequency():
    ds = open_dataset("ERA5/daily_surface_cancities_1990-1993.nc")

    out = atmos.freezethaw_spell_frequency(
        tasmin=ds.tasmin, tasmax=ds.tasmax, freq="YS"
    )
    np.testing.assert_array_equal(out.isel(location=0), [32, 38, 37, 30])

    # At location -1, year 2 has no spells of length >=2
    out = atmos.freezethaw_spell_frequency(
        tasmin=convert_units_to(ds.tasmin, "degF"),
        tasmax=ds.tasmax,
        window=2,
        freq="YS",
    )
    np.testing.assert_array_equal(out.isel(location=-1), [1, 0, 1, 1])

    assert out.attrs["long_name"] == "Annual number of freeze-thaw spells."


def test_freezethaw_spell_mean_length():
    ds = open_dataset("ERA5/daily_surface_cancities_1990-1993.nc")

    out = atmos.freezethaw_spell_mean_length(
        tasmin=ds.tasmin, tasmax=ds.tasmax, freq="YS"
    )
    np.testing.assert_allclose(out.isel(location=0), [2.09375, 2, 1.8648648, 1.7666666])

    # At location -1, year 2 has no spells of length >=2
    out = atmos.freezethaw_spell_mean_length(
        tasmin=convert_units_to(ds.tasmin, "degF"),
        tasmax=ds.tasmax,
        window=2,
        freq="YS",
    )
    np.testing.assert_array_equal(out.isel(location=-1), [2, 0, 2, 2])

    assert out.attrs["long_name"] == "Annual average length of freeze-thaw spells."


def test_freezethaw_spell_max_length():
    ds = open_dataset("ERA5/daily_surface_cancities_1990-1993.nc")

    out = atmos.freezethaw_spell_max_length(
        tasmin=ds.tasmin, tasmax=ds.tasmax, freq="YS"
    )
    np.testing.assert_array_equal(out.isel(location=0), [12, 7, 7, 4])

    # At location -1, year 2 has no spells of length >=2
    out = atmos.freezethaw_spell_max_length(
        tasmin=convert_units_to(ds.tasmin, "degF"),
        tasmax=ds.tasmax,
        window=2,
        freq="YS",
    )
    np.testing.assert_array_equal(out.isel(location=-1), [2, 0, 2, 2])

    assert out.attrs["long_name"] == "Annual maximal length of freeze-thaw spells."

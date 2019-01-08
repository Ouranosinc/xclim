import os

import numpy as np
import xarray as xr

import xclim.temperature as temp
from xclim.testing.common import tas_series, tasmin_series, tasmax_series
from xclim.utils import percentile_doy

TESTS_HOME = os.path.abspath(os.path.dirname(__file__))
TESTS_DATA = os.path.join(TESTS_HOME, 'testdata')

TAS_SERIES = tas_series
TASMIN_SERIES = tasmin_series
TASMAX_SERIES = tasmax_series

K2C = 273.15


class TestCSDI:
    def test_simple(self, tasmin_series):
        i = 3650
        A = 10.
        tn = np.zeros(i) + A * np.sin(np.arange(i) / 365. * 2 * np.pi) + .1 * np.random.rand(i)
        tn += K2C
        tn[10:20] -= 2
        tn = tasmin_series(tn)
        tn10 = percentile_doy(tn, per=.1)

        out = temp.cold_spell_duration_index(tn, tn10, freq='AS-JUL')
        assert out[0] == 10

    def test_convert_units(self, tasmin_series):
        i = 3650
        A = 10.
        tn = np.zeros(i) + A * np.sin(np.arange(i) / 365. * 2 * np.pi) + .1 * np.random.rand(i)
        tn[10:20] -= 2
        tn = tasmin_series(tn)
        tn.attrs['units'] = 'C'
        tn10 = percentile_doy(tn + K2C, per=.1)

        out = temp.cold_spell_duration_index(tn, tn10, freq='AS-JUL')
        assert out[0] == 10

    def test_nan_presence(self, tasmin_series):
        i = 3650
        A = 10.
        tn = np.zeros(i) + K2C + A * np.sin(np.arange(i) / 365. * 2 * np.pi) + .1 * np.random.rand(i)
        tn[10:20] -= 2
        tn[9] = np.nan
        tn = tasmin_series(tn)
        tn10 = percentile_doy(tn, per=.1)

        out = temp.cold_spell_duration_index(tn, tn10, freq='AS-JUL')
        assert np.isnan(out[0])


class TestDTR:
    nc_tasmax = os.path.join(TESTS_DATA, 'NRCANdaily', 'nrcan_canada_daily_tasmax_1990.nc')
    nc_tasmin = os.path.join(TESTS_DATA, 'NRCANdaily', 'nrcan_canada_daily_tasmin_1990.nc')

    def test_DTR_3d_data_with_nans(self):
        tasmax = xr.open_dataset(self.nc_tasmax).tasmax
        tasmax_C = xr.open_dataset(self.nc_tasmax).tasmax
        tasmax_C -= K2C
        tasmax_C.attrs['units'] = 'C'
        tasmin = xr.open_dataset(self.nc_tasmin).tasmin
        tasmin_C = xr.open_dataset(self.nc_tasmin).tasmin
        tasmin_C -= K2C
        tasmin_C.attrs['units'] = 'C'
        # put a nan somewhere
        tasmin.values[32, 1, 0] = np.nan
        tasmin_C.values[32, 1, 0] = np.nan
        dtr = temp.daily_temperature_range(tasmax, tasmin, freq='MS')
        dtrC = temp.daily_temperature_range(tasmax_C, tasmin_C, freq='MS')
        min1 = tasmin.values[:, 0, 0]
        max1 = tasmax.values[:, 0, 0]

        dtr1 = (max1 - min1)

        np.testing.assert_array_equal(dtr, dtrC)

        assert (np.allclose(dtr1[0:31].mean(), dtr.values[0, 0, 0], dtrC.values[0, 0, 0]))

        assert (np.isnan(dtr.values[1, 1, 0]))

        assert (np.isnan(dtr.values[0, -1, -1]))


class TestDTRVar:
    nc_tasmax = os.path.join(TESTS_DATA, 'NRCANdaily', 'nrcan_canada_daily_tasmax_1990.nc')
    nc_tasmin = os.path.join(TESTS_DATA, 'NRCANdaily', 'nrcan_canada_daily_tasmin_1990.nc')

    def test_dtr_var_3d_data_with_nans(self):
        tasmax = xr.open_dataset(self.nc_tasmax).tasmax
        tasmax_C = xr.open_dataset(self.nc_tasmax).tasmax
        tasmax_C -= K2C
        tasmax_C.attrs['units'] = 'C'
        tasmin = xr.open_dataset(self.nc_tasmin).tasmin
        tasmin_C = xr.open_dataset(self.nc_tasmin).tasmin
        tasmin_C -= K2C
        tasmin_C.attrs['units'] = 'C'
        # put a nan somewhere
        tasmin.values[32, 1, 0] = np.nan
        tasmin_C.values[32, 1, 0] = np.nan
        dtr = temp.daily_temperature_range_variability(tasmax, tasmin, freq='MS')
        dtrC = temp.daily_temperature_range_variability(tasmax_C, tasmin_C, freq='MS')
        min1 = tasmin.values[:, 0, 0]
        max1 = tasmax.values[:, 0, 0]

        dtr1a = (max1 - min1)
        dtr1 = abs(np.diff(dtr1a))
        np.testing.assert_array_equal(dtr, dtrC)

        # first month jan use 0:30 (n==30) because of day to day diff
        assert (np.allclose(dtr1[0:30].mean(), dtr.values[0, 0, 0], dtrC.values[0, 0, 0]))

        assert (np.isnan(dtr.values[1, 1, 0]))

        assert (np.isnan(dtr.values[0, -1, -1]))


class TestETR:
    nc_tasmax = os.path.join(TESTS_DATA, 'NRCANdaily', 'nrcan_canada_daily_tasmax_1990.nc')
    nc_tasmin = os.path.join(TESTS_DATA, 'NRCANdaily', 'nrcan_canada_daily_tasmin_1990.nc')

    def test_dtr_var_3d_data_with_nans(self):
        tasmax = xr.open_dataset(self.nc_tasmax).tasmax
        tasmax_C = xr.open_dataset(self.nc_tasmax).tasmax
        tasmax_C -= K2C
        tasmax_C.attrs['units'] = 'C'
        tasmin = xr.open_dataset(self.nc_tasmin).tasmin
        tasmin_C = xr.open_dataset(self.nc_tasmin).tasmin
        tasmin_C -= K2C
        tasmin_C.attrs['units'] = 'C'
        # put a nan somewhere
        tasmin.values[32, 1, 0] = np.nan
        tasmin_C.values[32, 1, 0] = np.nan

        etr = temp.extreme_temperature_range(tasmax, tasmin, freq='MS')
        etrC = temp.extreme_temperature_range(tasmax_C, tasmin_C, freq='MS')
        min1 = tasmin.values[:, 0, 0]
        max1 = tasmax.values[:, 0, 0]

        np.testing.assert_array_equal(etr, etrC)

        etr1 = max1[0:31].max() - min1[0:31].min()
        assert (np.allclose(etr1, etr.values[0, 0, 0], etrC.values[0, 0, 0]))

        assert (np.isnan(etr.values[1, 1, 0]))

        assert (np.isnan(etr.values[0, -1, -1]))


class TestTmean:
    nc_file = os.path.join(TESTS_DATA, 'NRCANdaily', 'nrcan_canada_daily_tasmax_1990.nc')

    def test_Tmean_3d_data(self):
        tas = xr.open_dataset(self.nc_file).tasmax
        tas_C = xr.open_dataset(self.nc_file).tasmax
        tas_C.values -= K2C
        tas_C.attrs['units'] = 'C'
        # put a nan somewhere
        tas.values[180, 1, 0] = np.nan
        tas_C.values[180, 1, 0] = np.nan
        tmmean = temp.tg_mean(tas)
        tmmeanC = temp.tg_mean(tas_C)
        x1 = tas.values[:, 0, 0]
        tmmean1 = x1.mean()

        np.testing.assert_array_equal(tmmeanC, tmmean)
        # test single point vs manual
        assert (np.allclose(tmmean1, tmmean.values[0, 0, 0], tmmeanC.values[0, 0, 0]))
        # test single nan point
        assert (np.isnan(tmmean.values[0, 1, 0]))
        # test all nan point
        assert (np.isnan(tmmean.values[0, -1, -1]))


class TestTx:
    nc_file = os.path.join(TESTS_DATA, 'NRCANdaily', 'nrcan_canada_daily_tasmax_1990.nc')

    def test_TX_3d_data(self):
        tasmax = xr.open_dataset(self.nc_file).tasmax
        tasmax_C = xr.open_dataset(self.nc_file).tasmax
        tasmax_C.values -= K2C
        tasmax_C.attrs['units'] = 'C'
        # put a nan somewhere
        tasmax.values[180, 1, 0] = np.nan
        tasmax_C.values[180, 1, 0] = np.nan
        txmean = temp.tx_mean(tasmax)
        txmax = temp.tx_max(tasmax)
        txmin = temp.tx_min(tasmax)

        txmeanC = temp.tx_mean(tasmax_C)
        txmaxC = temp.tx_max(tasmax_C)
        txminC = temp.tx_min(tasmax_C)

        no_nan = ~np.isnan(txmean).values & ~np.isnan(txmax).values & ~np.isnan(txmin).values

        # test maxes always greater than mean and mean alwyas greater than min (non nan values only)
        assert (np.all(txmax.values[no_nan] > txmean.values[no_nan]) & np.all(
            txmean.values[no_nan] > txmin.values[no_nan]))

        np.testing.assert_array_equal(txmeanC, txmean)
        np.testing.assert_array_equal(txminC, txmin)
        np.testing.assert_array_equal(txmaxC, txmax)
        x1 = tasmax.values[:, 0, 0]
        txmean1 = x1.mean()
        txmin1 = x1.min()
        txmax1 = x1.max()

        # test single point vs manual
        assert (np.allclose(txmean1, txmean.values[0, 0, 0], txmeanC.values[0, 0, 0]))
        assert (np.allclose(txmax1, txmax.values[0, 0, 0], txmaxC.values[0, 0, 0]))
        assert (np.allclose(txmin1, txmin.values[0, 0, 0], txminC.values[0, 0, 0]))
        # test single nan point
        assert (np.isnan(txmean.values[0, 1, 0]))
        assert (np.isnan(txmin.values[0, 1, 0]))
        assert (np.isnan(txmax.values[0, 1, 0]))
        # test all nan point
        assert (np.isnan(txmean.values[0, -1, -1]))
        assert (np.isnan(txmin.values[0, -1, -1]))
        assert (np.isnan(txmax.values[0, -1, -1]))


class TestTn:
    nc_file = os.path.join(TESTS_DATA, 'NRCANdaily', 'nrcan_canada_daily_tasmin_1990.nc')

    def test_TN_3d_data(self):
        tasmin = xr.open_dataset(self.nc_file).tasmin
        tasmin_C = xr.open_dataset(self.nc_file).tasmin
        tasmin_C.values -= K2C
        tasmin_C.attrs['units'] = 'C'
        # put a nan somewhere
        tasmin.values[180, 1, 0] = np.nan
        tasmin_C.values[180, 1, 0] = np.nan
        tnmean = temp.tn_mean(tasmin)
        tnmax = temp.tn_max(tasmin)
        tnmin = temp.tn_min(tasmin)

        tnmeanC = temp.tn_mean(tasmin_C)
        tnmaxC = temp.tn_max(tasmin_C)
        tnminC = temp.tn_min(tasmin_C)

        no_nan = ~np.isnan(tnmean).values & ~np.isnan(tnmax).values & ~np.isnan(tnmin).values

        # test maxes always greater than mean and mean alwyas greater than min (non nan values only)
        assert (np.all(tnmax.values[no_nan] > tnmean.values[no_nan]) & np.all(
            tnmean.values[no_nan] > tnmin.values[no_nan]))

        np.testing.assert_array_equal(tnmeanC, tnmean)
        np.testing.assert_array_equal(tnminC, tnmin)
        np.testing.assert_array_equal(tnmaxC, tnmax)

        x1 = tasmin.values[:, 0, 0]
        txmean1 = x1.mean()
        txmin1 = x1.min()
        txmax1 = x1.max()

        # test single point vs manual
        assert (np.allclose(txmean1, tnmean.values[0, 0, 0], tnmeanC.values[0, 0, 0]))
        assert (np.allclose(txmax1, tnmax.values[0, 0, 0], tnmaxC.values[0, 0, 0]))
        assert (np.allclose(txmin1, tnmin.values[0, 0, 0], tnminC.values[0, 0, 0]))
        # test single nan point
        assert (np.isnan(tnmean.values[0, 1, 0]))
        assert (np.isnan(tnmin.values[0, 1, 0]))
        assert (np.isnan(tnmax.values[0, 1, 0]))
        # test all nan point
        assert (np.isnan(tnmean.values[0, -1, -1]))
        assert (np.isnan(tnmin.values[0, -1, -1]))
        assert (np.isnan(tnmax.values[0, -1, -1]))


class TestConsecutiveFrostDays:

    def test_one_freeze_day(self, tasmin_series):
        a = np.zeros(365) + K2C + 5.0
        a[2] -= 20

        ts = tasmin_series(a)
        out = temp.consecutive_frost_days(ts)
        np.testing.assert_array_equal(out, [1])

    def test_three_freeze_day(self, tasmin_series):
        a = np.zeros(365) + K2C + 5.0
        a[2:5] -= 20

        ts = tasmin_series(a)
        out = temp.consecutive_frost_days(ts)
        np.testing.assert_array_equal(out, [3])

    def test_two_equal_freeze_day(self, tasmin_series):
        a = np.zeros(365) + K2C + 5.0
        a[2:5] -= 20
        a[6:9] -= 20
        ts = tasmin_series(a)
        out = temp.consecutive_frost_days(ts)
        np.testing.assert_array_equal(out, [3])

    def test_two_events_freeze_day(self, tasmin_series):
        a = np.zeros(365) + K2C + 5.0
        a[2:5] -= 20
        a[6:10] -= 20
        ts = tasmin_series(a)
        out = temp.consecutive_frost_days(ts)
        np.testing.assert_array_equal(out, [4])

    def test_convert_units_freeze_day(self, tasmin_series):
        a = np.zeros(365) + 5.0
        a[2:5] -= 20
        a[6:10] -= 20
        ts = tasmin_series(a)
        ts.attrs['units'] = 'C'
        out = temp.consecutive_frost_days(ts)
        np.testing.assert_array_equal(out, [4])

    def test_one_nan_day(self, tasmin_series):
        a = np.zeros(365) + K2C + 5.0
        a[2] -= 20
        a[-1] = np.nan

        ts = tasmin_series(a)
        out = temp.consecutive_frost_days(ts)
        np.testing.assert_array_equal(out, [np.nan])


class TestColdSpellDays:

    def test_simple(self, tas_series):
        a = np.zeros(365) + K2C
        a[10:20] -= 15  # 10 days
        a[40:43] -= 50  # too short -> 0
        a[80:100] -= 30  # at the end and beginning
        ts = tas_series(a)
        out = temp.cold_spell_days(ts, thresh=-10, freq='MS')
        np.testing.assert_array_equal(out, [10, 0, 12, 8, 0, 0, 0, 0, 0, 0, 0, 0])

    def test_convert_units(self, tas_series):
        a = np.zeros(365)
        a[10:20] -= 15  # 10 days
        a[40:43] -= 50  # too short -> 0
        a[80:100] -= 30  # at the end and beginning
        ts = tas_series(a)
        ts.attrs['units'] = 'C'
        out = temp.cold_spell_days(ts, thresh=-10, freq='MS')
        np.testing.assert_array_equal(out, [10, 0, 12, 8, 0, 0, 0, 0, 0, 0, 0, 0])

    def test_nan_presence(self, tas_series):
        a = np.zeros(365) + K2C
        a[10:20] -= 15  # 10 days
        a[40:43] -= 50  # too short -> 0
        a[80:100] -= 30  # at the end and beginning
        a[-1] = np.nan
        ts = tas_series(a)

        out = temp.cold_spell_days(ts, thresh=-10, freq='MS')
        np.testing.assert_array_equal(out, [10, 0, 12, 8, 0, 0, 0, 0, 0, 0, 0, np.nan])


class TestFrostDays:
    nc_file = os.path.join(TESTS_DATA, 'NRCANdaily', 'nrcan_canada_daily_tasmin_1990.nc')

    def test_3d_data_with_nans(self):
        # test with 3d data
        tasmin = xr.open_dataset(self.nc_file).tasmin
        tasminC = xr.open_dataset(self.nc_file).tasmin
        tasminC -= K2C
        tasminC.attrs['units'] = 'C'
        # put a nan somewhere
        tasmin.values[180, 1, 0] = np.nan
        tasminC.values[180, 1, 0] = np.nan
        # compute with both skipna options
        thresh = 273.16
        fd = temp.frost_days(tasmin, freq='YS')
        fdC = temp.frost_days(tasminC, freq='YS')
        # fds = xci.frost_days(tasmin, thresh=thresh, freq='YS', skipna=True)

        x1 = tasmin.values[:, 0, 0]

        fd1 = (x1[x1 < thresh]).size

        np.testing.assert_array_equal(fd, fdC)

        assert (np.allclose(fd1, fd.values[0, 0, 0]))
        # assert (np.allclose(fd1, fds.values[0, 0, 0]))
        assert (np.isnan(fd.values[0, 1, 0]))
        # assert (np.allclose(fd2, fds.values[0, 1, 0]))
        assert (np.isnan(fd.values[0, -1, -1]))
        # assert (np.isnan(fds.values[0, -1, -1]))


class TestIceDays:
    nc_file = os.path.join(TESTS_DATA, 'NRCANdaily', 'nrcan_canada_daily_tasmax_1990.nc')

    def test_3d_data_with_nans(self):
        # test with 3d data
        tas = xr.open_dataset(self.nc_file).tasmax
        tasC = xr.open_dataset(self.nc_file).tasmax
        tasC -= K2C
        tasC.attrs['units'] = 'C'
        # put a nan somewhere
        tas.values[180, 1, 0] = np.nan
        tasC.values[180, 1, 0] = np.nan
        # compute with both skipna options
        thresh = 273.16
        fd = temp.ice_days(tas, freq='YS')
        fdC = temp.ice_days(tasC, freq='YS')

        x1 = tas.values[:, 0, 0]

        fd1 = (x1[x1 < thresh]).size

        np.testing.assert_array_equal(fd, fdC)

        assert (np.allclose(fd1, fd.values[0, 0, 0]))

        assert (np.isnan(fd.values[0, 1, 0]))

        assert (np.isnan(fd.values[0, -1, -1]))


class TestCoolingDegreeDays:
    nc_file = os.path.join(TESTS_DATA, 'NRCANdaily', 'nrcan_canada_daily_tasmax_1990.nc')

    def test_3d_data_with_nans(self):
        # test with 3d data
        tas = xr.open_dataset(self.nc_file).tasmax
        # put a nan somewhere
        tas.values[180, 1, 0] = np.nan

        # compute with both skipna options
        thresh = 18 + K2C
        cdd = temp.cooling_degree_days(tas, thresh=18, freq='YS')

        x1 = tas.values[:, 0, 0]

        cdd1 = (x1[x1 > thresh] - thresh).sum()

        assert (np.allclose(cdd1, cdd.values[0, 0, 0]))

        assert (np.isnan(cdd.values[0, 1, 0]))

        assert (np.isnan(cdd.values[0, -1, -1]))

    def test_convert_units(self):
        # test with 3d data
        tas = xr.open_dataset(self.nc_file).tasmax
        tas.values -= K2C
        tas.attrs['units'] = 'C'
        # put a nan somewhere
        tas.values[180, 1, 0] = np.nan

        # compute with both skipna options
        thresh = 18
        cdd = temp.cooling_degree_days(tas, thresh=18, freq='YS')

        x1 = tas.values[:, 0, 0]
        # x2 = tas.values[:, 1, 0]

        cdd1 = (x1[x1 > thresh] - thresh).sum()
        # gdd2 = (x2[x2 > thresh] - thresh).sum()

        assert (np.allclose(cdd1, cdd.values[0, 0, 0]))
        # assert (np.allclose(gdd1, gdds.values[0, 0, 0]))
        assert (np.isnan(cdd.values[0, 1, 0]))
        # assert (np.allclose(gdd2, gdds.values[0, 1, 0]))
        assert (np.isnan(cdd.values[0, -1, -1]))
        # assert (np.isnan(gdds.values[0, -1, -1]))


class TestHeatingDegreeDays:
    nc_file = os.path.join(TESTS_DATA, 'NRCANdaily', 'nrcan_canada_daily_tasmax_1990.nc')

    def test_3d_data_with_nans(self):
        # test with 3d data
        tas = xr.open_dataset(self.nc_file).tasmax
        # put a nan somewhere
        tas.values[180, 1, 0] = np.nan

        # compute with both skipna options
        thresh = 17 + K2C
        hdd = temp.heating_degree_days(tas, freq='YS')

        x1 = tas.values[:, 0, 0]

        hdd1 = (thresh - x1).clip(min=0).sum()

        assert (np.allclose(hdd1, hdd.values[0, 0, 0]))

        assert (np.isnan(hdd.values[0, 1, 0]))

        assert (np.isnan(hdd.values[0, -1, -1]))

    def test_convert_units(self):
        # test with 3d data
        tas = xr.open_dataset(self.nc_file).tasmax
        # put a nan somewhere
        tas.values[180, 1, 0] = np.nan
        tas.values -= K2C
        tas.attrs['units'] = 'C'
        # compute with both skipna options
        thresh = 17
        hdd = temp.heating_degree_days(tas, freq='YS')

        x1 = tas.values[:, 0, 0]

        hdd1 = (thresh - x1).clip(min=0).sum()

        assert (np.allclose(hdd1, hdd.values[0, 0, 0]))

        assert (np.isnan(hdd.values[0, 1, 0]))

        assert (np.isnan(hdd.values[0, -1, -1]))


class TestGrowingDegreeDays:
    nc_file = os.path.join(TESTS_DATA, 'NRCANdaily', 'nrcan_canada_daily_tasmax_1990.nc')

    def test_3d_data_with_nans(self):
        # test with 3d data
        tas = xr.open_dataset(self.nc_file).tasmax
        # put a nan somewhere
        tas.values[180, 1, 0] = np.nan

        # compute with both skipna options
        thresh = K2C + 4
        gdd = temp.growing_degree_days(tas, freq='YS')
        # gdds = xci.growing_degree_days(tas, thresh=thresh, freq='YS', skipna=True)

        x1 = tas.values[:, 0, 0]
        # x2 = tas.values[:, 1, 0]

        gdd1 = (x1[x1 > thresh] - thresh).sum()
        # gdd2 = (x2[x2 > thresh] - thresh).sum()

        assert (np.allclose(gdd1, gdd.values[0, 0, 0]))

        assert (np.isnan(gdd.values[0, 1, 0]))

        assert (np.isnan(gdd.values[0, -1, -1]))


class TestHeatWaveFrequency:
    def test_1d(self, tasmax_series, tasmin_series):
        tn1 = np.zeros(366)
        tx1 = np.zeros(366)
        tn1[:10] = np.array([20, 23, 23, 23, 23, 21, 23, 23, 23, 23])
        tx1[:10] = np.array([29, 31, 31, 31, 29, 31, 31, 31, 31, 31])

        tn = tasmin_series(tn1 + K2C, start='1/1/2000')
        tx = tasmax_series(tx1 + K2C, start='1/1/2000')
        tnC = tasmin_series(tn1, start='1/1/2000')
        tnC.attrs['units'] = 'C'
        txC = tasmax_series(tx1, start='1/1/2000')
        txC.attrs['units'] = 'C'

        hwf = temp.heat_wave_frequency(tn, tx, thresh_tasmin=22,
                                       thresh_tasmax=30)
        hwfC = temp.heat_wave_frequency(tnC, txC, thresh_tasmin=22,
                                        thresh_tasmax=30)
        np.testing.assert_array_equal(hwf, hwfC)
        np.testing.assert_allclose(hwf.values[:1], 2)

        hwf = temp.heat_wave_frequency(tn, tx, thresh_tasmin=22,
                                       thresh_tasmax=30, window=4)
        np.testing.assert_allclose(hwf.values[:1], 1)

        # one long hw
        hwf = temp.heat_wave_frequency(tn, tx, thresh_tasmin=10,
                                       thresh_tasmax=10)
        np.testing.assert_allclose(hwf.values[:1], 1)
        # no hw
        hwf = temp.heat_wave_frequency(tn, tx, thresh_tasmin=40,
                                       thresh_tasmax=40)
        np.testing.assert_allclose(hwf.values[:1], 0)


class TestHeatWaveMaxLength:
    def test_1d(self, tasmax_series, tasmin_series):
        tn1 = np.zeros(366)
        tx1 = np.zeros(366)
        tn1[:10] = np.array([20, 23, 23, 23, 23, 21, 23, 23, 23, 23])
        tx1[:10] = np.array([29, 31, 31, 31, 29, 31, 31, 31, 31, 31])

        tn = tasmin_series(tn1 + K2C, start='1/1/2000')
        tx = tasmax_series(tx1 + K2C, start='1/1/2000')
        tnC = tasmin_series(tn1, start='1/1/2000')
        tnC.attrs['units'] = 'C'
        txC = tasmax_series(tx1, start='1/1/2000')
        txC.attrs['units'] = 'C'

        hwf = temp.heat_wave_max_length(tn, tx, thresh_tasmin=22,
                                        thresh_tasmax=30)
        hwfC = temp.heat_wave_max_length(tnC, txC, thresh_tasmin=22,
                                         thresh_tasmax=30)
        np.testing.assert_array_equal(hwf, hwfC)
        np.testing.assert_allclose(hwf.values[:1], 4)

        hwf = temp.heat_wave_max_length(tn, tx, thresh_tasmin=20,
                                        thresh_tasmax=30, window=4)
        np.testing.assert_allclose(hwf.values[:1], 5)

        # one long hw
        hwf = temp.heat_wave_max_length(tn, tx, thresh_tasmin=10,
                                        thresh_tasmax=10)
        np.testing.assert_allclose(hwf.values[:1], 10)
        # no hw
        hwf = temp.heat_wave_max_length(tn, tx, thresh_tasmin=40,
                                        thresh_tasmax=40)
        np.testing.assert_allclose(hwf.values[:1], 0)


class TestHeatWaveIndex:

    def test_simple(self, tasmax_series):
        tx = np.zeros(366)
        tx[:10] = np.array([29, 31, 31, 31, 29, 31, 31, 31, 31, 31])
        tx = tasmax_series(tx + K2C, start='1/1/2000')
        hwi = temp.heat_wave_index(tx, freq='YS')
        np.testing.assert_array_equal(hwi, [10])

    def test_convert_units(self, tasmax_series):
        tx = np.zeros(366)
        tx[:10] = np.array([29, 31, 31, 31, 29, 31, 31, 31, 31, 31])
        tx = tasmax_series(tx, start='1/1/2000')
        tx.attrs['units'] = 'C'
        hwi = temp.heat_wave_index(tx, freq='YS')
        np.testing.assert_array_equal(hwi, [10])

    def test_nan_presence(self, tasmax_series):
        tx = np.zeros(366)
        tx[:10] = np.array([29, 31, 31, 31, 29, 31, 31, 31, 31, 31])
        tx[-1] = np.nan
        tx = tasmax_series(tx + K2C, start='1/1/2000')

        hwi = temp.heat_wave_index(tx, freq='YS')
        np.testing.assert_array_equal(hwi, [np.nan])


class TestDailyFreezeThaw:
    nc_tasmax = os.path.join(TESTS_DATA, 'NRCANdaily', 'nrcan_canada_daily_tasmax_1990.nc')
    nc_tasmin = os.path.join(TESTS_DATA, 'NRCANdaily', 'nrcan_canada_daily_tasmin_1990.nc')

    def test_3d_data_with_nans(self):
        tasmax = xr.open_dataset(self.nc_tasmax).tasmax
        tasmin = xr.open_dataset(self.nc_tasmin).tasmin

        # put a nan somewhere
        tasmin.values[180, 1, 0] = np.nan

        frzthw = temp.daily_freezethaw_cycles(tasmax, tasmin, freq='YS')

        min1 = tasmin.values[:, 0, 0]
        max1 = tasmax.values[:, 0, 0]

        frzthw1 = ((min1 < K2C) * (max1 > K2C) * 1.0).sum()

        assert (np.allclose(frzthw1, frzthw.values[0, 0, 0]))

        assert (np.isnan(frzthw.values[0, 1, 0]))

        assert (np.isnan(frzthw.values[0, -1, -1]))

    def test_convert_units(self):
        tasmax = xr.open_dataset(self.nc_tasmax).tasmax
        tasmin = xr.open_dataset(self.nc_tasmin).tasmin
        tasmax.values -= K2C
        tasmax.attrs['units'] = 'C'
        tasmin.values -= K2C
        tasmin.attrs['units'] = 'C'
        # put a nan somewhere
        tasmin.values[180, 1, 0] = np.nan

        frzthw = temp.daily_freezethaw_cycles(tasmax, tasmin, freq='YS')

        min1 = tasmin.values[:, 0, 0]
        max1 = tasmax.values[:, 0, 0]

        frzthw1 = ((min1 < 0) * (max1 > 0) * 1.0).sum()

        assert (np.allclose(frzthw1, frzthw.values[0, 0, 0]))

        assert (np.isnan(frzthw.values[0, 1, 0]))

        assert (np.isnan(frzthw.values[0, -1, -1]))


class TestGrowingSeasonLength:
    def test_single_year(self, tas_series):
        a = np.zeros(366) + K2C
        ts = tas_series(a, start='1/1/2000')
        tt = (ts.time.dt.month >= 5) & (ts.time.dt.month <= 8)
        offset = np.random.uniform(low=5.5, high=23, size=(tt.sum().values,))
        ts[tt] = ts[tt] + offset

        out = temp.growing_season_length(ts)

        np.testing.assert_array_equal(out, tt.sum())

    def test_convert_units(self, tas_series):
        a = np.zeros(366)

        ts = tas_series(a, start='1/1/2000')
        ts.attrs['units'] = 'C'
        tt = (ts.time.dt.month >= 5) & (ts.time.dt.month <= 8)
        offset = np.random.uniform(low=5.5, high=23, size=(tt.sum().values,))
        ts[tt] = ts[tt] + offset

        out = temp.growing_season_length(ts)

        np.testing.assert_array_equal(out, tt.sum())

    def test_nan_presence(self, tas_series):
        a = np.zeros(366)
        a[50] = np.nan
        ts = tas_series(a, start='1/1/2000')
        ts.attrs['units'] = 'C'
        tt = (ts.time.dt.month >= 5) & (ts.time.dt.month <= 8)

        offset = np.random.uniform(low=5.5, high=23, size=(tt.sum().values,))
        ts[tt] = ts[tt] + offset

        out = temp.growing_season_length(ts)

        np.testing.assert_array_equal(out, [np.nan])

    def test_multiyear(self, tas_series):
        a = np.zeros(366 * 10)

        ts = tas_series(a, start='1/1/2000')
        ts.attrs['units'] = 'C'
        tt = (ts.time.dt.month >= 5) & (ts.time.dt.month <= 8)

        offset = np.random.uniform(low=5.5, high=23, size=(tt.sum().values,))
        ts[tt] = ts[tt] + offset

        out = temp.growing_season_length(ts)

        np.testing.assert_array_equal(out[3], tt[0:366].sum().values)


class TestTxDaysAbove:
    nc_file = os.path.join(TESTS_DATA, 'NRCANdaily', 'nrcan_canada_daily_tasmax_1990.nc')

    def test_3d_data_with_nans(self):
        # test with 3d data
        tas = xr.open_dataset(self.nc_file).tasmax
        tasC = xr.open_dataset(self.nc_file).tasmax
        tasC -= K2C
        tasC.attrs['units'] = 'C'
        # put a nan somewhere
        tas.values[180, 1, 0] = np.nan
        tasC.values[180, 1, 0] = np.nan
        # compute with both skipna options
        thresh = 273.16 + 25
        fd = temp.tx_days_above(tas, freq='YS')
        fdC = temp.tx_days_above(tasC, freq='YS')

        x1 = tas.values[:, 0, 0]

        fd1 = (x1[x1 > thresh]).size

        np.testing.assert_array_equal(fd, fdC)

        assert (np.allclose(fd1, fd.values[0, 0, 0]))

        assert (np.isnan(fd.values[0, 1, 0]))

        assert (np.isnan(fd.values[0, -1, -1]))


class TestTropicalNights:
    nc_file = os.path.join(TESTS_DATA, 'NRCANdaily', 'nrcan_canada_daily_tasmin_1990.nc')

    def test_3d_data_with_nans(self):
        # test with 3d data
        tas = xr.open_dataset(self.nc_file).tasmin
        tasC = xr.open_dataset(self.nc_file).tasmin
        tasC -= K2C
        tasC.attrs['units'] = 'C'
        # put a nan somewhere
        tas.values[180, 1, 0] = np.nan
        tasC.values[180, 1, 0] = np.nan
        # compute with both skipna options
        thresh = 273.16 + 20
        out = temp.tropical_nights(tas, freq='YS')
        outC = temp.tropical_nights(tasC, freq='YS')
        # fds = xci.frost_days(tasmin, thresh=thresh, freq='YS', skipna=True)

        x1 = tas.values[:, 0, 0]

        out1 = (x1[x1 > thresh]).size

        np.testing.assert_array_equal(out, outC)

        assert (np.allclose(out1, out.values[0, 0, 0]))

        assert (np.isnan(out.values[0, 1, 0]))

        assert (np.isnan(out.values[0, -1, -1]))


class TestTxTnDaysAbove:
    nc_tasmax = os.path.join(TESTS_DATA, 'NRCANdaily', 'nrcan_canada_daily_tasmax_1990.nc')
    nc_tasmin = os.path.join(TESTS_DATA, 'NRCANdaily', 'nrcan_canada_daily_tasmin_1990.nc')

    def test_3d_data_with_nans(self):
        tasmax = xr.open_dataset(self.nc_tasmax).tasmax
        tasmin = xr.open_dataset(self.nc_tasmin).tasmin

        tasmaxC = xr.open_dataset(self.nc_tasmax).tasmax
        tasminC = xr.open_dataset(self.nc_tasmin).tasmin
        tasmaxC -= K2C
        tasmaxC.attrs['units'] = 'C'
        tasminC -= K2C
        tasminC.attrs['units'] = 'C'

        # put a nan somewhere
        tasmin.values[180, 1, 0] = np.nan
        tasminC.values[180, 1, 0] = np.nan

        out = temp.tx_tn_days_above(tasmin, tasmax, thresh_tasmax=25, thresh_tasmin=18)
        outC = temp.tx_tn_days_above(tasminC, tasmaxC, thresh_tasmax=25, thresh_tasmin=18)
        np.testing.assert_array_equal(out, outC, )

        min1 = tasmin.values[:, 53, 76]
        max1 = tasmax.values[:, 53, 76]

        out1 = ((min1 > (K2C + 18)) * (max1 > (K2C + 25)) * 1.0).sum()

        assert (np.allclose(out1, out.values[0, 53, 76]))

        assert (np.isnan(out.values[0, 1, 0]))

        assert (np.isnan(out.values[0, -1, -1]))


class TestT90p:

    def test_tg90p_simple(self, tas_series):
        i = 366
        arr = np.asarray(np.arange(i), 'float')
        tas = tas_series(arr, start='1/1/2000')
        tasC = tas.copy()
        tasC -= K2C
        tasC.attrs['units'] = 'C'
        t90 = percentile_doy(tas, per=.1)

        # create cold spell in june
        tas[175:180] = 1
        tasC[175:180] = 1 - K2C
        out = temp.tg90p(tas, t90, freq='MS')
        outC = temp.tg90p(tasC, t90, freq='MS')

        np.testing.assert_array_equal(out, outC)
        assert out[0] == 30
        assert out[1] == 29
        assert out[5] == 25

        # nan treatment
        tas[33] = np.nan
        tasC[33] = np.nan
        out = temp.tg90p(tas, t90, freq='MS')
        outC = temp.tg90p(tasC, t90, freq='MS')

        np.testing.assert_array_equal(out, outC)
        assert out[0] == 30
        assert np.isnan(out[1])
        assert out[5] == 25

    def test_tn90p_simple(self, tasmin_series):
        i = 366
        arr = np.asarray(np.arange(i), 'float')
        tas = tasmin_series(arr, start='1/1/2000')
        tasC = tas.copy()
        tasC -= K2C
        tasC.attrs['units'] = 'C'
        t90 = percentile_doy(tas, per=.1)

        # create cold spell in june
        tas[175:180] = 1
        tasC[175:180] = 1 - K2C
        out = temp.tn90p(tas, t90, freq='MS')
        outC = temp.tn90p(tasC, t90, freq='MS')

        np.testing.assert_array_equal(out, outC)
        assert out[0] == 30
        assert out[1] == 29
        assert out[5] == 25

        # nan treatment
        tas[33] = np.nan
        tasC[33] = np.nan
        out = temp.tn90p(tas, t90, freq='MS')
        outC = temp.tn90p(tasC, t90, freq='MS')

        np.testing.assert_array_equal(out, outC)
        assert out[0] == 30
        assert np.isnan(out[1])
        assert out[5] == 25

    def test_tx90p_simple(self, tasmax_series):
        i = 366
        arr = np.asarray(np.arange(i), 'float')
        tas = tasmax_series(arr, start='1/1/2000')
        tasC = tas.copy()
        tasC -= K2C
        tasC.attrs['units'] = 'C'
        t90 = percentile_doy(tas, per=.1)

        # create cold spell in june
        tas[175:180] = 1
        tasC[175:180] = 1 - K2C
        out = temp.tx90p(tas, t90, freq='MS')
        outC = temp.tx90p(tasC, t90, freq='MS')

        np.testing.assert_array_equal(out, outC)
        assert out[0] == 30
        assert out[1] == 29
        assert out[5] == 25

        # nan treatment
        tas[33] = np.nan
        tasC[33] = np.nan
        out = temp.tx90p(tas, t90, freq='MS')
        outC = temp.tx90p(tasC, t90, freq='MS')

        np.testing.assert_array_equal(out, outC)
        assert out[0] == 30
        assert np.isnan(out[1])
        assert out[5] == 25


class TestT10p:

    def test_tg10p_simple(self, tas_series):
        i = 366
        arr = np.asarray(np.arange(i), 'float')
        tas = tas_series(arr, start='1/1/2000')
        tasC = tas.copy()
        tasC -= K2C
        tasC.attrs['units'] = 'C'
        t10 = percentile_doy(tas, per=.1)

        # create cold spell in june
        tas[175:180] = 1
        tasC[175:180] = 1 - K2C
        out = temp.tg10p(tas, t10, freq='MS')
        outC = temp.tg10p(tasC, t10, freq='MS')

        np.testing.assert_array_equal(out, outC)

        assert out[0] == 1
        assert out[5] == 5

        # nan treatment
        tas[33] = np.nan
        tasC[33] = np.nan
        out = temp.tg10p(tas, t10, freq='MS')
        outC = temp.tg10p(tasC, t10, freq='MS')

        np.testing.assert_array_equal(out, outC)
        assert out[0] == 1
        assert np.isnan(out[1])
        assert out[5] == 5

    def test_tn10p_simple(self, tasmin_series):
        i = 366
        arr = np.asarray(np.arange(i), 'float')
        tas = tasmin_series(arr, start='1/1/2000')
        tasC = tas.copy()
        tasC -= K2C
        tasC.attrs['units'] = 'C'
        t10 = percentile_doy(tas, per=.1)

        # create cold spell in june
        tas[175:180] = 1
        tasC[175:180] = 1 - K2C
        out = temp.tn10p(tas, t10, freq='MS')
        outC = temp.tn10p(tasC, t10, freq='MS')

        np.testing.assert_array_equal(out, outC)
        assert out[0] == 1
        assert out[5] == 5

        # nan treatment
        tas[33] = np.nan
        tasC[33] = np.nan
        out = temp.tn10p(tas, t10, freq='MS')
        outC = temp.tn10p(tasC, t10, freq='MS')

        np.testing.assert_array_equal(out, outC)
        assert out[0] == 1
        assert np.isnan(out[1])
        assert out[5] == 5

    def test_tx10p_simple(self, tasmax_series):
        i = 366
        arr = np.asarray(np.arange(i), 'float')
        tas = tasmax_series(arr, start='1/1/2000')
        tasC = tas.copy()
        tasC -= K2C
        tasC.attrs['units'] = 'C'
        t10 = percentile_doy(tas, per=.1)

        # create cold spell in june
        tas[175:180] = 1
        tasC[175:180] = 1 - K2C
        out = temp.tx10p(tas, t10, freq='MS')
        outC = temp.tx10p(tasC, t10, freq='MS')

        np.testing.assert_array_equal(out, outC)
        assert out[0] == 1
        assert out[5] == 5

        # nan treatment
        tas[33] = np.nan
        tasC[33] = np.nan
        out = temp.tx10p(tas, t10, freq='MS')
        outC = temp.tx10p(tasC, t10, freq='MS')

        np.testing.assert_array_equal(out, outC)
        assert out[0] == 1
        assert np.isnan(out[1])
        assert out[5] == 5

import os
import numpy as np
import xarray as xr

from xclim.testing.common import tas_series, tasmin_series, tasmax_series
import xclim.temperature as temp

TESTS_HOME = os.path.abspath(os.path.dirname(__file__))
TESTS_DATA = os.path.join(TESTS_HOME, 'testdata')

TAS_SERIES = tas_series()
TASMIN_SERIES = tasmin_series()
TASMAX_SERIES = tasmax_series()

K2C = 273.15


class TestTxMax:

    def test_simple(self, tas_series):
        ts = tas_series(np.arange(720))
        temp.tx_max(ts, freq='Y')


class TestTxMin:

    def test_simple(self, tas_series):
        ts = tas_series(np.arange(720))
        temp.tx_min(ts, freq='Y')


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


class TestColdSpellIndex:

    def test_simple(self, tas_series):
        a = np.zeros(365) + K2C
        a[10:20] -= 15  # 10 days
        a[40:43] -= 50  # too short -> 0
        a[80:100] -= 30  # at the end and beginning
        ts = tas_series(a)
        out = temp.cold_spell_index(ts, thresh=-10, freq='MS')
        np.testing.assert_array_equal(out, [10, 0, 12, 8, 0, 0, 0, 0, 0, 0, 0, 0])

    def test_convert_units(self, tas_series):
        a = np.zeros(365)
        a[10:20] -= 15  # 10 days
        a[40:43] -= 50  # too short -> 0
        a[80:100] -= 30  # at the end and beginning
        ts = tas_series(a)
        ts.attrs['units'] = 'C'
        out = temp.cold_spell_index(ts, thresh=-10, freq='MS')
        np.testing.assert_array_equal(out, [10, 0, 12, 8, 0, 0, 0, 0, 0, 0, 0, 0])

    def test_nan_presence(self, tas_series):
        a = np.zeros(365) + K2C
        a[10:20] -= 15  # 10 days
        a[40:43] -= 50  # too short -> 0
        a[80:100] -= 30  # at the end and beginning
        a[-1] = np.nan
        ts = tas_series(a)

        out = temp.cold_spell_index(ts, thresh=-10, freq='MS')
        np.testing.assert_array_equal(out, [10, 0, 12, 8, 0, 0, 0, 0, 0, 0, 0, np.nan])


class TestFrostDays:
    nc_file = os.path.join(TESTS_DATA, 'NRCANdaily', 'nrcan_canada_daily_tasmin_1990.nc')

    def test_3d_data_with_nans(self):
        # test with 3d data
        tasmin = xr.open_dataset(self.nc_file).tasmin
        # put a nan somewhere
        tasmin.values[180, 1, 0] = np.nan

        # compute with both skipna options
        thresh = 273.16
        fd = temp.frost_days(tasmin, freq='YS')
        # fds = xci.frost_days(tasmin, thresh=thresh, freq='YS', skipna=True)

        x1 = tasmin.values[:, 0, 0]
        # x2 = tasmin.values[:, 1, 0]

        fd1 = (x1[x1 < thresh]).size
        # fd2 = (x2[x2 < thresh]).size

        assert (np.allclose(fd1, fd.values[0, 0, 0]))
        # assert (np.allclose(fd1, fds.values[0, 0, 0]))
        assert (np.isnan(fd.values[0, 1, 0]))
        # assert (np.allclose(fd2, fds.values[0, 1, 0]))
        assert (np.isnan(fd.values[0, -1, -1]))
        # assert (np.isnan(fds.values[0, -1, -1]))


class TestCoolingDegreeDays:
    nc_file = os.path.join(TESTS_DATA, 'NRCANdaily', 'nrcan_canada_daily_tasmax_1990.nc')

    def test_3d_data_with_nans(self):
        # test with 3d data
        tas = xr.open_dataset(self.nc_file).tasmax
        # put a nan somewhere
        tas.values[180, 1, 0] = np.nan

        # compute with both skipna options
        thresh = 18 + K2C
        cdd = temp.cooling_dd(tas, thresh=18, freq='YS')

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
        cdd = temp.cooling_dd(tas, thresh=18, freq='YS')

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
        hdd = temp.heating_dd(tas, freq='YS')

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
        hdd = temp.heating_dd(tas, freq='YS')

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
        gdd = temp.growing_dd(tas, freq='YS')
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
        tn = np.zeros(366)
        tx = np.zeros(366)
        tn[:10] = np.array([20, 23, 23, 23, 23, 22, 23, 23, 23, 23])
        tx[:10] = np.array([29, 31, 31, 31, 29, 31, 31, 31, 31, 31])

        tn = tasmin_series(tn + K2C, start='1/1/2000')
        tx = tasmax_series(tx + K2C, start='1/1/2000')

        # some hw
        hwf = temp.heat_wave_frequency(tn, tx, thresh_tasmin=22,
                                       thresh_tasmax=30)
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

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
        # assert (np.allclose(gdd1, gdds.values[0, 0, 0]))
        assert (np.isnan(gdd.values[0, 1, 0]))
        # assert (np.allclose(gdd2, gdds.values[0, 1, 0]))
        assert (np.isnan(gdd.values[0, -1, -1]))
        # assert (np.isnan(gdds.values[0, -1, -1]))


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

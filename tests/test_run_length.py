from xclim import run_length as rl
from xclim.testing.common import tas_series
import xarray as xr
import pandas as pd
import numpy as np
import os

TAS_SERIES = tas_series
TESTS_HOME = os.path.abspath(os.path.dirname(__file__))
TESTS_DATA = os.path.join(TESTS_HOME, 'testdata')
K2C = 273.15


class TestRLE:

    def test_dataarray(self):
        values = np.zeros(365)
        time = pd.date_range('7/1/2000', periods=len(values), freq=pd.DateOffset(days=1))
        values[1:11] = 1
        da = xr.DataArray(values, coords={'time': time}, dims='time')

        v, l, p = rl.rle_1d(da != 0)


class TestLongestRun:
    nc_pr = os.path.join(TESTS_DATA, 'NRCANdaily', 'nrcan_canada_daily_pr_1990.nc')

    def test_simple(self):
        values = np.zeros(365)
        time = pd.date_range('7/1/2000', periods=len(values), freq=pd.DateOffset(days=1))
        values[1:11] = 1
        da = xr.DataArray(values != 0, coords={'time': time}, dims='time')
        lt = da.resample(time='M').apply(rl.longest_run_ufunc)
        assert lt[0] == 10
        np.testing.assert_array_equal(lt[1:], 0)

        # n-dim version versus ufunc
        da3d = xr.open_dataset(self.nc_pr).pr[:, 40:50, 50:68] != 0
        lt_orig = da3d.resample(time='M').apply(rl.longest_run_ufunc)
        lt_Ndim = da3d.resample(time='M').apply(rl.longest_run, dim='time')
        np.testing.assert_array_equal(lt_orig, lt_Ndim)

    def test_start_at_0(self):
        values = np.zeros(365)
        time = pd.date_range('7/1/2000', periods=len(values), freq=pd.DateOffset(days=1))
        values[0:10] = 1
        da = xr.DataArray(values != 0, coords={'time': time}, dims='time')
        lt = da.resample(time='M').apply(rl.longest_run_ufunc)
        assert lt[0] == 10
        np.testing.assert_array_equal(lt[1:], 0)

        # n-dim version versus ufunc
        da3d = xr.open_dataset(self.nc_pr).pr[:, 40:50, 50:68] * 0
        da3d[0:10, ] = da3d[0:10, ] + 1
        da3d = da3d == 1
        lt_orig = da3d.resample(time='M').apply(rl.longest_run_ufunc)
        lt_Ndim = da3d.resample(time='M').apply(rl.longest_run, dim='time')
        np.testing.assert_array_equal(lt_orig, lt_Ndim)

    def test_end_start_at_0(self):
        values = np.zeros(365)
        time = pd.date_range('7/1/2000', periods=len(values), freq=pd.DateOffset(days=1))
        values[-10:] = 1
        da = xr.DataArray(values != 0, coords={'time': time}, dims='time')

        lt = da.resample(time='M').apply(rl.longest_run_ufunc)
        assert lt[-1] == 10
        np.testing.assert_array_equal(lt[:-1], 0)

        # n-dim version versus ufunc
        da3d = xr.open_dataset(self.nc_pr).pr[:, 40:50, 50:68] * 0
        da3d[-10:, ] = da3d[-10:, ] + 1
        da3d = da3d == 1
        lt_orig = da3d.resample(time='M').apply(rl.longest_run_ufunc)
        lt_Ndim = da3d.resample(time='M').apply(rl.longest_run, dim='time')
        np.testing.assert_array_equal(lt_orig, lt_Ndim)

    def test_all_true(self):
        values = np.ones(365)
        time = pd.date_range('7/1/2000', periods=len(values), freq=pd.DateOffset(days=1))
        da = xr.DataArray(values != 0, coords={'time': time}, dims='time')

        lt = da.resample(time='M').apply(rl.longest_run_ufunc)
        np.testing.assert_array_equal(lt, da.resample(time='M').count(dim='time'))

        # n-dim version versus ufunc
        da3d = xr.open_dataset(self.nc_pr).pr[:, 40:50, 50:68] * 0 + 1
        da3d = da3d == 1
        lt_orig = da3d.resample(time='M').apply(rl.longest_run_ufunc)
        lt_Ndim = da3d.resample(time='M').apply(rl.longest_run, dim='time')
        np.testing.assert_array_equal(lt_orig, lt_Ndim)

    def test_almost_all_true(self):
        values = np.ones(365)
        values[35] = 0
        time = pd.date_range('7/1/2000', periods=len(values), freq=pd.DateOffset(days=1))
        da = xr.DataArray(values != 0, coords={'time': time}, dims='time')

        lt = da.resample(time='M').apply(rl.longest_run_ufunc)
        n = da.resample(time='M').count(dim='time')
        np.testing.assert_array_equal(lt[0], n[0])
        np.testing.assert_array_equal(lt[1], 26)

        # n-dim version versus ufunc
        da3d = xr.open_dataset(self.nc_pr).pr[:, 40:50, 50:68] * 0 + 1
        da3d[35, ] = da3d[35, ] + 1
        da3d = da3d == 1
        lt_orig = da3d.resample(time='M').apply(rl.longest_run_ufunc)
        lt_Ndim = da3d.resample(time='M').apply(rl.longest_run, dim='time')
        np.testing.assert_array_equal(lt_orig, lt_Ndim)


class TestFirstRun:
    nc_pr = os.path.join(TESTS_DATA, 'NRCANdaily', 'nrcan_canada_daily_pr_1990.nc')

    def test_simple(self):
        a = np.zeros(100, bool)
        a[10:20] = 1
        i = rl.first_run_1d(a, 5)
        assert 10 == i

        # n-dim version versus ufunc
        da3d = xr.open_dataset(self.nc_pr).pr[:, 40:50, 50:68] != 0
        lt_orig = da3d.resample(time='M').apply(rl.first_run_ufunc, window=5)
        lt_Ndim = da3d.resample(time='M').apply(rl.first_run, window=5, dim='time')
        np.testing.assert_array_equal(lt_orig, lt_Ndim)


class TestWindowedRunEvents:
    nc_pr = os.path.join(TESTS_DATA, 'NRCANdaily', 'nrcan_canada_daily_pr_1990.nc')

    def test_simple(self):
        a = np.zeros(50, bool)
        a[4:7] = True
        a[34:45] = True
        assert rl.windowed_run_events_1d(a, 3) == 2

        # n-dim version versus ufunc
        da3d = xr.open_dataset(self.nc_pr).pr[:, 40:50, 50:68] != 0
        lt_orig = da3d.resample(time='M').apply(rl.windowed_run_events_ufunc, window=4)
        lt_Ndim = da3d.resample(time='M').apply(rl.windowed_run_events, window=4, dim='time')
        np.testing.assert_array_equal(lt_orig, lt_Ndim)


class TestWindowedRunCount:
    nc_pr = os.path.join(TESTS_DATA, 'NRCANdaily', 'nrcan_canada_daily_pr_1990.nc')

    def test_simple(self):
        a = np.zeros(50, bool)
        a[4:7] = True
        a[34:45] = True
        assert rl.windowed_run_count_1d(a, 3) == len(a[4:7]) + len(a[34:45])

        # n-dim version versus ufunc
        da3d = xr.open_dataset(self.nc_pr).pr[:, 40:50, 50:68] != 0
        lt_orig = da3d.resample(time='M').apply(rl.windowed_run_count_ufunc, window=4)
        lt_Ndim = da3d.resample(time='M').apply(rl.windowed_run_count, window=4, dim='time')
        np.testing.assert_array_equal(lt_orig, lt_Ndim)

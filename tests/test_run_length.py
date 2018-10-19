from xclim import run_length as rl

import xarray as xr
import pandas as pd
import numpy as np


class TestRLE:
    def test_dataarray(self):
        values = np.zeros(365)
        time = pd.date_range('7/1/2000', periods=len(values), freq=pd.DateOffset(days=1))
        values[1:11] = 1
        da = xr.DataArray(values, coords={'time': time}, dims='time')

        v, l, p = rl.rle(da != 0)


class TestLongestRun:
    def test_simple(self):
        values = np.zeros(365)
        time = pd.date_range('7/1/2000', periods=len(values), freq=pd.DateOffset(days=1))
        values[1:11] = 1
        da = xr.DataArray(values != 0, coords={'time': time}, dims='time')

        lt = da.resample(time='M').apply(rl.longest_run_ufunc)
        assert lt[0] == 10
        np.testing.assert_array_equal(lt[1:], 0)

    def test_start_at_0(self):
        values = np.zeros(365)
        time = pd.date_range('7/1/2000', periods=len(values), freq=pd.DateOffset(days=1))
        values[0:10] = 1
        da = xr.DataArray(values != 0, coords={'time': time}, dims='time')

        lt = da.resample(time='M').apply(rl.longest_run_ufunc)
        assert lt[0] == 10
        np.testing.assert_array_equal(lt[1:], 0)

    def test_end_start_at_0(self):
        values = np.zeros(365)
        time = pd.date_range('7/1/2000', periods=len(values), freq=pd.DateOffset(days=1))
        values[-10:] = 1
        da = xr.DataArray(values != 0, coords={'time': time}, dims='time')

        lt = da.resample(time='M').apply(rl.longest_run_ufunc)
        assert lt[-1] == 10
        np.testing.assert_array_equal(lt[:-1], 0)

    def test_all_true(self):
        values = np.ones(365)
        time = pd.date_range('7/1/2000', periods=len(values), freq=pd.DateOffset(days=1))
        da = xr.DataArray(values != 0, coords={'time': time}, dims='time')

        lt = da.resample(time='M').apply(rl.longest_run_ufunc)
        np.testing.assert_array_equal(lt, da.resample(time='M').count(dim='time'))

    def test_almost_all_true(self):
        values = np.ones(365)
        values[35] = 0
        time = pd.date_range('7/1/2000', periods=len(values), freq=pd.DateOffset(days=1))
        da = xr.DataArray(values != 0, coords={'time': time}, dims='time')

        lt = da.resample(time='M').apply(rl.longest_run_ufunc)
        n = da.resample(time='M').count(dim='time')
        np.testing.assert_array_equal(lt[0], n[0])
        np.testing.assert_array_equal(lt[1], 26)

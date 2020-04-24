import os

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from xclim.indices import run_length as rl

TESTS_HOME = os.path.abspath(os.path.dirname(__file__))
TESTS_DATA = os.path.join(TESTS_HOME, "testdata")
K2C = 273.15


class TestRLE:
    def test_dataarray(self):
        values = np.zeros(365)
        time = pd.date_range(
            "7/1/2000", periods=len(values), freq=pd.DateOffset(days=1)
        )
        values[1:11] = 1
        da = xr.DataArray(values, coords={"time": time}, dims="time")

        v, l, p = rl.rle_1d(da != 0)
        np.testing.assert_array_equal(v, [False, True, False])
        np.testing.assert_array_equal(l, [1, 10, 354])
        np.testing.assert_array_equal(p, [0, 1, 11])


class TestLongestRun:
    nc_pr = os.path.join(TESTS_DATA, "NRCANdaily", "nrcan_canada_daily_pr_1990.nc")

    def test_simple(self):
        values = np.zeros(365)
        time = pd.date_range(
            "7/1/2000", periods=len(values), freq=pd.DateOffset(days=1)
        )
        values[1:11] = 1
        da = xr.DataArray(values != 0, coords={"time": time}, dims="time")
        lt = da.resample(time="M").map(rl.longest_run_ufunc)
        assert lt[0] == 10
        np.testing.assert_array_equal(lt[1:], 0)

        # n-dim version versus ufunc
        da3d = xr.open_dataset(self.nc_pr).pr[:, 40:50, 50:68] != 0
        lt_orig = da3d.resample(time="M").map(rl.longest_run_ufunc)
        # override 'auto' usage of ufunc for small number of gridpoints
        lt_Ndim = da3d.resample(time="M").map(
            rl.longest_run, dim="time", ufunc_1dim=False
        )
        np.testing.assert_array_equal(lt_orig, lt_Ndim)

    def test_start_at_0(self):
        values = np.zeros(365)
        time = pd.date_range(
            "7/1/2000", periods=len(values), freq=pd.DateOffset(days=1)
        )
        values[0:10] = 1
        da = xr.DataArray(values != 0, coords={"time": time}, dims="time")
        lt = da.resample(time="M").map(rl.longest_run_ufunc)
        assert lt[0] == 10
        np.testing.assert_array_equal(lt[1:], 0)

        # n-dim version versus ufunc
        da3d = xr.open_dataset(self.nc_pr).pr[:, 40:50, 50:68] * 0
        da3d[0:10,] = da3d[0:10,] + 1
        da3d = da3d == 1
        lt_orig = da3d.resample(time="M").map(rl.longest_run_ufunc)
        # override 'auto' usage of ufunc for small number of gridpoints
        lt_Ndim = da3d.resample(time="M").map(
            rl.longest_run, dim="time", ufunc_1dim=False
        )  # override 'auto' for small
        np.testing.assert_array_equal(lt_orig, lt_Ndim)

    def test_end_start_at_0(self):
        values = np.zeros(365)
        time = pd.date_range(
            "7/1/2000", periods=len(values), freq=pd.DateOffset(days=1)
        )
        values[-10:] = 1
        da = xr.DataArray(values != 0, coords={"time": time}, dims="time")

        lt = da.resample(time="M").map(rl.longest_run_ufunc)
        assert lt[-1] == 10
        np.testing.assert_array_equal(lt[:-1], 0)

        # n-dim version versus ufunc
        da3d = xr.open_dataset(self.nc_pr).pr[:, 40:50, 50:68] * 0
        da3d[-10:,] = da3d[-10:,] + 1
        da3d = da3d == 1
        lt_orig = da3d.resample(time="M").map(rl.longest_run_ufunc)
        lt_Ndim = da3d.resample(time="M").map(
            rl.longest_run, dim="time", ufunc_1dim=False
        )
        np.testing.assert_array_equal(lt_orig, lt_Ndim)

    def test_all_true(self):
        values = np.ones(365)
        time = pd.date_range(
            "7/1/2000", periods=len(values), freq=pd.DateOffset(days=1)
        )
        da = xr.DataArray(values != 0, coords={"time": time}, dims="time")

        lt = da.resample(time="M").map(rl.longest_run_ufunc)
        np.testing.assert_array_equal(lt, da.resample(time="M").count(dim="time"))

        # n-dim version versus ufunc
        da3d = xr.open_dataset(self.nc_pr).pr[:, 40:50, 50:68] * 0 + 1
        da3d = da3d == 1
        lt_orig = da3d.resample(time="M").map(rl.longest_run_ufunc)
        lt_Ndim = da3d.resample(time="M").map(
            rl.longest_run, dim="time", ufunc_1dim=False
        )
        np.testing.assert_array_equal(lt_orig, lt_Ndim)

    def test_almost_all_true(self):
        values = np.ones(365)
        values[35] = 0
        time = pd.date_range(
            "7/1/2000", periods=len(values), freq=pd.DateOffset(days=1)
        )
        da = xr.DataArray(values != 0, coords={"time": time}, dims="time")

        lt = da.resample(time="M").map(rl.longest_run_ufunc)
        n = da.resample(time="M").count(dim="time")
        np.testing.assert_array_equal(lt[0], n[0])
        np.testing.assert_array_equal(lt[1], 26)

        # n-dim version versus ufunc
        da3d = xr.open_dataset(self.nc_pr).pr[:, 40:50, 50:68] * 0 + 1
        da3d[35,] = da3d[35,] + 1
        da3d = da3d == 1
        lt_orig = da3d.resample(time="M").map(rl.longest_run_ufunc)
        lt_Ndim = da3d.resample(time="M").map(
            rl.longest_run, dim="time", ufunc_1dim=False
        )
        np.testing.assert_array_equal(lt_orig, lt_Ndim)


class TestFirstRun:
    nc_pr = os.path.join(TESTS_DATA, "NRCANdaily", "nrcan_canada_daily_pr_1990.nc")

    def test_run_1d(self):
        a = np.zeros(100, bool)
        a[10:20] = 1
        i = rl.first_run_1d(a, 5)
        assert 10 == i

    def test_real_data(self):
        # n-dim version versus ufunc
        da3d = xr.open_dataset(self.nc_pr).pr[:, 40:50, 50:68] != 0
        lt_orig = da3d.resample(time="M").map(rl.first_run_ufunc, window=5)
        lt_Ndim = da3d.resample(time="M").map(
            rl.first_run, window=5, dim="time", ufunc_1dim=False
        )
        np.testing.assert_array_equal(lt_orig, lt_Ndim)

    @pytest.mark.parametrize(
        "coord,expected",
        [(False, 30), (True, np.datetime64("2000-01-31")), ("dayofyear", 31)],
    )
    @pytest.mark.parametrize(
        "use_dask,use_1dim", [(True, False), (False, False), (False, True)]
    )
    def test_simple(self, tas_series, coord, expected, use_dask, use_1dim):
        t = np.zeros(60)
        t[30:40] = 2
        tas = tas_series(t, start="2000-01-01")
        runs = xr.concat((tas, tas), dim="dim0")

        if use_dask:
            runs = runs.chunk({"time": 10, "dim0": 1})

        out = rl.first_run(runs, window=1, dim="time", coord=coord, ufunc_1dim=use_1dim)
        np.testing.assert_array_equal(out.load(), expected)


class TestWindowedRunEvents:
    nc_pr = os.path.join(TESTS_DATA, "NRCANdaily", "nrcan_canada_daily_pr_1990.nc")

    def test_simple(self):
        a = np.zeros(50, bool)
        a[4:7] = True
        a[34:45] = True
        assert rl.windowed_run_events_1d(a, 3) == 2

        # n-dim version versus ufunc
        da3d = xr.open_dataset(self.nc_pr).pr[:, 40:50, 50:68] != 0
        lt_orig = da3d.resample(time="M").map(rl.windowed_run_events_ufunc, window=4)
        lt_Ndim = da3d.resample(time="M").map(
            rl.windowed_run_events, window=4, dim="time", ufunc_1dim=False
        )
        np.testing.assert_array_equal(lt_orig, lt_Ndim)


class TestWindowedRunCount:
    nc_pr = os.path.join(TESTS_DATA, "NRCANdaily", "nrcan_canada_daily_pr_1990.nc")

    def test_simple(self):
        a = np.zeros(50, bool)
        a[4:7] = True
        a[34:45] = True
        assert rl.windowed_run_count_1d(a, 3) == len(a[4:7]) + len(a[34:45])

        # n-dim version versus ufunc
        da3d = xr.open_dataset(self.nc_pr).pr[:, 40:50, 50:68] != 0
        lt_orig = da3d.resample(time="M").map(rl.windowed_run_count_ufunc, window=4)
        lt_Ndim = da3d.resample(time="M").map(
            rl.windowed_run_count, window=4, dim="time", ufunc_1dim=False
        )
        np.testing.assert_array_equal(lt_orig, lt_Ndim)


class TestLastRun:
    @pytest.mark.parametrize(
        "coord,expected",
        [(False, 39), (True, np.datetime64("2000-02-09")), ("dayofyear", 40)],
    )
    @pytest.mark.parametrize(
        "use_dask,use_1dim", [(True, False), (False, False), (False, True)]
    )
    def test_simple(self, tas_series, coord, expected, use_dask, use_1dim):
        t = np.zeros(60)
        t[30:40] = 2
        tas = tas_series(t, start="2000-01-01")
        runs = xr.concat((tas, tas), dim="dim0")

        if use_dask:
            runs = runs.chunk({"time": 10, "dim0": 1})

        out = rl.last_run(runs, window=1, dim="time", coord=coord, ufunc_1dim=use_1dim)
        np.testing.assert_array_equal(out.load(), expected)


class TestRunsWithDates:
    @pytest.mark.parametrize(
        "date,end,expected",
        [
            ("07-01", 210, 70),
            ("07-01", 190, 50),
            ("04-01", 150, np.NaN),  # date falls early
            ("11-01", 150, 164),  # date ends late
        ],
    )
    @pytest.mark.parametrize("use_dask", [True, False])
    def test_run_length_with_date(self, tas_series, date, end, expected, use_dask):
        t = np.zeros(360)
        t[140:end] = 1
        tas = tas_series(t, start="2000-01-01")
        runs = xr.concat((tas, tas), dim="dim0")
        runs = runs == 1

        if use_dask:
            runs = runs.chunk({"time": 10, "dim0": 1})

        out = rl.run_length_with_date(runs, window=1, dim="time", date=date,)
        np.testing.assert_array_equal(np.mean(out.load()), expected)

    @pytest.mark.parametrize(
        "coord,date,end,expected",
        [
            ("dayofyear", "07-01", 210, 211),
            (False, "07-01", 190, 190),
            ("dayofyear", "04-01", 150, np.NaN),  # date falls early
            ("dayofyear", "11-01", 150, 305),  # date ends late
        ],
    )
    @pytest.mark.parametrize("use_dask", [True, False])
    def test_run_end_after_date(self, tas_series, coord, date, end, expected, use_dask):
        t = np.zeros(360)
        t[140:end] = 1
        tas = tas_series(t, start="2000-01-01")
        runs = xr.concat((tas, tas), dim="dim0")
        runs = runs == 1

        if use_dask:
            runs = runs.chunk({"time": 10, "dim0": 1})

        out = rl.run_end_after_date(runs, window=1, date=date, dim="time", coord=coord)
        np.testing.assert_array_equal(np.mean(out.load()), expected)

    @pytest.mark.parametrize(
        "coord,date,end,expected",
        [
            ("dayofyear", "07-01", 210, 182),
            (False, "07-01", 190, 181),
            ("dayofyear", "04-01", 150, np.NaN),  # date falls early
            ("dayofyear", "11-01", 150, 150),  # date ends late
        ],
    )
    @pytest.mark.parametrize("use_dask", [True, False])
    def test_last_run_before_date(
        self, tas_series, coord, date, end, expected, use_dask
    ):
        t = np.zeros(360)
        t[140:end] = 1
        tas = tas_series(t, start="2000-01-01")
        runs = xr.concat((tas, tas), dim="dim0")
        runs = runs == 1

        if use_dask:
            runs = runs.chunk({"time": 10, "dim0": 1})

        out = rl.last_run_before_date(
            runs, window=1, date=date, dim="time", coord=coord
        )
        np.testing.assert_array_equal(np.mean(out.load()), expected)

    @pytest.mark.parametrize(
        "func",
        [rl.last_run_before_date, rl.run_end_after_date, rl.run_length_with_date],
    )
    @pytest.mark.parametrize("use_dask", [True, False])
    def test_run_with_dates_no_date(self, tas_series, use_dask, func):
        t = np.ones(90)
        tas = tas_series(t, start="2000-01-01")
        runs = xr.concat((tas, tas), dim="dim0")
        runs = runs == 1

        if use_dask:
            runs = runs.chunk({"time": 10, "dim0": 1})

        out = runs.resample(time="MS").map(func, window=1, date="07-01", dim="time")
        assert out.isnull().all()

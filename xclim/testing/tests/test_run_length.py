import os

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from xclim.core.options import set_options
from xclim.indices import run_length as rl
from xclim.testing import open_dataset

K2C = 273.15


class TestSuspiciousRun:
    def test_simple(self, tas_series):
        t = np.zeros(365)
        tas = tas_series(t, start="2000-01-01")
        sus = rl.suspicious_run(tas)
        # Only zeroes
        assert sus.all()

        t = np.zeros(365)
        t[30:39] = 5
        tas = tas_series(t, start="2000-01-01")
        sus = rl.suspicious_run(tas, thresh=0)
        # Not enough 5s to trigger suspicion
        assert not sus[30:39].all()
        assert not sus[0:10].all()

        t = np.zeros(365)
        t[30:40] = 1
        tas = tas_series(t, start="2000-01-01")
        sus = rl.suspicious_run(tas, thresh=0)
        # Run of 10 identical values
        assert sus[30:40].all()
        assert not sus[30:41].all()

    def test_above_thresh(self, tas_series):
        t = np.zeros(365)
        t[30:40] = 0.1
        t[40:50] = 1e-6
        t[50:60] = 0.0001
        t[60:65] = 1e-9
        tas = tas_series(t, start="2000-01-01")

        sus = rl.suspicious_run(tas, thresh=0, window=5)
        assert not sus[:30].any()
        assert sus[30:65].all()
        assert not sus[65:].any()

        sus = rl.suspicious_run(tas, thresh=1e-9, window=5)
        assert sus[30:60].all()
        assert not sus[60:].any()

        sus = rl.suspicious_run(tas, thresh=1e-5, window=5)
        assert sus[30:40].all()
        assert not sus[40:50].any()
        assert sus[50:60].all()
        assert not sus[60:].any()

        sus = rl.suspicious_run(tas, thresh=0, window=11)
        assert not sus.any()

    @pytest.mark.parametrize("use_dask", [True, False])
    def test_dask(self, use_dask):
        values = np.zeros((10, 200))
        time = pd.date_range("2015-01-01", periods=200, freq="D")
        values[:, :10] = 1
        values[9, :] = 1
        da = xr.DataArray(values, coords={"time": time}, dims=("qq", "time"))

        if use_dask:
            da = da.chunk({"qq": 2})

        sus = rl.suspicious_run(da, thresh=0)
        assert sus[:, :10].all()
        assert not sus[1, 10:].any()
        assert sus[9].all()

        sus = rl.suspicious_run(da)
        assert sus.all()


@pytest.fixture(scope="module", params=[True, False], autouse=True)
def ufunc(request):
    with set_options(run_length_ufunc=request.param):
        yield request.param


@pytest.mark.parametrize("use_dask", [True, False])
@pytest.mark.parametrize("index", ["first", "last"])
def test_rle(ufunc, use_dask, index):
    if use_dask and ufunc:
        pytest.xfail("rle_1d is not implemented for dask arrays.")

    values = np.zeros((10, 365, 4, 4))
    time = pd.date_range("2000-01-01", periods=365, freq="D")
    values[:, 1:11, ...] = 1
    da = xr.DataArray(values, coords={"time": time}, dims=("a", "time", "b", "c"))

    if ufunc:
        da = da[0, :, 0, 0]
        v, l, p = rl.rle_1d(da != 0)
        np.testing.assert_array_equal(v, [False, True, False])
        np.testing.assert_array_equal(l, [1, 10, 354])
        np.testing.assert_array_equal(p, [0, 1, 11])
    else:
        if use_dask:
            da = da.chunk({"a": 1, "b": 2})

        out = rl.rle(da != 0, index=index).mean(["a", "b", "c"])
        if index == "last":
            expected = np.zeros(365)
            expected[1:10] = np.nan
            expected[10] = 10
        else:
            expected = np.zeros(365)
            expected[1] = 10
            expected[2:11] = np.nan
        np.testing.assert_array_equal(out, expected)


class TestStatisticsRun:
    nc_pr = os.path.join("NRCANdaily", "nrcan_canada_daily_pr_1990.nc")

    def test_simple(self):
        values = np.zeros(365)
        time = pd.date_range(
            "7/1/2000", periods=len(values), freq=pd.DateOffset(days=1)
        )
        values[1:11] = 1
        da = xr.DataArray(values != 0, coords={"time": time}, dims="time")
        lt = da.resample(time="M").map(rl.rle_statistics, reducer="max")
        assert lt[0] == 10
        np.testing.assert_array_equal(lt[1:], 0)

    def test_start_at_0(self):
        values = np.zeros(365)
        time = pd.date_range(
            "7/1/2000", periods=len(values), freq=pd.DateOffset(days=1)
        )
        values[0:10] = 1
        da = xr.DataArray(values != 0, coords={"time": time}, dims="time")
        lt = da.resample(time="M").map(rl.rle_statistics, reducer="max")
        assert lt[0] == 10
        np.testing.assert_array_equal(lt[1:], 0)

    def test_end_start_at_0(self):
        values = np.zeros(365)
        time = pd.date_range(
            "7/1/2000", periods=len(values), freq=pd.DateOffset(days=1)
        )
        values[-10:] = 1
        da = xr.DataArray(values != 0, coords={"time": time}, dims="time")

        lt = da.resample(time="M").map(rl.rle_statistics, reducer="max")
        assert lt[-1] == 10
        np.testing.assert_array_equal(lt[:-1], 0)

    def test_all_true(self):
        values = np.ones(365)
        time = pd.date_range(
            "7/1/2000", periods=len(values), freq=pd.DateOffset(days=1)
        )
        da = xr.DataArray(values != 0, coords={"time": time}, dims="time")

        lt = da.resample(time="M").map(rl.rle_statistics, reducer="max")
        np.testing.assert_array_equal(lt, da.resample(time="M").count(dim="time"))

    def test_almost_all_true(self):
        values = np.ones(365)
        values[35] = 0
        time = pd.date_range(
            "7/1/2000", periods=len(values), freq=pd.DateOffset(days=1)
        )
        da = xr.DataArray(values != 0, coords={"time": time}, dims="time")

        lt = da.resample(time="M").map(rl.rle_statistics, reducer="max")
        n = da.resample(time="M").count(dim="time")
        np.testing.assert_array_equal(lt[0], n[0])
        np.testing.assert_array_equal(lt[1], 26)

    def test_other_stats(self):
        values = np.ones(365)
        values[35] = 0
        time = pd.date_range(
            "1/1/2000", periods=len(values), freq=pd.DateOffset(days=1)
        )
        da = xr.DataArray(values != 0, coords={"time": time}, dims="time")

        lt = da.resample(time="YS").map(rl.rle_statistics, reducer="min")
        assert lt == 35

        lt = da.resample(time="YS").map(rl.rle_statistics, reducer="mean", window=36)
        assert lt == 329

        lt = da.resample(time="YS").map(rl.rle_statistics, reducer="std")
        assert lt == 147


class TestFirstRun:
    nc_pr = os.path.join("NRCANdaily", "nrcan_canada_daily_pr_1990.nc")

    def test_real_simple(self):
        a = xr.DataArray(np.zeros(100, bool), dims=("x",))
        a[10:20] = 1
        i = rl.first_run(a, 5, dim="x")
        assert 10 == i

    def test_real_data(self):
        # n-dim version versus ufunc
        da3d = open_dataset(self.nc_pr).pr[:, 40:50, 50:68] != 0
        da3d.resample(time="M").map(rl.first_run, window=5)

    @pytest.mark.parametrize(
        "coord,expected",
        [(False, 30), (True, np.datetime64("2000-01-31")), ("dayofyear", 31)],
    )
    @pytest.mark.parametrize("use_dask", [True, False])
    def test_simple(self, tas_series, coord, expected, use_dask, ufunc):
        # if use_dask and ufunc:
        #     pytest.xfail("Ufunc run length algorithms not implemented for dask arrays.")
        t = np.zeros(60)
        t[30:40] = 2
        tas = tas_series(t, start="2000-01-01")
        runs = xr.concat((tas, tas), dim="dim0")

        if use_dask:
            runs = runs.chunk({"time": -1 if ufunc else 10, "dim0": 1})

        out = rl.first_run(runs, window=1, dim="time", coord=coord)
        np.testing.assert_array_equal(out.load(), expected)


class TestWindowedRunEvents:
    @pytest.mark.parametrize("index", ["first", "last"])
    def test_simple(self, index):
        a = xr.DataArray(np.zeros(50, bool), dims=("x",))
        a[4:7] = True
        a[34:45] = True
        assert rl.windowed_run_events(a, 3, dim="x", index=index) == 2


class TestWindowedRunCount:
    @pytest.mark.parametrize("index", ["first", "last"])
    def test_simple(self, index):
        a = xr.DataArray(np.zeros(50, bool), dims=("time",))
        a[4:7] = True
        a[34:45] = True
        assert rl.windowed_run_count(a, 3, dim="time", index=index) == len(
            a[4:7]
        ) + len(a[34:45])


class TestLastRun:
    @pytest.mark.parametrize(
        "coord,expected",
        [(False, 39), (True, np.datetime64("2000-02-09")), ("dayofyear", 40)],
    )
    @pytest.mark.parametrize("use_dask", [True, False])
    def test_simple(self, tas_series, coord, expected, use_dask, ufunc):
        # if use_dask and ufunc:
        #     pytest.xfail("Ufunc run length algorithms not implemented for dask arrays.")
        t = np.zeros(60)
        t[30:40] = 2
        tas = tas_series(t, start="2000-01-01")
        runs = xr.concat((tas, tas), dim="dim0")

        if use_dask:
            runs = runs.chunk({"time": -1 if ufunc else 10, "dim0": 1})

        out = rl.last_run(runs, window=1, dim="time", coord=coord, ufunc_1dim=ufunc)
        np.testing.assert_array_equal(out.load(), expected)


def test_run_bounds_synthetic():
    run = xr.DataArray(
        [0, 1, 1, 1, 0, 0, 1, 1, 1, 0], dims="x", coords={"x": np.arange(10) ** 2}
    )
    bounds = rl.run_bounds(run, "x", coord=True)
    np.testing.assert_array_equal(bounds, [[1, 36], [16, 81]])

    bounds = rl.run_bounds(run, "x", coord=False)
    np.testing.assert_array_equal(bounds, [[1, 6], [4, 9]])


def test_run_bounds_data():
    era5 = open_dataset("ERA5/daily_surface_cancities_1990-1993.nc")
    cond = era5.tas.rolling(time=7).mean() > 285

    bounds = rl.run_bounds(cond, "time")  # def coord = True
    np.testing.assert_array_equal(
        bounds.isel(location=0, events=0),
        pd.to_datetime(["1990-06-19", "1990-10-26"]).values,
    )

    bounds = rl.run_bounds(cond, "time", coord="dayofyear")
    np.testing.assert_array_equal(bounds.isel(location=1, events=4), [279, 283])
    assert bounds.events.size == 15


def test_keep_longest_run_synthetic():
    runs = xr.DataArray([0, 1, 1, 1, 0, 0, 1, 1, 1, 0], dims="time").astype(bool)
    lrun = rl.keep_longest_run(runs, "time")
    np.testing.assert_array_equal(
        lrun, np.array([0, 1, 1, 1, 0, 0, 0, 0, 0, 0], dtype=bool)
    )


def test_keep_longest_run_data():
    era5 = open_dataset("ERA5/daily_surface_cancities_1990-1993.nc")
    cond = era5.swe > 0.001
    lrun = rl.keep_longest_run(cond, "time")
    np.testing.assert_array_equal(
        lrun.isel(time=slice(651, 658), location=2),
        np.array([0, 0, 0, 1, 1, 1, 1], dtype=bool),
    )

    xr.testing.assert_equal(
        rl.keep_longest_run(cond, "time").sum("time"),
        rl.longest_run(cond, "time"),
    )


class TestRunsWithDates:
    @pytest.mark.parametrize(
        "date,end,expected",
        [
            ("07-01", 210, 70),
            ("07-01", 190, 50),
            ("04-01", 150, np.NaN),  # date falls early
            ("11-01", 150, 165),  # date ends late
            (None, 150, 10),  # no date, real length
        ],
    )
    @pytest.mark.parametrize("use_dask", [True, False])
    def test_season_length(self, tas_series, date, end, expected, use_dask, ufunc):
        # if use_dask and ufunc:
        #     pytest.xfail("Ufunc run length algorithms not implemented for dask arrays.")
        t = np.zeros(360)
        t[140:end] = 1
        tas = tas_series(t, start="2000-01-01")
        runs = xr.concat((tas, tas), dim="dim0")
        runs = runs == 1

        if use_dask:
            runs = runs.chunk({"time": -1 if ufunc else 10, "dim0": 1})

        out = rl.season_length(
            runs,
            window=1,
            dim="time",
            date=date,
        )
        np.testing.assert_array_equal(np.mean(out.load()), expected)

    @pytest.mark.parametrize(
        "coord,date,end,expected",
        [
            ("dayofyear", "07-01", 210, 211),
            (False, "07-01", 190, 190),
            ("dayofyear", "04-01", 150, np.NaN),  # date falls early
            ("dayofyear", "11-01", 150, 306),  # date ends late
        ],
    )
    @pytest.mark.parametrize("use_dask", [True, False])
    def test_run_end_after_date(
        self, tas_series, coord, date, end, expected, use_dask, ufunc
    ):
        # if use_dask and ufunc:
        #     pytest.xfail("Ufunc run length algorithms not implemented for dask arrays.")

        t = np.zeros(360)
        t[140:end] = 1
        tas = tas_series(t, start="2000-01-01")
        runs = xr.concat((tas, tas), dim="dim0")
        runs = runs == 1

        if use_dask:
            runs = runs.chunk({"time": -1 if ufunc else 10, "dim0": 1})

        out = rl.run_end_after_date(runs, window=1, date=date, dim="time", coord=coord)
        np.testing.assert_array_equal(np.mean(out.load()), expected)

    @pytest.mark.parametrize(
        "coord,date,beg,expected",
        [
            ("dayofyear", "07-01", 210, 211),
            (False, "07-01", 190, 190),
            ("dayofyear", "04-01", False, np.NaN),  # no run
            ("dayofyear", "11-01", 150, 306),  # run already started
        ],
    )
    @pytest.mark.parametrize("use_dask", [True, False])
    def test_first_run_after_date(
        self, tas_series, coord, date, beg, expected, use_dask, ufunc
    ):
        # if use_dask and ufunc:
        #     pytest.xfail("Ufunc run length algorithms not implemented for dask arrays.")
        t = np.zeros(365)
        if beg:
            t[beg:] = 1
        tas = tas_series(t, start="2000-01-01")
        runs = xr.concat((tas, tas), dim="dim0")
        runs = runs == 1

        if use_dask:
            runs = runs.chunk({"time": -1 if ufunc else 10, "dim0": 1})

        out = rl.first_run_after_date(
            runs, window=1, date=date, dim="time", coord=coord
        )
        np.testing.assert_array_equal(np.mean(out.load()), expected)

    @pytest.mark.parametrize(
        "coord,date,end,expected",
        [
            ("dayofyear", "07-01", 210, 183),
            (False, "07-01", 190, 182),
            ("dayofyear", "04-01", 150, np.NaN),  # date falls early
            ("dayofyear", "11-01", 150, 150),  # date ends late
        ],
    )
    @pytest.mark.parametrize("use_dask", [True, False])
    def test_last_run_before_date(
        self, tas_series, coord, date, end, expected, use_dask, ufunc
    ):
        # if use_dask and ufunc:
        #     pytest.xfail("Ufunc run length algorithms not implemented for dask arrays.")
        t = np.zeros(360)
        t[140:end] = 1
        tas = tas_series(t, start="2000-01-01")
        runs = xr.concat((tas, tas), dim="dim0")
        runs = runs == 1

        if use_dask:
            runs = runs.chunk({"time": -1 if ufunc else 10, "dim0": 1})

        out = rl.last_run_before_date(
            runs, window=1, date=date, dim="time", coord=coord
        )
        np.testing.assert_array_equal(np.mean(out.load()), expected)

    @pytest.mark.parametrize(
        "func",
        [rl.last_run_before_date, rl.run_end_after_date],
    )
    @pytest.mark.parametrize("use_dask", [True, False])
    def test_run_with_dates_no_date(self, tas_series, use_dask, func, ufunc):
        # if use_dask and ufunc:
        #     pytest.xfail("Ufunc run length algorithms not implemented for dask arrays.")
        t = np.ones(90)
        tas = tas_series(t, start="2000-01-01")
        runs = xr.concat((tas, tas), dim="dim0")
        runs = runs == 1

        if use_dask:
            runs = runs.chunk({"time": 10, "dim0": 1})

        out = runs.resample(time="MS").map(func, window=1, date="07-01", dim="time")
        assert out.isnull().all()

    @pytest.mark.parametrize(
        "calendar,expected",
        [("standard", [61, 60]), ("365_day", [60, 60]), ("366_day", [61, 61])],
    )
    def test_run_with_dates_different_calendars(self, calendar, expected):
        time = xr.cftime_range(
            "2004-01-01", end="2005-12-31", freq="D", calendar=calendar
        )
        tas = np.zeros(time.size)
        start = np.where((time.day == 1) & (time.month == 3))[0]
        tas[start[0] : start[0] + 250] = 5
        tas[start[1] : start[1] + 250] = 5
        tas = xr.DataArray(tas, coords={"time": time}, dims=("time",))
        out = (
            (tas > 0)
            .resample(time="AS-MAR")
            .map(rl.first_run_after_date, date="03-01", window=2)
        )
        np.testing.assert_array_equal(out.values[1:], expected)

        out = (
            (tas > 0)
            .resample(time="AS-MAR")
            .map(rl.season_length, date="03-02", window=2)
        )
        np.testing.assert_array_equal(out.values[1:], [250, 250])

        out = (
            (tas > 0)
            .resample(time="AS-MAR")
            .map(rl.run_end_after_date, date="03-03", window=2)
        )
        np.testing.assert_array_equal(out.values[1:], np.array(expected) + 250)

        out = (
            (tas > 0)
            .resample(time="AS-MAR")
            .map(rl.last_run_before_date, date="03-02", window=2)
        )
        np.testing.assert_array_equal(out.values[1:], np.array(expected) + 1)

    @pytest.mark.parametrize(
        "func", [rl.first_run_after_date, rl.season_length, rl.run_end_after_date]
    )
    def test_too_many_dates(self, func, tas_series):
        tas = tas_series(np.zeros(730), start="2000-01-01")
        with pytest.raises(ValueError, match="More than 1 instance of date"):
            func((tas == 0), date="03-01", window=5)


@pytest.mark.parametrize("use_dask", [True, False])
def test_lazy_indexing_special_cases(use_dask):
    a = xr.DataArray(np.random.rand(10, 10, 10), dims=("x", "y", "z"))
    b = xr.DataArray(np.random.rand(10, 10, 10), dims=("x", "y", "z"))

    if use_dask:
        a = a.chunk({"y": 5, "z": 5})
        b = b.chunk({"y": 5, "z": 1})

    with pytest.raises(ValueError):
        rl.lazy_indexing(a, b)

    b = b.argmin("x").argmin("y")

    with pytest.raises(ValueError, match="more than one dimension more than index"):
        rl.lazy_indexing(a, b)

    c = xr.DataArray([1], dims=("x",)).chunk()[0]
    b["z"] = np.arange(b.z.size)
    rl.lazy_indexing(b, c)


@pytest.mark.parametrize("use_dask", [True, False])
def test_season(use_dask, tas_series, ufunc):
    # if use_dask and ufunc:
    #     pytest.xfail("Ufunc run length algorithms not implemented for dask arrays.")
    t = np.zeros(360)
    t[140:150] = 1
    tas = tas_series(t, start="2000-01-01")
    runs = xr.concat((tas, tas), dim="dim0")
    runs = runs >= 1

    if use_dask:
        runs = runs.chunk({"time": -1 if ufunc else 10, "dim0": 1})

    out = rl.season(runs, window=2)
    np.testing.assert_array_equal(out.start.load(), [140, 140])
    np.testing.assert_array_equal(out.end.load(), [150, 150])
    np.testing.assert_array_equal(out.length.load(), [10, 10])

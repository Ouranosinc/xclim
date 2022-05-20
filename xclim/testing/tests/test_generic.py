from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from xclim.core.calendar import date_range, doy_to_days_since, select_time
from xclim.indices import generic
from xclim.indices.helpers import day_lengths


class TestSelectResampleOp:
    def test_month(self, q_series):
        q = q_series(np.arange(1000))
        o = generic.select_resample_op(q, "count", freq="YS", month=3)
        np.testing.assert_array_equal(o, 31)

    def test_season_default(self, q_series):
        # Will use freq='YS', so count J, F and D of each year.
        q = q_series(np.arange(1000))
        o = generic.select_resample_op(q, "min", season="DJF")
        assert o[0] == 0
        assert o[1] == 366

    def test_season(self, q_series):
        q = q_series(np.arange(1000))
        o = generic.select_resample_op(q, "count", freq="AS-DEC", season="DJF")
        assert o[0] == 31 + 29


class TestThresholdCount:
    def test_simple(self, tas_series):
        ts = tas_series(np.arange(365))
        out = generic.threshold_count(ts, "<", 50, "Y")
        np.testing.assert_array_equal(out, [50, 0])


class TestDomainCount:
    def test_simple(self, tas_series):
        ts = tas_series(np.arange(365))
        out = generic.domain_count(ts, low=10, high=20, freq="Y")
        np.testing.assert_array_equal(out, [10, 0])


def test_doyminmax(q_series):
    a = np.ones(365)
    a[9] = 2
    a[19] = -2
    a[39] = 4
    a[49] = -4
    q = q_series(a)
    dmx = generic.doymax(q)
    dmn = generic.doymin(q)
    assert dmx.values == [40]
    assert dmn.values == [50]
    for da in [dmx, dmn]:
        for attr in ["units", "is_dayofyear", "calendar"]:
            assert attr in da.attrs.keys()
        assert da.attrs["units"] == ""
        assert da.attrs["is_dayofyear"] == 1


class TestAggregateBetweenDates:
    def test_calendars(self):
        # generate test DataArray
        time_std = date_range("1991-07-01", "1993-06-30", freq="D", calendar="standard")
        time_365 = date_range("1991-07-01", "1993-06-30", freq="D", calendar="noleap")
        data_std = xr.DataArray(
            np.ones((time_std.size, 4)),
            dims=("time", "lon"),
            coords={"time": time_std, "lon": [-72, -71, -70, -69]},
        )
        # generate test start and end dates
        start_v = [[200, 200, np.nan, np.nan], [200, 200, 60, 60]]
        end_v = [[200, np.nan, 60, np.nan], [360, 60, 360, 80]]
        start_std = xr.DataArray(
            start_v,
            dims=("time", "lon"),
            coords={"time": [time_std[0], time_std[366]], "lon": data_std.lon},
            attrs={"calendar": "standard", "is_dayofyear": 1},
        )
        end_std = xr.DataArray(
            end_v,
            dims=("time", "lon"),
            coords={"time": [time_std[0], time_std[366]], "lon": data_std.lon},
            attrs={"calendar": "standard", "is_dayofyear": 1},
        )

        end_noleap = xr.DataArray(
            end_v,
            dims=("time", "lon"),
            coords={"time": [time_365[0], time_365[365]], "lon": data_std.lon},
            attrs={"calendar": "noleap", "is_dayofyear": 1},
        )

        out = generic.aggregate_between_dates(
            data_std, start_std, end_std, op="sum", freq="AS-JUL"
        )

        # expected output
        s = doy_to_days_since(start_std)
        e = doy_to_days_since(end_std)
        expected = e - s
        expected = xr.where(((s > e) | (s.isnull()) | (e.isnull())), np.nan, expected)

        np.testing.assert_allclose(out, expected)

        # check calendar convertion
        out_noleap = generic.aggregate_between_dates(
            data_std, start_std, end_noleap, op="sum", freq="AS-JUL"
        )

        np.testing.assert_allclose(out, out_noleap)

    def test_time_length(self):
        # generate test DataArray
        time_data = date_range(
            "1991-01-01", "1993-12-31", freq="D", calendar="standard"
        )
        time_start = date_range(
            "1990-01-01", "1992-12-31", freq="D", calendar="standard"
        )
        time_end = date_range("1991-01-01", "1993-12-31", freq="D", calendar="standard")
        data = xr.DataArray(
            np.ones((time_data.size, 4)),
            dims=("time", "lon"),
            coords={"time": time_data, "lon": [-72, -71, -70, -69]},
        )
        # generate test start and end dates
        start_v = [[200, 200, np.nan, np.nan], [200, 200, 60, 60], [150, 100, 40, 10]]
        end_v = [[200, np.nan, 60, np.nan], [360, 60, 360, 80], [200, 200, 60, 50]]
        start = xr.DataArray(
            start_v,
            dims=("time", "lon"),
            coords={
                "time": [time_start[0], time_start[365], time_start[730]],
                "lon": data.lon,
            },
            attrs={"calendar": "standard", "is_dayofyear": 1},
        )
        end = xr.DataArray(
            end_v,
            dims=("time", "lon"),
            coords={
                "time": [time_end[0], time_end[365], time_end[731]],
                "lon": data.lon,
            },
            attrs={"calendar": "standard", "is_dayofyear": 1},
        )

        out = generic.aggregate_between_dates(data, start, end, op="sum", freq="YS")

        # expected output
        s = doy_to_days_since(start)
        e = doy_to_days_since(end)
        expected = e - s
        expected[1, 1] = np.nan

        np.testing.assert_allclose(out[0:2], expected)
        np.testing.assert_allclose(out[2], np.array([np.nan, np.nan, np.nan, np.nan]))

    def test_frequency(self):
        # generate test DataArray
        time_data = date_range(
            "1991-01-01", "1992-05-31", freq="D", calendar="standard"
        )
        data = xr.DataArray(
            np.ones((time_data.size, 2)),
            dims=("time", "lon"),
            coords={"time": time_data, "lon": [-70, -69]},
        )
        # generate test start and end dates
        start_v = [[70, 100], [200, 200], [270, 300], [35, 35], [80, 80]]
        end_v = [[130, 70], [200, np.nan], [330, 270], [35, np.nan], [150, 150]]
        end_m_v = [[20, 20], [40, 40], [80, 80], [100, 100], [130, 130]]
        start = xr.DataArray(
            start_v,
            dims=("time", "lon"),
            coords={
                "time": [
                    time_data[59],
                    time_data[151],
                    time_data[243],
                    time_data[334],
                    time_data[425],
                ],
                "lon": data.lon,
            },
            attrs={"calendar": "standard", "is_dayofyear": 1},
        )
        end = xr.DataArray(
            end_v,
            dims=("time", "lon"),
            coords={
                "time": [
                    time_data[59],
                    time_data[151],
                    time_data[243],
                    time_data[334],
                    time_data[425],
                ],
                "lon": data.lon,
            },
            attrs={"calendar": "standard", "is_dayofyear": 1},
        )
        end_m = xr.DataArray(
            end_m_v,
            dims=("time", "lon"),
            coords={
                "time": [
                    time_data[0],
                    time_data[31],
                    time_data[59],
                    time_data[90],
                    time_data[120],
                ],
                "lon": data.lon,
            },
            attrs={"calendar": "standard", "is_dayofyear": 1},
        )

        out = generic.aggregate_between_dates(data, start, end, op="sum", freq="QS-DEC")

        # expected output
        s = doy_to_days_since(start)
        e = doy_to_days_since(end)
        expected = e - s
        expected = xr.where(expected < 0, np.nan, expected)

        np.testing.assert_allclose(out[0], np.array([np.nan, np.nan]))
        np.testing.assert_allclose(out[1:6], expected)

        with pytest.raises(ValueError):
            generic.aggregate_between_dates(data, start, end_m)

    def test_day_of_year_strings(self):
        # generate test DataArray
        time_data = date_range(
            "1990-08-01", "1995-06-01", freq="D", calendar="standard"
        )
        data = xr.DataArray(
            np.ones(time_data.size),
            dims="time",
            coords={"time": time_data},
        )
        # set start and end dates
        start = "02-01"
        end = "10-31"

        out = generic.aggregate_between_dates(data, start, end, op="sum", freq="YS")

        np.testing.assert_allclose(out, np.array([np.nan, 272, 273, 272, 272, np.nan]))

        # given no freq and only strings for start and end dates
        with pytest.raises(ValueError):
            generic.aggregate_between_dates(data, start, end, op="sum")

        # given a malformed date string
        bad_start = "02-31"
        with pytest.raises(ValueError):
            generic.aggregate_between_dates(data, bad_start, end, op="sum", freq="YS")


class TestDegreeDays:
    def test_simple(self, tas_series):
        tas = tas_series(np.array([-10, 15, 20, 3, 10]) + 273.15)

        out = generic.degree_days(tas, thresh="10 degC", condition=">")
        out_kelvin = generic.degree_days(tas, thresh="283.15 degK", condition=">")

        np.testing.assert_allclose(out, [0, 5, 10, 0, 0])
        np.testing.assert_allclose(out, out_kelvin)


def series(start, end, calendar):
    time = date_range(start, end, calendar=calendar)
    return xr.DataArray([1] * time.size, dims=("time",), coords={"time": time})


def test_select_time_month():
    da = series("1993-01-05", "1994-12-31", "default")

    out = select_time(da, drop=True, month=1)
    exp = xr.concat(
        (
            series("1993-01-05", "1993-01-31", "default"),
            series("1994-01-01", "1994-01-31", "default"),
        ),
        "time",
    )
    xr.testing.assert_equal(out, exp)

    out = select_time(da, month=1)
    xr.testing.assert_equal(out.time, da.time)
    assert out.sum() == 58

    da = series("1993-01-05", "1994-12-30", "360_day")
    out = select_time(da, drop=True, month=[3, 6])
    exp = xr.concat(
        (
            series("1993-03-01", "1993-03-30", "360_day"),
            series("1993-06-01", "1993-06-30", "360_day"),
            series("1994-03-01", "1994-03-30", "360_day"),
            series("1994-06-01", "1994-06-30", "360_day"),
        ),
        "time",
    )
    xr.testing.assert_equal(out, exp)


def test_select_time_season():
    da = series("1993-01-05", "1994-12-31", "default")

    out = select_time(da, drop=True, season="DJF")
    exp = xr.concat(
        (
            series("1993-01-05", "1993-02-28", "default"),
            series("1993-12-01", "1994-02-28", "default"),
            series("1994-12-01", "1994-12-31", "default"),
        ),
        "time",
    )
    xr.testing.assert_equal(out, exp)

    da = series("1993-01-05", "1994-12-31", "365_day")
    out = select_time(da, drop=True, season=["MAM", "SON"])
    exp = xr.concat(
        (
            series("1993-03-01", "1993-05-31", "365_day"),
            series("1993-09-01", "1993-11-30", "365_day"),
            series("1994-03-01", "1994-05-31", "365_day"),
            series("1994-09-01", "1994-11-30", "365_day"),
        ),
        "time",
    )
    xr.testing.assert_equal(out, exp)


def test_select_time_doys():
    da = series("2003-02-13", "2004-12-31", "default")

    out = select_time(da, drop=True, doy_bounds=(360, 75))
    exp = xr.concat(
        (
            series("2003-02-13", "2003-03-16", "default"),
            series("2003-12-26", "2004-03-15", "default"),
            series("2004-12-25", "2004-12-31", "default"),
        ),
        "time",
    )
    xr.testing.assert_equal(out, exp)

    da = series("2003-02-13", "2004-12-31", "proleptic_gregorian")

    out = select_time(da, drop=True, doy_bounds=(25, 80))
    exp = xr.concat(
        (
            series("2003-02-13", "2003-03-21", "proleptic_gregorian"),
            series("2004-01-25", "2004-03-20", "proleptic_gregorian"),
        ),
        "time",
    )
    xr.testing.assert_equal(out, exp)


def test_select_time_dates():
    da = series("2003-02-13", "2004-11-01", "all_leap")
    da = da.where(da.time.dt.dayofyear != 92, drop=True)  # no 04-01

    out = select_time(da, drop=True, date_bounds=("04-01", "12-04"))
    exp = xr.concat(
        (
            series("2003-04-02", "2003-12-04", "all_leap"),
            series("2004-04-02", "2004-11-01", "all_leap"),
        ),
        "time",
    )
    xr.testing.assert_equal(out, exp)

    da = series("2003-02-13", "2005-11-01", "standard")

    out = select_time(da, drop=True, date_bounds=("10-05", "02-29"))
    exp = xr.concat(
        (
            series("2003-02-13", "2003-02-28", "standard"),
            series("2003-10-05", "2004-02-29", "standard"),
            series("2004-10-05", "2005-02-28", "standard"),
            series("2005-10-05", "2005-11-01", "standard"),
        ),
        "time",
    )
    xr.testing.assert_equal(out, exp)


def test_select_time_errors():
    da = series("2003-01-01", "2004-01-01", "standard")

    xr.testing.assert_identical(da, select_time(da))

    with pytest.raises(ValueError, match="Only one method of indexing may be given"):
        select_time(da, season="DJF", month=[3, 4, 5])

    with pytest.raises(ValueError, match="invalid day number provided in cftime."):
        select_time(da, date_bounds=("02-30", "03-03"))

    with pytest.raises(ValueError):
        select_time(da, date_bounds=("02-30",))

    with pytest.raises(TypeError):
        select_time(da, doy_bounds=(300, 203, 202))

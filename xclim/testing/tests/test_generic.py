import cftime
import numpy as np
import pandas as pd
import pytest
import xarray as xr

import xclim as xc
from xclim.core.calendar import date_range
from xclim.indices import generic


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


class TestDailyDownsampler:
    def test_std_calendar(self):

        # standard calendar
        # generate test DataArray
        time_std = pd.date_range("2000-01-01", "2000-12-31", freq="D")
        da_std = xr.DataArray(np.arange(time_std.size), coords=[time_std], dims="time")

        for freq in "YS MS QS-DEC".split():
            resampler = da_std.resample(time=freq)
            grouper = generic.daily_downsampler(da_std, freq=freq)

            x1 = resampler.mean()
            x2 = grouper.mean()

            # add time coords to x2 and change dimension tags to time
            time1 = generic.daily_downsampler(da_std.time, freq=freq).first()
            x2.coords["time"] = ("tags", time1.values)
            x2 = x2.swap_dims({"tags": "time"})
            x2 = x2.sortby("time")

            # assert the values of resampler and grouper are the same
            assert np.allclose(x1.values, x2.values)

    def test_365_day(self):

        # 365_day calendar
        # generate test DataArray
        units = "days since 2000-01-01 00:00"
        time_365 = cftime.num2date(np.arange(0, 1 * 365), units, "365_day")
        da_365 = xr.DataArray(
            np.arange(time_365.size), coords=[time_365], dims="time", name="data"
        )
        units = "days since 2001-01-01 00:00"
        time_std = cftime.num2date(np.arange(0, 1 * 365), units, "standard")
        da_std = xr.DataArray(
            np.arange(time_std.size), coords=[time_std], dims="time", name="data"
        )

        for freq in "YS MS QS-DEC".split():
            resampler = da_std.resample(time=freq)
            grouper = generic.daily_downsampler(da_365, freq=freq)

            x1 = resampler.mean()
            x2 = grouper.mean()

            # add time coords to x2 and change dimension tags to time
            time1 = generic.daily_downsampler(da_365.time, freq=freq).first()
            x2.coords["time"] = ("tags", time1.values)
            x2 = x2.swap_dims({"tags": "time"})
            x2 = x2.sortby("time")

            # assert the values of resampler of non leap year with standard calendar
            # is identical to grouper
            assert np.allclose(x1.values, x2.values)

    def test_360_days(self):
        #
        # 360_day calendar
        #
        units = "days since 2000-01-01 00:00"
        time_360 = cftime.num2date(np.arange(0, 360), units, "360_day")
        da_360 = xr.DataArray(
            np.arange(1, time_360.size + 1), coords=[time_360], dims="time", name="data"
        )

        for freq in "YS MS QS-DEC".split():
            grouper = generic.daily_downsampler(da_360, freq=freq)

            x2 = grouper.mean()

            # add time coords to x2 and change dimension tags to time
            time1 = generic.daily_downsampler(da_360.time, freq=freq).first()
            x2.coords["time"] = ("tags", time1.values)
            x2 = x2.swap_dims({"tags": "time"})
            x2 = x2.sortby("time")

            # assert grouper values == expected values
            target_year = 180.5
            target_month = [n * 30 + 15.5 for n in range(0, 12)]
            target_season = [30.5] + [(n - 1) * 30 + 15.5 for n in [4, 7, 10, 12]]
            target = {"YS": target_year, "MS": target_month, "QS-DEC": target_season}[
                freq
            ]
            assert np.allclose(x2.values, target)


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
        s = xc.core.calendar.doy_to_days_since(start_std)
        e = xc.core.calendar.doy_to_days_since(end_std)
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
        s = xc.core.calendar.doy_to_days_since(start)
        e = xc.core.calendar.doy_to_days_since(end)
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
        s = xc.core.calendar.doy_to_days_since(start)
        e = xc.core.calendar.doy_to_days_since(end)
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


class TestDayLength:
    def test_multiple_lats(self):
        time_data = date_range(
            "1992-12-01", "1994-01-01", freq="D", calendar="standard"
        )
        data = xr.DataArray(
            np.ones((time_data.size, 7)),
            dims=("time", "lat"),
            coords={"time": time_data, "lat": [-60, -45, -30, 0, 30, 45, 60]},
        )

        dl = generic.day_lengths(dates=data.time, lat=data.lat)

        events = dict(
            solstice=[
                ["1992-12-21", [[18.49, 15.43, 13.93, 12.0, 10.07, 8.57, 5.51]]],
                ["1993-06-21", [[5.51, 8.57, 10.07, 12.0, 13.93, 15.43, 18.49]]],
                ["1993-12-21", [[18.49, 15.43, 13.93, 12.0, 10.07, 8.57, 5.51]]],
            ],
            equinox=[
                ["1993-03-20", [[12] * 7]]
            ],  # True equinox on 1993-03-20 at 14:41 GMT. Some relative tolerance is needed.
        )

        for event, evaluations in events.items():
            for e in evaluations:
                if event == "solstice":
                    np.testing.assert_array_almost_equal(
                        dl.sel(time=e[0]).transpose(), np.array(e[1]), 2
                    )
                elif event == "equinox":
                    np.testing.assert_allclose(
                        dl.sel(time=e[0]).transpose(), np.array(e[1]), rtol=2e-1
                    )


class TestDegreeDays:
    def test_simple(self, tas_series):
        tas = tas_series(np.array([-10, 15, 20, 3, 10]) + 273.15)

        out = generic.degree_days(tas, thresh="10 degC", condition=">")
        out_kelvin = generic.degree_days(tas, thresh="283.15 degK", condition=">")

        np.testing.assert_allclose(out, [0, 5, 10, 0, 0])
        np.testing.assert_allclose(out, out_kelvin)

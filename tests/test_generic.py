import cftime
import numpy as np
import pandas as pd
import xarray as xr

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
    assert dmx.units == ""

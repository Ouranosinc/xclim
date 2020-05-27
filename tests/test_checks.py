import logging
from collections import namedtuple
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from xclim import set_options
from xclim.core import checks
from xclim.core.utils import ValidationError
from xclim.indicators.atmos import tg_mean

K2C = 273.15

TESTS_HOME = Path(__file__).absolute().parent
TESTS_DATA = Path(TESTS_HOME, "testdata")
set_options(cf_compliance="raise", data_validation="raise")
TestObj = namedtuple("TestObj", ["test"])


@pytest.mark.parametrize(
    "value,expected", [("a string", "a string"), ("a long string", "a * string")]
)
def test_check_valid_ok(value, expected):
    d = TestObj(value)
    checks.check_valid(d, "test", expected)


@pytest.mark.parametrize(
    "value,expected", [(None, "a string"), ("a long string", "a * strings")]
)
def test_check_valid_raise(value, expected):
    d = TestObj(value)
    with pytest.raises(ValidationError):
        checks.check_valid(d, "test", expected)


class TestDateHandling:
    tas_attrs = {
        "units": "K",
        "cell_methods": "time: mean within days",
        "standard_name": "air_temperature",
    }

    def test_assert_daily(self):
        n = 365  # one day short of a full year
        times = pd.date_range("2000-01-01", freq="1D", periods=n)
        da = xr.DataArray(np.arange(n), [("time", times)], attrs=self.tas_attrs)
        tg_mean(da)

    # Bad frequency
    def test_bad_frequency(self):
        with pytest.raises(ValidationError):
            n = 365
            times = pd.date_range("2000-01-01", freq="12H", periods=n)
            da = xr.DataArray(np.arange(n), [("time", times)], attrs=self.tas_attrs)
            tg_mean(da)

    # Missing one day between the two years
    def test_missing_one_day_between_two_years(self):
        with pytest.raises(ValidationError):
            n = 365
            times = pd.date_range("2000-01-01", freq="1D", periods=n)
            times = times.append(pd.date_range("2001-01-01", freq="1D", periods=n))
            da = xr.DataArray(np.arange(2 * n), [("time", times)], attrs=self.tas_attrs)
            tg_mean(da)

    # Duplicate dates
    def test_duplicate_dates(self):
        with pytest.raises(ValidationError):
            n = 365
            times = pd.date_range("2000-01-01", freq="1D", periods=n)
            times = times.append(pd.date_range("2000-12-29", freq="1D", periods=n))
            da = xr.DataArray(np.arange(2 * n), [("time", times)], attrs=self.tas_attrs)
            tg_mean(da)


def test_cf_compliance_options(tas_series, caplog):
    tas = tas_series(np.ones(365))
    tas.attrs["standard_name"] = "not the right name"

    with caplog.at_level(logging.INFO):
        with set_options(cf_compliance="log"):
            checks.check_valid_temperature(tas, "degK")

            assert all([rec.levelname == "INFO" for rec in caplog.records])
            assert (
                "Variable has a non-conforming standard_name" in caplog.records[0].msg
            )
            assert "Variable has a non-conforming units" in caplog.records[1].msg

    with pytest.warns(UserWarning, match="Variable has a non-conforming"):
        with set_options(cf_compliance="warn"):
            checks.check_valid_temperature(tas, "degK")

    with pytest.raises(
        ValidationError, match="Variable has a non-conforming standard_name"
    ):
        with set_options(cf_compliance="raise"):
            checks.check_valid_temperature(tas, "degK")


class TestMissingAnyFills:
    def test_missing_days(self, tas_series):
        a = np.arange(360.0)
        a[5:10] = np.nan
        ts = tas_series(a)
        out = checks.missing_any(ts, freq="MS")
        assert out[0]
        assert not out[1]

    def test_missing_months(self):
        n = 66
        times = pd.date_range("2001-12-30", freq="1D", periods=n)
        da = xr.DataArray(np.arange(n), [("time", times)])
        miss = checks.missing_any(da, "MS")
        np.testing.assert_array_equal(miss, [True, False, False, True])

    def test_missing_years(self):
        n = 378
        times = pd.date_range("2001-12-31", freq="1D", periods=n)
        da = xr.DataArray(np.arange(n), [("time", times)])
        miss = checks.missing_any(da, "YS")
        np.testing.assert_array_equal(miss, [True, False, True])

    def test_missing_season(self):
        n = 378
        times = pd.date_range("2001-12-31", freq="1D", periods=n)
        da = xr.DataArray(np.arange(n), [("time", times)])
        miss = checks.missing_any(da, "Q-NOV")
        np.testing.assert_array_equal(miss, [True, False, False, False, True])

    def test_to_period_start(self, tasmin_series):
        a = np.zeros(365) + K2C + 5.0
        a[2] -= 20
        ts = tasmin_series(a)
        miss = checks.missing_any(ts, freq="AS-JUL")
        np.testing.assert_equal(miss, [False])

    def test_to_period_end(self, tasmin_series):
        a = np.zeros(365) + K2C + 5.0
        a[2] -= 20
        ts = tasmin_series(a)
        miss = checks.missing_any(ts, freq="A-JUN")
        np.testing.assert_equal(miss, [False])

    def test_month(self, tasmin_series):
        ts = tasmin_series(np.zeros(36))
        miss = checks.missing_any(ts, freq="YS", month=7)
        np.testing.assert_equal(miss, [False])

        miss = checks.missing_any(ts, freq="YS", month=8)
        np.testing.assert_equal(miss, [True])

        with pytest.raises(ValueError, match=r"No data for selected period."):
            checks.missing_any(ts, freq="YS", month=1)

        miss = checks.missing_any(ts, freq="YS", month=[7, 8])
        np.testing.assert_equal(miss, [True])

        ts = tasmin_series(np.zeros(76))
        miss = checks.missing_any(ts, freq="YS", month=[7, 8])
        np.testing.assert_equal(miss, [False])

    def test_season(self, tasmin_series):
        ts = tasmin_series(np.zeros(360))
        miss = checks.missing_any(ts, freq="YS", season="MAM")
        np.testing.assert_equal(miss, [False])

        miss = checks.missing_any(ts, freq="YS", season="JJA")
        np.testing.assert_array_equal(miss, [True, True])

        miss = checks.missing_any(ts, freq="YS", season="SON")
        np.testing.assert_equal(miss, [False])

    def test_no_freq(self, tasmin_series):
        ts = tasmin_series(np.zeros(360))

        miss = checks.missing_any(ts, freq=None)
        np.testing.assert_array_equal(miss, False)

        t = list(range(31))
        t.pop(5)
        ts2 = ts.isel(time=t)
        miss = checks.missing_any(ts2, freq=None)
        np.testing.assert_array_equal(miss, True)

        # With indexer
        miss = checks.missing_any(ts, freq=None, month=[7])
        np.testing.assert_array_equal(miss, False)

        miss = checks.missing_any(ts2, freq=None, month=[7])
        np.testing.assert_array_equal(miss, True)

    def test_hydro(self):
        fn = Path(TESTS_DATA, "Raven", "q_sim.nc")
        ds = xr.open_dataset(fn)
        miss = checks.missing_any(ds.q_sim, freq="YS")
        np.testing.assert_array_equal(miss[:-1], False)
        np.testing.assert_array_equal(miss[-1], True)

        miss = checks.missing_any(ds.q_sim, freq="YS", season="JJA")
        np.testing.assert_array_equal(miss, False)


class TestMissingWMO:
    def test_missing_days(self, tas_series):
        a = np.arange(360.0)
        a[5:7] = np.nan  # Number of missing values under acceptable limit in a month
        a[40:45] = np.nan  # Too many consecutive missing values
        a[70:92:2] = np.nan  # Too many non-consecutive missing values
        ts = tas_series(a)
        out = checks.missing_wmo(ts, freq="MS")
        assert not out[0]
        assert out[1]
        assert out[2]

    def test_missing_days_in_quarter(self, tas_series):
        a = np.arange(360.0)
        a[5:16] = np.nan  # Number of missing values under acceptable limit in a month
        ts = tas_series(a)
        out = checks.missing_wmo(ts, freq="QS-JAN")
        np.testing.assert_array_equal(out, [True, False, False, True])


class TestMissingPct:
    def test_missing_days(self, tas_series):
        a = np.arange(360.0)
        a[5:7] = np.nan  # Number of missing values under acceptable limit in a month
        a[40:45] = np.nan  # Too many missing values
        ts = tas_series(a)
        out = checks.missing_pct(ts, freq="MS", tolerance=0.1)
        assert not out[0]
        assert out[1]


class TestAtLeastNValid:
    def test_at_least_n_valid(self, tas_series):
        a = np.arange(360.0)
        a[5:10] = np.nan  # Number of missing values under acceptable limit in a month
        a[40:55] = np.nan  # Too many missing values
        ts = tas_series(a)
        out = checks.at_least_n_valid(ts, freq="MS", n=20)
        np.testing.assert_array_equal(out[:2], [False, True])

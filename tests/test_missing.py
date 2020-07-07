from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from xclim.core import missing

K2C = 273.15

TESTS_HOME = Path(__file__).absolute().parent
TESTS_DATA = Path(TESTS_HOME, "testdata")


class TestMissingAnyFills:
    def test_missing_days(self, tas_series):
        a = np.arange(360.0)
        a[5:10] = np.nan
        ts = tas_series(a)
        out = missing.missing_any(ts, freq="MS")
        assert out[0]
        assert not out[1]

    def test_missing_months(self):
        n = 66
        times = pd.date_range("2001-12-30", freq="1D", periods=n)
        da = xr.DataArray(np.arange(n), [("time", times)])
        miss = missing.missing_any(da, "MS")
        np.testing.assert_array_equal(miss, [True, False, False, True])

    def test_missing_years(self):
        n = 378
        times = pd.date_range("2001-12-31", freq="1D", periods=n)
        da = xr.DataArray(np.arange(n), [("time", times)])
        miss = missing.missing_any(da, "YS")
        np.testing.assert_array_equal(miss, [True, False, True])

    def test_missing_season(self):
        n = 378
        times = pd.date_range("2001-12-31", freq="1D", periods=n)
        da = xr.DataArray(np.arange(n), [("time", times)])
        miss = missing.missing_any(da, "Q-NOV")
        np.testing.assert_array_equal(miss, [True, False, False, False, True])

    def test_to_period_start(self, tasmin_series):
        a = np.zeros(365) + K2C + 5.0
        a[2] -= 20
        ts = tasmin_series(a)
        miss = missing.missing_any(ts, freq="AS-JUL")
        np.testing.assert_equal(miss, [False])

    def test_to_period_end(self, tasmin_series):
        a = np.zeros(365) + K2C + 5.0
        a[2] -= 20
        ts = tasmin_series(a)
        miss = missing.missing_any(ts, freq="A-JUN")
        np.testing.assert_equal(miss, [False])

    def test_month(self, tasmin_series):
        ts = tasmin_series(np.zeros(36))
        miss = missing.missing_any(ts, freq="YS", month=7)
        np.testing.assert_equal(miss, [False])

        miss = missing.missing_any(ts, freq="YS", month=8)
        np.testing.assert_equal(miss, [True])

        with pytest.raises(ValueError, match=r"No data for selected period."):
            missing.missing_any(ts, freq="YS", month=1)

        miss = missing.missing_any(ts, freq="YS", month=[7, 8])
        np.testing.assert_equal(miss, [True])

        ts = tasmin_series(np.zeros(76))
        miss = missing.missing_any(ts, freq="YS", month=[7, 8])
        np.testing.assert_equal(miss, [False])

    def test_season(self, tasmin_series):
        ts = tasmin_series(np.zeros(360))
        miss = missing.missing_any(ts, freq="YS", season="MAM")
        np.testing.assert_equal(miss, [False])

        miss = missing.missing_any(ts, freq="YS", season="JJA")
        np.testing.assert_array_equal(miss, [True, True])

        miss = missing.missing_any(ts, freq="YS", season="SON")
        np.testing.assert_equal(miss, [False])

    def test_no_freq(self, tasmin_series):
        ts = tasmin_series(np.zeros(360))

        miss = missing.missing_any(ts, freq=None)
        np.testing.assert_array_equal(miss, False)

        t = list(range(31))
        t.pop(5)
        ts2 = ts.isel(time=t)
        miss = missing.missing_any(ts2, freq=None)
        np.testing.assert_array_equal(miss, True)

        # With indexer
        miss = missing.missing_any(ts, freq=None, month=[7])
        np.testing.assert_array_equal(miss, False)

        miss = missing.missing_any(ts2, freq=None, month=[7])
        np.testing.assert_array_equal(miss, True)

    def test_hydro(self):
        fn = Path(TESTS_DATA, "Raven", "q_sim.nc")
        ds = xr.open_dataset(fn)
        miss = missing.missing_any(ds.q_sim, freq="YS")
        np.testing.assert_array_equal(miss[:-1], False)
        np.testing.assert_array_equal(miss[-1], True)

        miss = missing.missing_any(ds.q_sim, freq="YS", season="JJA")
        np.testing.assert_array_equal(miss, False)


class TestMissingWMO:
    def test_missing_days(self, tas_series):
        a = np.arange(360.0)
        a[5:7] = np.nan  # Number of missing values under acceptable limit in a month
        a[40:45] = np.nan  # Too many consecutive missing values
        a[70:92:2] = np.nan  # Too many non-consecutive missing values
        ts = tas_series(a)
        out = missing.missing_wmo(ts, freq="MS")
        assert not out[0]
        assert out[1]
        assert out[2]

    def test_missing_days_in_quarter(self, tas_series):
        a = np.arange(360.0)
        a[5:16] = np.nan  # Number of missing values under acceptable limit in a month
        ts = tas_series(a)
        out = missing.missing_wmo(ts, freq="QS-JAN")
        np.testing.assert_array_equal(out, [True, False, False, True])


class TestMissingPct:
    def test_missing_days(self, tas_series):
        a = np.arange(360.0)
        a[5:7] = np.nan  # Number of missing values under acceptable limit in a month
        a[40:45] = np.nan  # Too many missing values
        ts = tas_series(a)
        out = missing.missing_pct(ts, freq="MS", tolerance=0.1)
        assert not out[0]
        assert out[1]


class TestAtLeastNValid:
    def test_at_least_n_valid(self, tas_series):
        a = np.arange(360.0)
        a[5:10] = np.nan  # Number of missing values under acceptable limit in a month
        a[40:55] = np.nan  # Too many missing values
        ts = tas_series(a)
        out = missing.at_least_n_valid(ts, freq="MS", n=20)
        np.testing.assert_array_equal(out[:2], [False, True])

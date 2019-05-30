import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xclim.atmos import tg_mean
from xclim.testing.common import tas_series, tasmin_series
from xclim import checks

TAS_SERIES = tas_series
TASMIN_SERIES = tasmin_series

K2C = 273.15


class TestDateHandling:

    def test_assert_daily(self):
        n = 365  # one day short of a full year
        times = pd.date_range('2000-01-01', freq='1D', periods=n)
        da = xr.DataArray(np.arange(n), [('time', times)], attrs={'units': 'K'})
        tg_mean(da)

    # Bad frequency
    def test_bad_frequency(self):
        with pytest.raises(ValueError):
            n = 365
            times = pd.date_range('2000-01-01', freq='12H', periods=n)
            da = xr.DataArray(np.arange(n), [('time', times)], attrs={'units': 'K'})
            tg_mean(da)

    # Missing one day between the two years
    def test_missing_one_day_between_two_years(self):
        with pytest.raises(ValueError):
            n = 365
            times = pd.date_range('2000-01-01', freq='1D', periods=n)
            times = times.append(pd.date_range('2001-01-01', freq='1D', periods=n))
            da = xr.DataArray(np.arange(2 * n), [('time', times)], attrs={'units': 'K'})
            tg_mean(da)

    # Duplicate dates
    def test_duplicate_dates(self):
        with pytest.raises(ValueError):
            n = 365
            times = pd.date_range('2000-01-01', freq='1D', periods=n)
            times = times.append(pd.date_range('2000-12-29', freq='1D', periods=n))
            da = xr.DataArray(np.arange(2 * n), [('time', times)], attrs={'units': 'K'})
            tg_mean(da)


class TestMissingAnyFills:

    def test_missing_days(self, tas_series):
        a = np.arange(360.)
        a[5:10] = np.nan
        ts = tas_series(a)
        out = checks.missing_any(ts, freq='MS')
        assert out[0]
        assert not out[1]

    def test_missing_months(self):
        n = 66
        times = pd.date_range('2001-12-30', freq='1D', periods=n)
        da = xr.DataArray(np.arange(n), [('time', times)])
        miss = checks.missing_any(da, 'MS')
        np.testing.assert_array_equal(miss, [True, False, False, True])

    def test_missing_years(self):
        n = 378
        times = pd.date_range('2001-12-31', freq='1D', periods=n)
        da = xr.DataArray(np.arange(n), [('time', times)])
        miss = checks.missing_any(da, 'YS')
        np.testing.assert_array_equal(miss, [True, False, True])

    def test_missing_season(self):
        n = 378
        times = pd.date_range('2001-12-31', freq='1D', periods=n)
        da = xr.DataArray(np.arange(n), [('time', times)])
        miss = checks.missing_any(da, 'Q-NOV')
        np.testing.assert_array_equal(miss, [True, False, False, False, True])

    def test_to_period_start(self, tasmin_series):
        a = np.zeros(365) + K2C + 5.0
        a[2] -= 20
        ts = tasmin_series(a)
        miss = checks.missing_any(ts, freq='AS-JUL')
        np.testing.assert_equal(miss, [False])

    def test_to_period_end(self, tasmin_series):
        a = np.zeros(365) + K2C + 5.0
        a[2] -= 20
        ts = tasmin_series(a)
        miss = checks.missing_any(ts, freq='A-JUN')
        np.testing.assert_equal(miss, [False])

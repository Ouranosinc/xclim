import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xclim.temperature import TGMean
from common import tas_series
from xclim import checks


class TestDateHandling:

    def test_assert_daily(self):
        tg_mean =  TGMean()
        n = 365  # one day short of a full year
        times = pd.date_range('2000-01-01', freq='1D', periods=n)
        da = xr.DataArray(np.arange(n), [('time', times)])
        assert tg_mean(da)

    # Bad frequency
    def test_bad_frequency(self):
        with pytest.raises(ValueError):
            tg_mean = TGMean()
            n = 365
            times = pd.date_range('2000-01-01', freq='12H', periods=n)
            da = xr.DataArray(np.arange(n), [('time', times)])
            tg_mean(da)

    # Missing one day between the two years
    def test_missing_one_day_between_two_years(self):
        with pytest.raises(ValueError):
            tg_mean = TGMean()
            n = 365
            times = pd.date_range('2000-01-01', freq='1D', periods=n)
            times = times.append(pd.date_range('2001-01-01', freq='1D', periods=n))
            da = xr.DataArray(np.arange(2 * n), [('time', times)])
            tg_mean(da)

    # Duplicate dates
    def test_duplicate_dates(self):
        with pytest.raises(ValueError):
            tg_mean = TGMean()
            n = 365
            times = pd.date_range('2000-01-01', freq='1D', periods=n)
            times = times.append(pd.date_range('2000-12-29', freq='1D', periods=n))
            da = xr.DataArray(np.arange(2 * n), [('time', times)])
            tg_mean(da)


class TestMissingAnyFills:

    def test_missing_months(self):
        n = 66
        times = pd.date_range('2001-12-30', freq='1D', periods=n)
        da = xr.DataArray(np.arange(n), [('time', times)])
        miss = checks.missing_any_fill(da, 'MS')
        np.testing.assert_array_equal(miss, [True, False, False, True])

    def test_missing_years(self):
        n = 378
        times = pd.date_range('2001-12-31', freq='1D', periods=n)
        da = xr.DataArray(np.arange(n), [('time', times)])
        miss = checks.missing_any_fill(da, 'YS')
        np.testing.assert_array_equal(miss, [True, False, True])

    def test_missing_season(self):
        n = 378
        times = pd.date_range('2001-12-31', freq='1D', periods=n)
        da = xr.DataArray(np.arange(n), [('time', times)])
        miss = checks.missing_any_fill(da, 'Q-NOV')
        np.testing.assert_array_equal(miss, [True, False, False, False, True])


def test_missing_any(tas_series):
    a = np.arange(360.)
    a[5:10] = np.nan
    ts = tas_series(a)
    out = checks.missing_any(ts, freq='MS')
    assert out[0]
    assert not out[1]

    n = 66
    times = pd.date_range('2001-12-30', freq='1D', periods=n)
    da = xr.DataArray(np.arange(n), [('time', times)])
    miss = checks.missing_any(da, 'MS')
    np.testing.assert_array_equal(miss, [True, False, False, True])

    n = 378
    times = pd.date_range('2001-12-31', freq='1D', periods=n)
    da = xr.DataArray(np.arange(n), [('time', times)])
    miss = checks.missing_any(da, 'YS')
    np.testing.assert_array_equal(miss, [True, False, True])

    miss = checks.missing_any(da, 'Q-NOV')
    np.testing.assert_array_equal(miss, [True,


# TODO: Add tests for check_is_dataarray() using functools
# class TestValidDataFormat:
#
#     def time_series(self, values):
#         coords = pd.date_range('7/1/2000', periods=len(values), freq=pd.DateOffset(days=1))
#         return xr.DataArray(values, coords=[coords, ], dims='time',
#                             attrs={'standard_name': 'precipitation_flux',
#                                    'cell_methods': 'time: sum (interval: 1 day)',
#                                    'units': 'mm'})

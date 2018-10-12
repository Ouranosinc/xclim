import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xclim.indices import tg_mean
from xclim import checks


def test_assert_daily():
    n = 365  # one day short of a full year
    times = pd.date_range('2000-01-01', freq='1D', periods=n)
    da = xr.DataArray(np.arange(n), [('time', times)])
    tg_mean(da)

    # Bad frequency
    with pytest.raises(ValueError):
        times = pd.date_range('2000-01-01', freq='12H', periods=n)
        da = xr.DataArray(np.arange(n), [('time', times)])
        tg_mean(da)

    # Missing one day between the two years
    with pytest.raises(ValueError):
        times = pd.date_range('2000-01-01', freq='1D', periods=n)
        times = times.append(pd.date_range('2001-01-01', freq='1D', periods=n))
        da = xr.DataArray(np.arange(2*n), [('time', times)])
        tg_mean(da)

    # Duplicate dates
    with pytest.raises(ValueError):
        times = pd.date_range('2000-01-01', freq='1D', periods=n)
        times = times.append(pd.date_range('2000-12-29', freq='1D', periods=n))
        da = xr.DataArray(np.arange(2*n), [('time', times)])
        tg_mean(da)


def test_missing_any_fill():
    n = 66
    times = pd.date_range('2001-12-30', freq='1D', periods=n)
    da = xr.DataArray(np.arange(n), [('time', times)])
    miss = checks.missing_any_fill(da, 'MS')
    np.testing.assert_array_equal(miss, [True, False, False, True])

    n = 378
    times = pd.date_range('2001-12-31', freq='1D', periods=n)
    da = xr.DataArray(np.arange(n), [('time', times)])
    miss = checks.missing_any_fill(da, 'YS')
    np.testing.assert_array_equal(miss, [True, False, True])

    miss = checks.missing_any_fill(da, 'Q-NOV')
    np.testing.assert_array_equal(miss, [True, False, False, False, True])

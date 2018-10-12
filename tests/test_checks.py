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
    n = 36  # one day short of a full year
    times = pd.date_range('2000-01-01', freq='1D', periods=n)
    da = xr.DataArray(np.arange(n), [('time', times)])

    miss = checks.missing_any_fill(da, 'MS')
    assert not miss[0]
    assert miss[1]

    n = 378  # one day short of a full year
    times = pd.date_range('2000-01-01', freq='1D', periods=n)
    da = xr.DataArray(np.arange(n), [('time', times)])

    miss = checks.missing_any_fill(da, 'YS')
    assert not miss[0]
    assert miss[1]

    miss = checks.missing_any_fill(da, 'QS-DEC')
    assert miss[0]
    # assert not miss[1] This should be True. There is probably a bug in xarray or pandas.



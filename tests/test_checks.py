import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xclim.indices import tg_mean


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








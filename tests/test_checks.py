import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xclim.temperature import TGMean
from xclim import checks
from common import tas_series


def test_assert_daily():
    tg_mean = TGMean()
    n = 365.  # one day short of a full year
    times = pd.date_range('2000-01-01', freq='1D', periods=n)
    da = xr.DataArray(np.arange(n), [('time', times)], attrs={'units': 'degK'})
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


def test_missing_any(tas_series):
    a = np.arange(360.)
    a[5:10] = np.nan

    ts = tas_series(a)

    out = checks.missing_any(ts, freq='MS')
    assert out[0]
    assert not out[1]

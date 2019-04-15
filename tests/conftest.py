import pytest
import pandas as pd
import xarray as xr
import numpy as np


@pytest.fixture
def q_series():
    def _q_series(values, start='1/1/2000'):
        coords = pd.date_range(start, periods=len(values), freq=pd.DateOffset(days=1))
        return xr.DataArray(values, coords=[coords, ], dims='time', name='q',
                            attrs={'standard_name': 'dis',
                                   'units': 'm3 s-1'})

    return _q_series


@pytest.fixture
def ndq_series():
    nx, ny, nt = 2, 3, 5000
    x = np.arange(0, nx)
    y = np.arange(0, ny)

    cx = xr.IndexVariable('x', x)
    cy = xr.IndexVariable('y', y)
    dates = pd.date_range('1900-01-01', periods=nt, freq=pd.DateOffset(days=1))

    time = xr.IndexVariable('time', dates, attrs={'units': 'days since 1900-01-01', 'calendar': 'standard'})

    return xr.DataArray(np.random.lognormal(10, 1, (nt, nx, ny)),
                        dims=('time', 'x', 'y'),
                        coords={'time': time, 'x': cx, 'y': cy},
                        attrs={'units': 'm^3 s-1',
                               'standard_name': 'streamflow'}
                        )

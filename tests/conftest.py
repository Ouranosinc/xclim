import pytest
import pandas as pd
import xarray as xr


@pytest.fixture
def q_series():
    def _q_series(values, start='1/1/2000'):
        coords = pd.date_range(start, periods=len(values), freq=pd.DateOffset(days=1))
        return xr.DataArray(values, coords=[coords, ], dims='time', name='q',
                            attrs={'standard_name': 'dis',
                                   'units': 'm3 s-1'})

    return _q_series

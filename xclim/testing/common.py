import pandas as pd
import pytest
import xarray as xr


@pytest.fixture
def tas_series():
    def _tas_series(values, start='7/1/2000'):
        coords = pd.date_range(start, periods=len(values), freq=pd.DateOffset(days=1))
        return xr.DataArray(values, coords=[coords, ], dims='time', name='tas',
                            attrs={'standard_name': 'air_temperature',
                                   'cell_methods': 'time: mean within days',
                                   'units': 'K'})

    return _tas_series


@pytest.fixture
def tasmax_series():
    def _tasmax_series(values, start='7/1/2000'):
        coords = pd.date_range(start, periods=len(values), freq=pd.DateOffset(days=1))
        return xr.DataArray(values, coords=[coords, ], dims='time', name='tasmax',
                            attrs={'standard_name': 'air_temperature',
                                   'cell_methods': 'time: maximum within days',
                                   'units': 'K'})

    return _tasmax_series


@pytest.fixture
def tasmin_series():
    def _tasmin_series(values, start='7/1/2000'):
        coords = pd.date_range(start, periods=len(values), freq=pd.DateOffset(days=1))
        return xr.DataArray(values, coords=[coords, ], dims='time', name='tasmin',
                            attrs={'standard_name': 'air_temperature',
                                   'cell_methods': 'time: minimum within days',
                                   'units': 'K'})

    return _tasmin_series


@pytest.fixture
def pr_series():
    def _pr_series(values, start='7/1/2000'):
        coords = pd.date_range(start, periods=len(values), freq=pd.DateOffset(days=1))
        return xr.DataArray(values, coords=[coords, ], dims='time', name='pr',
                            attrs={'standard_name': 'precipitation_flux',
                                   'cell_methods': 'time: sum over day',
                                   'units': 'kg m-2 s-1'})

    return _pr_series

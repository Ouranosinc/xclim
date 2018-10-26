import pandas as pd
import pytest
import xarray as xr


@pytest.fixture
def tas_series():
    def _tas_series(values, start='7/1/2000'):
        coords = pd.date_range(start, periods=len(values), freq=pd.DateOffset(days=1))
        return xr.DataArray(values, coords=[coords, ], dims='time',
                            attrs={'standard_name': 'tas',
                                   'cell_methods': 'time: mean within days',
                                   'units': 'K'})

    return _tas_series


@pytest.fixture
def tasmax_series():
    def _tasmax_series(values):
        coords = pd.date_range('7/1/2000', periods=len(values), freq=pd.DateOffset(days=1))
        return xr.DataArray(values, coords=[coords, ], dims='time',
                            attrs={'standard_name': 'tasmax',
                                   'cell_methods': 'time: maximum within days',
                                   'units': 'K'})

    return _tasmax_series


@pytest.fixture
def tasmin_series():
    def _tasmin_series(values):
        coords = pd.date_range('7/1/2000', periods=len(values), freq=pd.DateOffset(days=1))
        return xr.DataArray(values, coords=[coords, ], dims='time',
                            attrs={'standard_name': 'tasmin',
                                   'cell_methods': 'time: minimum within days',
                                   'units': 'K'})

    return _tasmin_series


@pytest.fixture
def pr_series():
    def _pr_series(values):
        coords = pd.date_range('7/1/2000', periods=len(values), freq=pd.DateOffset(days=1))
        return xr.DataArray(values, coords=[coords, ], dims='time',
                            attrs={'standard_name': 'pr',
                                   'cell_methods': 'time: sum over day',
                                   'units': 'kg m-2 s-1'})

    return _pr_series

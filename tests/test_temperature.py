import pytest
import xarray as xr
import numpy as np
import pandas as pd
from xclim.temperature import TxMax

@pytest.fixture
def tas_series():

    def _tas_series(values):
        coords = pd.date_range('7/1/2000', periods=len(values), freq=pd.DateOffset(days=1))
        return xr.DataArray(values, coords=[coords, ], dims='time',
                        attrs={'standard_name': 'tasmax',
                               'cell_methods': 'time: maximum within days',
                               'units': 'K'})
    return _tas_series



class Test_TxMax():

    def test_simple(self, tas_series):
        ts = tas_series(np.arange(720))
        I = TxMax()
        I(ts, freq='Y')


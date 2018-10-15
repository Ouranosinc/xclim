import numpy as np
import pandas as pd
import pytest
import xarray as xr

from xclim.temperature import TxMax, TxMin


@pytest.fixture
def tas_series():
    def _tas_series(values):
        coords = pd.date_range('7/1/2000', periods=len(values), freq=pd.DateOffset(days=1))
        return xr.DataArray(values, coords=[coords, ], dims='time',
                            attrs={'standard_name': 'tasmax',
                                   'cell_methods': 'time: maximum within days',
                                   'units': 'K'})

    return _tas_series


class TestTxMax:

    def test_simple(self, tas_series):
        ts = tas_series(np.arange(720))
        tx_obj = TxMax()
        tx_obj(ts, freq='Y')


class TestTxMin:

    def test_simple(self, tas_series):
        ts = tas_series(np.arange(720))
        tx_obj = TxMin()
        tx_obj(ts, freq='Y')

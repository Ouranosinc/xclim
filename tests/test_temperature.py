import numpy as np

from xclim.testing.common import tas_series
from xclim.temperature import TxMax, TxMin

TAS_SERIES = tas_series()


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

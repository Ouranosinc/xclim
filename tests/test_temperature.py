import numpy as np

from tests.common import tas_series

from xclim.temperature import TxMax, TxMin


class TestTxMax(tas_series):

    def test_simple(self, tas_series):
        ts = tas_series(np.arange(720))
        tx_obj = TxMax()
        tx_obj(ts, freq='Y')


class TestTxMin(tas_series):

    def test_simple(self, tas_series):
        ts = tas_series(np.arange(720))
        tx_obj = TxMin()
        tx_obj(ts, freq='Y')

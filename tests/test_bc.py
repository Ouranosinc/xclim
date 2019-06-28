import pytest
import numpy as np
from xclim.bc import qm

class TestQM:
    def test_simple(self, tas_series):
        n = 10000
        r = np.random.rand(n)
        ref = tas_series(r)
        fut = tas_series(r*2)
        d = qm.delta(ref, fut, 20, 'time.month', '+')

# Tests for detrended quantile mapping
import numpy as np

from xclim.downscaling import dqm


class TestDQM:
    def test_mon(self, mon_tas, tas_series, mon_triangular):
        r = 1 + np.random.rand(10000)
        x = tas_series(r)
        noise = np.random.rand(10000) * 1e-6
        y = mon_tas(r + noise)

        # Test train
        dqm.train(x, y, 5, "time.month")

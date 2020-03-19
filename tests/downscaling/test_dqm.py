# Tests for detrended quantile mapping
import numpy as np

from xclim.downscaling import dqm


class TestDQM:
    def test_mon(self, mon_tas, tas_series, mon_triangular):
        n = 10000
        r = 10 + np.random.rand(n)
        x = tas_series(r)  # sim

        # Delta varying with month
        noise = np.random.rand(n) * 1e-6
        y = mon_tas(r + noise)  # obs

        # Train
        qm = dqm.train(x, y, "time.month", nquantiles=5)

        trend = np.linspace(1, 1, n)
        f = tas_series(r + trend)  # fut

        expected = mon_tas(r + noise + trend)

        # Predict on series with trend
        p = dqm.predict(qm, f)
        np.testing.assert_array_almost_equal(p, expected, 3)

# Tests for detrended quantile mapping
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm

from xclim.downscaling import dqm


class TestDQM:
    def test_time(self, tas_series):
        """No temporal grouping"""
        n = 10000
        r = np.random.rand(n)
        x = tas_series(r + 10)

        y = tas_series(norm.ppf(r))

        # Test train
        # qm = y - (x - <x>)
        qm = dqm.train(x, y, group="time", nq=50)
        q = qm.attrs["quantile"]
        q = np.concatenate([q[:1], q, q[-1:]])

        rn = norm.ppf(q)
        expected = rn - q + np.mean(q)

        # Results are not so good at the endpoints
        np.testing.assert_array_almost_equal(qm[2:-2], expected[2:-2], 1)

        # Test predict
        # Accept discrepancies near extremes
        # No trend
        middle = (x > 1e-2) * (x < 0.99)
        p = dqm.predict(x, qm, interp=True, detrend=False)
        np.testing.assert_array_almost_equal(p[middle], y[middle], 1)

        # With trend
        trend = np.linspace(-0.4, 0.4, n)
        xt = tas_series(r + 9 + trend)
        yt = tas_series(norm.ppf(r) + trend)
        pt = dqm.predict(xt, qm, interp=True)
        # yt.plot(label="Expected", alpha=.5); pt.plot(label="Corrected", alpha=.5); plt.legend(); plt.show()
        np.testing.assert_array_almost_equal(pt[middle], yt[middle], 1)

    def test_mon(self, mon_tas, tas_series, mon_triangular):
        """Monthly grouping"""
        n = 10000
        r = 10 + np.random.rand(n)
        x = tas_series(r)  # sim

        # Delta varying with month
        noise = np.random.rand(n) * 1e-6
        y = mon_tas(r + noise)  # obs

        # Train
        qm = dqm.train(x, y, group="time.month", nq=5)

        trend = np.linspace(-0.2, 0.2, n)
        f = tas_series(r + trend)  # fut

        expected = mon_tas(r + noise + trend)

        # Predict on series with trend
        p = dqm.predict(f, qm)
        np.testing.assert_array_almost_equal(p, expected, 1)

import numpy as np
from scipy.stats import norm

from xclim.downscaling import eqm


class TestEQM:
    qm = eqm

    def test_mon_U(self, mon_tas, tas_series, mon_triangular):
        """
        Train on
        sim: U
        obs: U + monthly cycle

        Predict on sim to get obs
        """
        r = 1 + np.random.rand(10000)
        x = tas_series(r)  # sim

        noise = np.random.rand(10000) * 1e-6
        y = mon_tas(r + noise)  # obs

        # Test train
        d = self.qm.train(x, y, "time.month", nq=5)
        md = d.mean(dim="x")
        np.testing.assert_array_almost_equal(md, mon_triangular, 1)
        # TODO: Test individual quantiles

        # Test predict
        p = self.qm.predict(x, d)
        np.testing.assert_array_almost_equal(p, y, 3)

    def test_norm(self, tas_series):
        """Train on
        sim: U
        obs: Normal

        Predict on sim to get obs
        """
        r = np.random.rand(10000)
        x = tas_series(r)

        y = tas_series(norm.ppf(r))

        # Test train
        d = eqm.train(x, y, "time", nq=50)
        q = d.attrs["quantile"]
        q = np.concatenate([q[:1], q, q[-1:]])
        expected = norm.ppf(q) - q

        # Results are not so good at the endpoints
        np.testing.assert_array_almost_equal(d[2:-2], expected[2:-2], 1)

        # Test predict
        # Accept discrepancies near extremes
        middle = (x > 1e-2) * (x < 0.99)
        p = eqm.predict(x[middle], d, interp=True)
        np.testing.assert_array_almost_equal(p, y[middle], 1)

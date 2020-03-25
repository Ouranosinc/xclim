import numpy as np
import pytest
from matplotlib import pyplot as plt
from scipy.stats import norm
from scipy.stats import uniform

from xclim.downscaling import eqm
from xclim.downscaling.utils import ADDITIVE
from xclim.downscaling.utils import get_correction
from xclim.downscaling.utils import MULTIPLICATIVE


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

    @pytest.mark.parametrize("kind", [ADDITIVE, MULTIPLICATIVE])
    def test_quantiles(self, tas_series, pr_series, kind):
        """Train on
        sim: U
        obs: Normal

        Predict on sim to get obs
        """
        if kind == ADDITIVE:
            ts = tas_series
        elif kind == MULTIPLICATIVE:
            ts = pr_series

        u = np.random.rand(10000)

        # Define distributions
        xd = uniform(loc=10, scale=1)
        yd = norm(loc=12, scale=1)

        # Generate random numbers with u so we get exact results for comparison
        x = xd.ppf(u)
        y = yd.ppf(u)

        # Test train
        qm = eqm.train(ts(x), ts(y), "time", nq=50, kind=kind)

        q = qm.attrs["quantile"]
        expected = get_correction(xd.ppf(q), yd.ppf(q), kind)

        # Results are not so good at the endpoints
        np.testing.assert_array_almost_equal(qm[2:-2], expected[1:-1], 1)

        # Test predict
        # Accept discrepancies near extremes
        middle = (x > 1e-2) * (x < 0.99)
        p = eqm.predict(x, qm, interp=True)
        np.testing.assert_array_almost_equal(p[middle], y[middle], 1)

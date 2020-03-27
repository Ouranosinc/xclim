import numpy as np
import pytest
from matplotlib import pyplot as plt
from scipy.stats import norm
from scipy.stats import uniform

from xclim.downscaling import eqm
from xclim.downscaling.utils import ADDITIVE
from xclim.downscaling.utils import apply_correction
from xclim.downscaling.utils import get_correction
from xclim.downscaling.utils import MULTIPLICATIVE


class TestEQM:
    @pytest.mark.parametrize("kind,name", [(ADDITIVE, "tas"), (MULTIPLICATIVE, "pr")])
    def test_quantiles(self, series, kind, name):
        """Train on
        sim: U
        obs: Normal

        Predict on sim to get obs
        """
        u = np.random.rand(10000)

        # Define distributions
        xd = uniform(loc=10, scale=1)
        yd = norm(loc=12, scale=1)

        # Generate random numbers with u so we get exact results for comparison
        x = xd.ppf(u)
        y = yd.ppf(u)

        # Test train
        sx, sy = series(x, name), series(y, name)
        qm = eqm.train(sx, sy, kind=kind, group="time", nq=50)

        q = qm.coords["quantiles"]
        expected = get_correction(xd.ppf(q), yd.ppf(q), kind)

        # Results are not so good at the endpoints
        np.testing.assert_array_almost_equal(qm.qf[1:-1], expected[1:-1], 1)

        # Test predict
        # Accept discrepancies near extremes
        middle = (x > 1e-2) * (x < 0.99)
        p = eqm.predict(sx, qm, interp=True)
        np.testing.assert_array_almost_equal(p[middle], sy[middle], 1)

    @pytest.mark.parametrize("kind,name", [(ADDITIVE, "tas"), (MULTIPLICATIVE, "pr")])
    def test_mon_U(self, mon_series, series, mon_triangular, kind, name):
        """
        Train on
        sim: U
        obs: U + monthly cycle

        Predict on sim to get obs
        """
        u = np.random.rand(10000)

        # Define distributions
        xd = uniform(loc=2, scale=0.1)
        yd = uniform(loc=4, scale=0.1)
        noise = uniform(loc=0, scale=1e-7)

        # Generate random numbers
        x = xd.ppf(u)
        y = yd.ppf(u) + noise.ppf(u)

        # Test train
        sx, sy = series(x, name), mon_series(y, name)
        qm = eqm.train(sx, sy, kind=kind, group="time.month", nq=5)
        mqm = qm.qf.mean(dim="quantiles")
        expected = apply_correction(mon_triangular, 2, kind)
        np.testing.assert_array_almost_equal(mqm, expected, 1)

        # Test predict
        p = eqm.predict(sx, qm)
        np.testing.assert_array_almost_equal(p, sy, 2)

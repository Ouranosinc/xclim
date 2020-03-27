import numpy as np
import pytest
from matplotlib import pyplot as plt
from scipy.stats import norm
from scipy.stats import uniform

from xclim.downscaling import qdm
from xclim.downscaling.utils import ADDITIVE
from xclim.downscaling.utils import apply_correction
from xclim.downscaling.utils import get_correction
from xclim.downscaling.utils import MULTIPLICATIVE


class TestQDM:
    @pytest.mark.parametrize("kind,name", [(ADDITIVE, "tas"), (MULTIPLICATIVE, "pr")])
    def test_quantiles(self, series, kind, name):
        """Train on
        x : U(1,1)
        y : U(1,2)

        """
        u = np.random.rand(10000)

        # Define distributions
        xd = uniform(loc=1, scale=1)
        yd = uniform(loc=2, scale=4)

        # Generate random numbers with u so we get exact results for comparison
        x = xd.ppf(u)
        y = yd.ppf(u)

        # Test train
        sx, sy = series(x, name), series(y, name)
        qm = qdm.train(sx, sy, kind=kind, group="time", nq=10)

        q = qm.coords["quantiles"]
        expected = get_correction(xd.ppf(q), yd.ppf(q), kind)

        # Results are not so good at the endpoints
        np.testing.assert_array_almost_equal(qm.qf.T, expected, 1)

        # Test predict
        # Accept discrepancies near extremes
        middle = (x > 1e-2) * (x < 0.99)
        p = qdm.predict(sx, qm, interp="linear")
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
        xd = uniform(loc=1, scale=1)
        yd = uniform(loc=2, scale=2)
        noise = uniform(loc=0, scale=1e-7)

        # Generate random numbers
        x = xd.ppf(u)
        y = yd.ppf(u) + noise.ppf(u)

        # Test train
        sx, sy = series(x, name), mon_series(y, name)
        qm = qdm.train(sx, sy, kind=kind, group="time.month", nq=40)

        q = qm.coords["quantiles"]
        expected = get_correction(xd.ppf(q), yd.ppf(q), kind)

        expected = apply_correction(
            mon_triangular[:, np.newaxis], expected[np.newaxis, :], kind
        )
        np.testing.assert_array_almost_equal(qm.qf.sel(quantiles=q).T, expected, 1)

        # Test predict
        p = qdm.predict(sx, qm)
        np.testing.assert_array_almost_equal(p, sy, 1)

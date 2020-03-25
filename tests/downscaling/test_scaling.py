import numpy as np
import pytest
from scipy.stats import uniform

from xclim.downscaling import scaling
from xclim.downscaling.utils import ADDITIVE
from xclim.downscaling.utils import apply_correction
from xclim.downscaling.utils import MULTIPLICATIVE


class TestScaling:
    @pytest.mark.parametrize("kind,name", [(ADDITIVE, "tas"), (MULTIPLICATIVE, "pr")])
    def test_time(self, kind, name, series):
        n = 10000
        u = np.random.rand(n)

        xd = uniform(loc=2, scale=1)
        x = xd.ppf(u)

        sx = series(x, name)
        sy = series(apply_correction(x, 2, kind), name)

        d = scaling.train(sx, sy, "time", kind)
        np.testing.assert_array_almost_equal(d, 2)

        p = scaling.predict(sx, d)
        np.testing.assert_array_almost_equal(p, sy)

    @pytest.mark.parametrize("kind,name", [(ADDITIVE, "tas"), (MULTIPLICATIVE, "pr")])
    def test_mon_U(self, mon_series, series, mon_triangular, kind, name):
        n = 10000
        u = np.random.rand(n)

        xd = uniform(loc=2, scale=1)
        x = xd.ppf(u)

        sx = series(x, name)
        sy = mon_series(apply_correction(x, 2, kind), name)

        # Test train
        d = scaling.train(sx, sy, "time.month", kind)
        expected = apply_correction(mon_triangular, 2, kind)
        np.testing.assert_array_almost_equal(d, expected)

        # Test predict
        p = scaling.predict(sx, d)
        np.testing.assert_array_almost_equal(p, sy)

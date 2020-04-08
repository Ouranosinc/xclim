import numpy as np
import pytest
from scipy.stats import uniform

from xclim.downscaling.correction import LOCI
from xclim.downscaling.correction import Scaling
from xclim.downscaling.utils import ADDITIVE
from xclim.downscaling.utils import apply_correction
from xclim.downscaling.utils import MULTIPLICATIVE


@pytest.mark.parametrize("group,dec", (["time", 2], ["time.month", 1]))
class TestLoci:
    def test_time(self, series, group, dec):
        n = 10000
        u = np.random.rand(n)

        xd = uniform(loc=0, scale=3)
        x = xd.ppf(u)

        sim = fut = series(x, "pr")
        y = x * 2
        thresh = 2
        obs_fit = series(y, "pr").where(y > thresh, 0.1)
        obs = series(y, "pr")

        loci = LOCI(group=group, thresh=thresh)
        loci.train(obs_fit, sim)
        np.testing.assert_array_almost_equal(loci.ds.sim_thresh, 1, dec)
        np.testing.assert_array_almost_equal(loci.ds.cf, 2, dec)

        p = loci.predict(fut)
        np.testing.assert_array_almost_equal(p, obs, dec)


class TestScaling:
    @pytest.mark.parametrize("kind,name", [(ADDITIVE, "tas"), (MULTIPLICATIVE, "pr")])
    def test_time(self, kind, name, series):
        n = 10000
        u = np.random.rand(n)

        xd = uniform(loc=2, scale=1)
        x = xd.ppf(u)

        sim = fut = series(x, name)
        obs = series(apply_correction(x, 2, kind), name)

        scaling = Scaling(group="time", kind=kind)
        scaling.train(obs, sim)
        np.testing.assert_array_almost_equal(scaling.ds.cf, 2)

        p = scaling.predict(fut)
        np.testing.assert_array_almost_equal(p, obs)

    @pytest.mark.parametrize("kind,name", [(ADDITIVE, "tas"), (MULTIPLICATIVE, "pr")])
    def test_mon_U(self, mon_series, series, mon_triangular, kind, name):
        n = 10000
        u = np.random.rand(n)

        xd = uniform(loc=2, scale=1)
        x = xd.ppf(u)

        sim = fut = series(x, name)
        obs = mon_series(apply_correction(x, 2, kind), name)

        # Test train
        scaling = Scaling(group="time.month", kind=kind)
        scaling.train(obs, sim)
        expected = apply_correction(mon_triangular, 2, kind)
        np.testing.assert_array_almost_equal(scaling.ds.cf, expected)

        # Test predict
        p = scaling.predict(fut)
        np.testing.assert_array_almost_equal(p, obs)

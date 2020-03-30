import numpy as np
import pytest
from scipy.stats import uniform

from xclim.downscaling import loci


@pytest.mark.parametrize("group,dec", (["time", 2], ["time.month", 1]))
class TestLoci:
    def test_time(self, series, group, dec):
        n = 10000
        u = np.random.rand(n)

        xd = uniform(loc=0, scale=3)
        x = xd.ppf(u)

        sx = series(x, "pr")
        y = x * 2
        thresh = 2
        sy_fit = series(y, "pr").where(y > thresh, 0.1)
        sy = series(y, "pr")

        ds = loci.train(sx, sy_fit, group, thresh=thresh)
        np.testing.assert_array_almost_equal(ds.x_thresh, 1, dec)
        np.testing.assert_array_almost_equal(ds.cf, 2, dec)

        p = loci.predict(sx, ds)
        np.testing.assert_array_almost_equal(p, sy, dec)

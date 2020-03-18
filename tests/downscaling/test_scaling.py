import numpy as np

from xclim.downscaling import scaling


class TestDelta:
    def test_mon(self, mon_tas, tas_series, mon_triangular):
        r = np.random.rand(10000)
        x = tas_series(r)
        y = mon_tas(r)

        # Test train
        d = scaling.train(x, y, "time.month")
        np.testing.assert_array_almost_equal(d, mon_triangular)

        # Test predict
        p = scaling.predict(x, d)
        np.testing.assert_array_almost_equal(p, y)

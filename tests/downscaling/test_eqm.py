import numpy as np

from xclim.downscaling import eqm


# TODO: Just a smoke test, check that it actually does the right thing
class TestEQM:
    def test_mon(self, mon_tas, tas_series, mon_triangular):
        r1 = np.random.rand(10000)
        x = tas_series(r1)
        r2 = np.random.rand(10000)
        y = mon_tas(r2)

        # Test train
        d = eqm.train(x, y, 5, "time.month")
        # np.testing.assert_array_almost_equal(d, mon_triangular)

        # Test predict
        eqm.predict(x, d)
        # np.testing.assert_array_almost_equal(p, y)

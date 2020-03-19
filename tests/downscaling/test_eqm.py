import numpy as np

from xclim.downscaling import eqm


class TestEQM:
    def test_mon(self, mon_tas, tas_series, mon_triangular):
        r = 1 + np.random.rand(10000)
        x = tas_series(r)
        noise = np.random.rand(10000) * 1e-6
        y = mon_tas(r + noise)

        # Test train
        d = eqm.train(x, y, 5, "time.month")
        md = d.mean(dim="x")
        np.testing.assert_array_almost_equal(md, mon_triangular, 1)
        # TODO: Test individual quantiles

        # Test predict
        p = eqm.predict(x, d)
        np.testing.assert_array_almost_equal(p, y, 3)

import numpy as np

import xclim.bc.qm as qm


class TestQM:
    def test_simple(self, tas_series):
        n = 10000
        r = np.random.rand(n)
        ref = tas_series(r)
        fut_add = tas_series(r + 2)
        d = qm.delta(ref, fut_add, 20, "time.month", "+")
        np.testing.assert_array_almost_equal(d, 2)

        out = qm.apply(ref, d)
        np.testing.assert_array_almost_equal(out - ref, 2)

        fut_mul = tas_series(r * 2)
        d = qm.delta(ref, fut_mul, 20, "time.month", "*")
        np.testing.assert_array_almost_equal(d, 2)

        out = qm.apply(ref, d)
        np.testing.assert_array_almost_equal(out / ref, 2)

    def test_interp(self, tas_series):
        n = 10000
        r = np.random.rand(n)
        ref = tas_series(r)
        m = np.sin(ref.time.dt.dayofyear / 366.0 * 2 * np.pi) * 20
        fut_add = tas_series(r + m)
        d = qm.delta(ref, fut_add, 1, "time.month", "+")
        np.testing.assert_array_almost_equal(
            d.sel(quantile=0.4, method="nearest"),
            20 * np.sin(2 * np.pi * (np.arange(12) + 0.5) / 12),
            0,
        )

        out = qm.apply(ref, d, interp=True)
        # The later half of December and beginning of January won't match due to the interpolation
        np.testing.assert_array_almost_equal(out, fut_add, 0)

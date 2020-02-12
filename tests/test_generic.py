import numpy as np
import xarray as xr
from scipy.stats import lognorm

from xclim import generic


class TestFA(object):
    def setup(self):
        self.nx, self.ny = 2, 3
        x = np.arange(0, self.nx)
        y = np.arange(0, self.ny)

        cx = xr.IndexVariable("x", x)
        cy = xr.IndexVariable("y", y)
        time = xr.IndexVariable("time", np.arange(50))

        self.da = xr.DataArray(
            np.random.lognormal(10, 1, (len(time), self.nx, self.ny)),
            dims=("time", "x", "y"),
            coords={"time": time, "x": cx, "y": cy},
        )

    def test_fit(self):
        p = generic.fit(self.da, "lognorm")

        assert p.dims[0] == "dparams"
        assert p.get_axis_num("dparams") == 0
        p0 = lognorm.fit(self.da.values[:, 0, 0])
        np.testing.assert_array_equal(p[:, 0, 0], p0)

        # Check that we can reuse the parameters with scipy distributions
        cdf = lognorm.cdf(0.99, *p.values)
        assert cdf.shape == (self.nx, self.ny)
        assert p.attrs["estimator"] == "Maximum likelihood"

    def test_fa(self):
        T = 10
        q = generic.fa(self.da, T, "lognorm")

        p0 = lognorm.fit(self.da.values[:, 0, 0])
        q0 = lognorm.ppf(1 - 1.0 / T, *p0)
        np.testing.assert_array_equal(q[0, 0, 0], q0)

    def test_fit_nan(self):
        da = self.da.copy()
        da[0, 0, 0] = np.nan
        out_nan = generic.fit(da, "lognorm")
        out_censor = generic.fit(da[1:], "lognorm")
        np.testing.assert_array_equal(
            out_nan.values[:, 0, 0], out_censor.values[:, 0, 0]
        )

    def test_empty(self):
        da = self.da.copy()
        da[:, 0, 0] = np.nan
        out = generic.fit(da, "lognorm").values
        assert np.isnan(out[:, 0, 0]).all()


class TestFrequencyAnalysis:
    def test_simple(self, ndq_series):
        q = ndq_series.copy()
        q[:, 0, 0] = np.nan
        out = generic.frequency_analysis(
            q, mode="max", t=2, dist="genextreme", window=6, freq="YS"
        )
        assert out.dims == ("return_period", "x", "y")
        assert out.shape == (1, 2, 3)
        v = out.values
        assert v.shape == (1, 2, 3)
        assert np.isnan(v[:, 0, 0])
        assert ~np.isnan(v[:, 1, 1])


class TestSelectResampleOp:
    def test_month(self, q_series):
        q = q_series(np.arange(1000))
        o = generic.select_resample_op(q, "count", freq="YS", month=3)
        np.testing.assert_array_equal(o, 31)

    def test_season_default(self, q_series):
        # Will use freq='YS', so count J, F and D of each year.
        q = q_series(np.arange(1000))
        o = generic.select_resample_op(q, "min", season="DJF")
        assert o[0] == 0
        assert o[1] == 366

    def test_season(self, q_series):
        q = q_series(np.arange(1000))
        o = generic.select_resample_op(q, "count", freq="AS-DEC", season="DJF")
        assert o[0] == 31 + 29

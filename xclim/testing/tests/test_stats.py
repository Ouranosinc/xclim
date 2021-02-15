import numpy as np
import pytest
import xarray as xr
from scipy.stats import lognorm, norm

from xclim.indices import stats


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

        self.weibull_min = xr.DataArray(
            [
                4836.6,
                823.6,
                3131.7,
                1343.4,
                709.7,
                610.6,
                3034.2,
                1973,
                7358.5,
                265,
                4590.5,
                5440.4,
                4613.7,
                4763.1,
                115.3,
                5385.1,
                6398.1,
                8444.6,
                2397.1,
                3259.7,
                307.5,
                4607.4,
                6523.7,
                600.3,
                2813.5,
                6119.8,
                6438.8,
                2799.1,
                2849.8,
                5309.6,
                3182.4,
                705.5,
                5673.3,
                2939.9,
                2631.8,
                5002.1,
                1967.3,
                2810.4,
                2948,
                6904.8,
            ],
            dims=("time",),
            coords={"time": xr.IndexVariable("time", np.arange(40))},
        )

        self.genextreme = xr.DataArray(
            [
                279,
                302,
                450,
                272,
                401,
                222,
                311,
                327,
                294,
                299,
                348,
                286,
                492,
                296,
                227,
                437,
                340,
                376,
                444,
                177,
            ],
            dims=("time",),
            coords={"time": xr.IndexVariable("time", np.arange(20))},
        )

    def test_fit(self):
        p = stats.fit(self.da, "lognorm")

        assert p.dims[0] == "dparams"
        assert p.get_axis_num("dparams") == 0
        p0 = lognorm.fit(self.da.values[:, 0, 0])
        np.testing.assert_array_equal(p[:, 0, 0], p0)

        # Check that we can reuse the parameters with scipy distributions
        cdf = lognorm.cdf(0.99, *p.values)
        assert cdf.shape == (self.nx, self.ny)
        assert p.attrs["estimator"] == "Maximum likelihood"

    def test_weibull_min_fit(self):
        """Check ML fit with a series that leads to poor values without good initial conditions."""
        p = stats.fit(self.weibull_min, "weibull_min")
        np.testing.assert_allclose(p, (1.7760067, -322.092552, 4355.262679), 1e-5)

    def test_genextreme_fit(self):
        """Check ML fit with a series that leads to poor values without good initial conditions."""
        p = stats.fit(self.genextreme, "genextreme")
        np.testing.assert_allclose(p, (0.20949, 297.954091, 75.7911863), 1e-5)

    def test_fa(self):
        T = 10
        q = stats.fa(self.da, T, "lognorm")
        assert "return_period" in q.coords
        p0 = lognorm.fit(self.da.values[:, 0, 0])
        q0 = lognorm.ppf(1 - 1.0 / T, *p0)
        np.testing.assert_array_equal(q[0, 0, 0], q0)

    def test_fit_nan(self):
        da = self.da.copy()
        da[0, 0, 0] = np.nan
        out_nan = stats.fit(da, "lognorm")
        out_censor = stats.fit(da[1:], "lognorm")
        np.testing.assert_array_equal(
            out_nan.values[:, 0, 0], out_censor.values[:, 0, 0]
        )

    def test_empty(self):
        da = self.da.copy()
        da[:, 0, 0] = np.nan
        out = stats.fit(da, "lognorm").values
        assert np.isnan(out[:, 0, 0]).all()

    def test_dims_order(self):
        da = self.da.transpose()
        p = stats.fit(da)
        assert p.dims[-1] == "dparams"


class TestPWMFit:
    params = {
        "expon": {"loc": 0.9527273, "scale": 2.2836364},
        "gamma": {"a": 2.295206, "loc": 0, "scale": 1.410054},
        "genextreme": {"c": -0.1555609, "loc": 2.1792884, "scale": 1.3956404},
        "genlogistic": {"k": -0.2738854, "loc": 2.7406580, "scale": 1.0060517},
        "gennorm": {"k": -0.5707506, "loc": 2.6888917, "scale": 1.7664322},
        "genpareto": {"c": -0.1400000, "loc": 0.7928727, "scale": 2.7855796},
        "gumbel_r": {"loc": 2.285519, "scale": 1.647295},
        "kappa4": {
            "h": 2.4727933,
            "k": 0.9719618,
            "loc": -9.0633543,
            "scale": 17.0127900,
        },
        "norm": {"loc": 3.236364, "scale": 2.023820},
        "pearson3": {"skew": 1.646184, "loc": 3.236364, "scale": 2.199489},
        "weibull_min": {"c": 1.1750218, "loc": 0.6740393, "scale": 2.7087887},
    }
    inputs_pdf = [4, 5, 6, 7]

    @pytest.mark.parametrize("dist", stats._lm3_dist_map.keys())
    def test_get_lm3_dist(self, dist):
        """Check that parameterization for lmoments3 and scipy is identical."""
        pytest.importorskip("lmoments3")
        dc = stats.get_dist(dist)
        lm3dc = stats.get_lm3_dist(dist)
        par = self.params[dist]
        expected = dc(**par).pdf(self.inputs_pdf)
        values = lm3dc(**par).pdf(self.inputs_pdf)
        np.testing.assert_array_almost_equal(values, expected)

    @pytest.mark.parametrize("dist", stats._lm3_dist_map.keys())
    def test_pwm_fit(self, dist):
        """Test that the fitted parameters match parameters used to generate a random sample."""
        pytest.importorskip("lmoments3")
        n = 500
        dc = stats.get_dist(dist)
        par = self.params[dist]
        da = xr.DataArray(
            dc(**par).rvs(size=n),
            dims=("time",),
            coords={"time": xr.cftime_range("1980-01-01", periods=n)},
        )
        out = stats.fit(da, dist=dist, method="PWM").compute()

        # Check that values are identical to lmoments3's output dict
        l3dc = stats.get_lm3_dist(dist)
        expected = l3dc.lmom_fit(da.values)
        for key, val in expected.items():
            np.testing.assert_array_equal(out.sel(dparams=key), val, 1)


class TestFrequencyAnalysis:
    def test_simple(self, ndq_series):
        q = ndq_series.copy()
        q[:, 0, 0] = np.nan
        out = stats.frequency_analysis(
            q, mode="max", t=2, dist="genextreme", window=6, freq="YS"
        )
        assert out.dims == ("return_period", "x", "y")
        assert out.shape == (1, 2, 3)
        v = out.values
        assert v.shape == (1, 2, 3)
        assert np.isnan(v[:, 0, 0])
        assert ~np.isnan(v[:, 1, 1])
        assert out.units == "m3 s-1"

        # smoke test when time is not the first dimension
        stats.frequency_analysis(
            q.transpose(), mode="max", t=2, dist="genextreme", window=6, freq="YS"
        )


class TestParametricQuantile:
    def test_synth(self):
        mu = 23
        sigma = 2
        n = 10000
        per = 0.9
        d = norm(loc=mu, scale=sigma)
        r = xr.DataArray(
            d.rvs(n),
            dims=("time",),
            coords={"time": xr.cftime_range(start="1980-01-01", periods=n)},
            attrs={"history": "Mosquito bytes per minute"},
        )
        expected = d.ppf(per)

        p = stats.fit(r, dist="norm")
        q = stats.parametric_quantile(p=p, q=per)

        np.testing.assert_array_almost_equal(q, expected, 1)
        assert "quantile" in q.coords

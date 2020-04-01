# Tests for detrended quantile mapping
import numpy as np
import pytest
from scipy.stats import norm
from scipy.stats import uniform

from xclim.downscaling import base
from xclim.downscaling import dqm
from xclim.downscaling.utils import ADDITIVE
from xclim.downscaling.utils import apply_correction
from xclim.downscaling.utils import get_correction
from xclim.downscaling.utils import invert
from xclim.downscaling.utils import MULTIPLICATIVE


class TestDQM:
    @pytest.mark.parametrize("kind,name", [(ADDITIVE, "tas"), (MULTIPLICATIVE, "pr")])
    def test_quantiles(self, series, kind, name):
        """Train on
        sim: U
        obs: Normal

        Predict on sim to get obs
        """
        ns = 10000
        u = np.random.rand(ns)

        # Define distributions
        xd = uniform(loc=10, scale=1)
        yd = norm(loc=12, scale=1)

        # Generate random numbers with u so we get exact results for comparison
        x = xd.ppf(u)
        y = yd.ppf(u)

        # Test train
        sx, sy = series(x, name), series(y, name)
        qm = dqm.train(sx, sy, kind=kind, group="time", nq=50)

        q = qm.quantiles
        ex = apply_correction(xd.ppf(q), invert(xd.mean(), kind), kind)
        ey = yd.ppf(q)
        expected = get_correction(ex, ey, kind)

        # Results are not so good at the endpoints
        np.testing.assert_array_almost_equal(qm.qf[2:-2], expected[2:-2], 1)

        # Test predict
        # Accept discrepancies near extremes
        middle = (x > 1e-2) * (x < 0.99)
        p = dqm.predict(sx, qm, interp="linear")
        np.testing.assert_array_almost_equal(p[middle], sy[middle], 1)

    @pytest.mark.parametrize("kind,name", [(ADDITIVE, "tas"), (MULTIPLICATIVE, "pr")])
    @pytest.mark.parametrize(
        "spatial_dims", [None, {"lat": np.arange(20), "lon": np.arange(20)}]
    )
    def test_mon_U(self, mon_series, series, mon_triangular, kind, name, spatial_dims):
        """
        Train on
        sim: U
        obs: U + monthly cycle

        Predict on sim to get obs
        """
        n = 10000
        u = np.random.rand(n)

        # Define distributions
        xd = uniform(loc=2, scale=0.1)
        yd = uniform(loc=4, scale=0.1)
        noise = uniform(loc=0, scale=1e-7)

        # Generate random numbers
        x = xd.ppf(u)
        y = yd.ppf(u) + noise.ppf(u)

        # Test train
        sx, sy = series(x, name), mon_series(y, name)

        if spatial_dims:
            sx = sx.expand_dims(**spatial_dims)
            sy = sy.expand_dims(**spatial_dims)

        qm = dqm.train(sx, sy, kind=kind, group="time.month", nq=5)
        mqm = qm.qf.mean(dim="quantiles")

        expected = apply_correction(mon_triangular, 4, kind)

        if spatial_dims:
            mqm = mqm.isel({crd: 0 for crd in spatial_dims.keys()})
        np.testing.assert_array_almost_equal(mqm, expected, 1)

        # Test predict
        trend = np.linspace(-0.2, 0.2, n) + (1 if kind == MULTIPLICATIVE else 0)
        ss = series(apply_correction(x, trend, kind), name)
        sy = mon_series(apply_correction(y, trend, kind), name)

        if spatial_dims:
            ss = ss.expand_dims(**spatial_dims)
            sy = sy.expand_dims(**spatial_dims)

        p = dqm.predict(ss, qm)
        np.testing.assert_array_almost_equal(p, sy, 1)

    def no_test_mon_add(self, mon_series, series, mon_triangular):
        """Monthly grouping"""
        n = 10000
        r = 10 + np.random.rand(n)
        x = series(r)  # sim

        # Delta varying with month
        noise = np.random.rand(n) * 1e-6
        y = mon_series(r + noise)  # obs

        # Train
        qm = dqm.train(x, y, group="time.month", nq=5)

        trend = np.linspace(-0.2, 0.2, n)
        f = series(r + trend)  # fut

        expected = mon_series(r + noise + trend)

        # Predict on series with trend
        p = dqm.predict(f, qm)
        np.testing.assert_array_almost_equal(p, expected, 1)

    def test_base(self, mon_series, series, mon_triangular):
        """Monthly grouping"""
        n = 10000
        r = 10 + np.random.rand(n)
        x = series(r, "tas")  # sim

        # Delta varying with month
        noise = np.random.rand(n) * 1e-6
        y = mon_series(r + noise, "tas")  # obs

        DQM = base.QuantileMapping(
            nquantiles=5,
            group="time.month",
            detrender=base.PolyDetrend(degree=1, kind=ADDITIVE),
            normalize=True,
        )
        DQM.train(x, y)
        # Train
        qm = dqm.train(x, y, group="time.month", nq=5)

        trend = np.linspace(-0.2, 0.2, n)
        f = series(r + trend, "tas")  # fut

        expected = mon_series(r + noise + trend, "tas")

        # Predict on series with trend
        p1 = DQM.predict(f)
        p = dqm.predict(f, qm)
        np.testing.assert_array_almost_equal(p, expected, 1)
        np.testing.assert_array_almost_equal(p, p1, 1)

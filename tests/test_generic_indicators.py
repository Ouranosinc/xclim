from __future__ import annotations

import numpy as np
import pytest

from xclim import generic, set_options
from xclim.core.utils import ValidationError


class TestFit:
    def test_simple(self, pr_ndseries):
        pr = pr_ndseries(np.random.rand(1000, 1, 2))
        ts = generic.stats(pr, freq="YS", op="max")
        p = generic.fit(ts, dist="gumbel_r")
        assert p.attrs["estimator"] == "Maximum likelihood"

    def test_nan(self, pr_series):
        r = np.random.rand(22)
        r[0] = np.nan
        pr = pr_series(r)

        out = generic.fit(pr, dist="norm")
        assert not np.isnan(out.values[0])

    def test_ndim(self, pr_ndseries):
        pr = pr_ndseries(np.random.rand(100, 1, 2))
        out = generic.fit(pr, dist="norm")
        assert out.shape == (2, 1, 2)
        np.testing.assert_array_equal(out.isnull(), False)

    def test_options(self, q_series):
        q = q_series(np.random.rand(19))
        with set_options(missing_options={"at_least_n": {"n": 10}}):
            out = generic.fit(q, dist="norm")
        np.testing.assert_array_equal(out.isnull(), False)


class TestReturnLevel:
    def test_seasonal(self, ndq_series):
        out = generic.return_level(
            ndq_series, mode="max", t=[2, 5], dist="gamma", season="DJF"
        )

        assert out.description == (
            "Frequency analysis for the maximal winter 1-day value estimated using the "
            "gamma distribution."
        )
        assert out.name == "fa_1maxwinter"
        assert out.shape == (2, 2, 3)  # nrt, nx, ny
        np.testing.assert_array_equal(out.isnull(), False)

    def test_any_variable(self, pr_series):
        pr = pr_series(np.random.rand(100))
        out = generic.return_level(pr, mode="max", t=2, dist="gamma")
        assert out.units == pr.units

    def test_no_indexer(self, ndq_series):
        out = generic.return_level(ndq_series, mode="max", t=[2, 5], dist="gamma")
        assert out.description in [
            "Frequency analysis for the maximal annual 1-day value estimated using the gamma distribution."
        ]
        assert out.name == "fa_1maxannual"
        assert out.shape == (2, 2, 3)  # nrt, nx, ny
        np.testing.assert_array_equal(out.isnull(), False)

    def test_q27(self, ndq_series):
        out = generic.return_level(ndq_series, mode="max", t=2, dist="gamma", window=7)
        assert out.shape == (1, 2, 3)

    def test_empty(self, ndq_series):
        q = ndq_series.copy()
        q[:, 0, 0] = np.nan
        out = generic.return_level(
            q, mode="max", t=2, dist="genextreme", window=6, freq="YS"
        )
        assert np.isnan(out.values[:, 0, 0]).all()


class TestStats:
    """See other tests in test_land::TestStats"""

    def test_simple(self, pr_series):
        pr = pr_series(np.random.rand(400))
        out = generic.stats(pr, freq="YS", op="min", season="MAM")
        assert out.units == pr.units

    def test_ndq(self, ndq_series):
        out = generic.stats(ndq_series, freq="YS", op="min", season="MAM")
        assert out.attrs["units"] == "m3 s-1"

    def test_missing(self, ndq_series):
        a = ndq_series
        a = ndq_series.where(~((a.time.dt.dayofyear == 5) * (a.time.dt.year == 1902)))
        assert a.shape == (5000, 2, 3)
        out = generic.stats(a, op="max", month=1)

        np.testing.assert_array_equal(out.sel(time="1900").isnull(), False)
        np.testing.assert_array_equal(out.sel(time="1902").isnull(), True)

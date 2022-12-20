import numpy as np

from xclim import generic, set_options


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


class TestFrequencyAnalysis:
    """See other tests in test_land::Test_FA"""

    def test_any_variable(self, pr_series):
        pr = pr_series(np.random.rand(100))
        out = generic.return_level(pr, mode="max", t=2, dist="gamma")
        assert out.units == pr.units


class TestStats:
    """See other tests in test_land::TestStats"""

    def test_simple(self, pr_series):
        pr = pr_series(np.random.rand(400))
        out = generic.stats(pr, freq="YS", op="min", season="MAM")
        assert out.units == pr.units

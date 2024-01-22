from __future__ import annotations

import numpy as np

from xclim import generic, set_options


class TestFit:
    def test_simple(self, pr_ndseries, random):
        pr = pr_ndseries(random.random((1000, 1, 2)))
        ts = generic.stats(pr, freq="YS", op="max")
        p = generic.fit(ts, dist="gumbel_r")
        assert p.attrs["estimator"] == "Maximum likelihood"
        assert "time" not in p.dims

    def test_nan(self, pr_series, random):
        r = random.random(22)
        r[0] = np.nan
        pr = pr_series(r)

        out = generic.fit(pr, dist="norm")
        assert np.isnan(out.values[0])
        with set_options(check_missing="skip"):
            out = generic.fit(pr, dist="norm")
            assert not np.isnan(out.values[0])

    def test_ndim(self, pr_ndseries, random):
        pr = pr_ndseries(random.random((100, 1, 2)))
        out = generic.fit(pr, dist="norm")
        assert out.shape == (2, 1, 2)
        np.testing.assert_array_equal(out.isnull(), False)

    def test_options(self, q_series, random):
        q = q_series(random.random(19))
        out = generic.fit(q, dist="norm")
        np.testing.assert_array_equal(out.isnull(), False)

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

    def test_any_variable(self, pr_series, random):
        pr = pr_series(random.random(100))
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

    def test_simple(self, pr_series, random):
        pr = pr_series(random.random(400))
        out = generic.stats(pr, freq="YS", op="min", season="MAM")
        assert out.units == pr.units

    def test_ndq(self, ndq_series):
        out = generic.stats(ndq_series, freq="YS", op="min", season="MAM")
        assert out.attrs["units"] == "m3 s-1"

    def test_missing(self, ndq_series):
        a = ndq_series.where(
            ~((ndq_series.time.dt.dayofyear == 5) & (ndq_series.time.dt.year == 1902))
        )
        assert a.shape == (5000, 2, 3)
        out = generic.stats(a, op="max", month=1)

        np.testing.assert_array_equal(out.sel(time="1900").isnull(), False)
        np.testing.assert_array_equal(out.sel(time="1902").isnull(), True)

    def test_3hourly(self, pr_hr_series, random):
        pr = pr_hr_series(random.random(366 * 24)).resample(time="3h").mean()
        out = generic.stats(pr, freq="MS", op="var")
        assert out.units == "kg2 m-4 s-2"
        assert out.long_name == "Variance of variable"

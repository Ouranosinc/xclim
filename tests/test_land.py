import numpy as np
import pytest

from xclim import land


def test_base_flow_index(ndq_series):
    out = land.base_flow_index(ndq_series, freq="YS")
    assert out.attrs["units"] == ""


class Test_FA:
    def test_simple(self, ndq_series):
        out = land.freq_analysis(
            ndq_series, mode="max", t=[2, 5], dist="gamma", season="DJF"
        )
        assert out.long_name == "N-year return period max winter 1-day flow"
        assert out.shape == (2, 2, 3)  # nrt, nx, ny

    def test_no_indexer(self, ndq_series):
        out = land.freq_analysis(ndq_series, mode="max", t=[2, 5], dist="gamma")
        assert out.long_name == "N-year return period max annual 1-day flow"
        assert out.shape == (2, 2, 3)  # nrt, nx, ny

    def test_q27(self, ndq_series):
        out = land.freq_analysis(ndq_series, mode="max", t=2, dist="gamma", window=7)
        assert out.shape == (1, 2, 3)

    def test_empty(self, ndq_series):
        q = ndq_series.copy()
        q[:, 0, 0] = np.nan
        out = land.freq_analysis(
            q, mode="max", t=2, dist="genextreme", window=6, freq="YS"
        )
        assert np.isnan(out.values[:, 0, 0]).all()


class TestStats:
    def test_simple(self, ndq_series):
        out = land.stats(ndq_series, freq="YS", op="min", season="MAM")
        assert out.attrs["units"] == "m^3 s-1"

    def test_missing(self, ndq_series):
        a = ndq_series
        a = ndq_series.where(~((a.time.dt.dayofyear == 5) * (a.time.dt.year == 1902)))
        assert a.shape == (5000, 2, 3)
        out = land.stats(a, op="max", month=1)

        np.testing.assert_array_equal(out.sel(time="1900").isnull(), False)
        np.testing.assert_array_equal(out.sel(time="1902").isnull(), True)


class TestFit:
    def test_simple(self, ndq_series):
        ts = land.stats(ndq_series, freq="YS", op="max")
        p = land.fit(ts, dist="gumbel_r")
        assert p.attrs["estimator"] == "Maximum likelihood"


def test_qdoy_max(ndq_series, q_series):
    out = land.doy_qmax(ndq_series, freq="YS", season="JJA")
    assert out.attrs["units"] == ""

    a = np.ones(450)
    a[100] = 2
    out = land.doy_qmax(q_series(a), freq="YS")
    assert out[0] == 101

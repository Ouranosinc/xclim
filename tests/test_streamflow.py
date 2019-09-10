import numpy as np
import pytest

from xclim import streamflow


def test_base_flow_index(ndq_series):
    out = streamflow.base_flow_index(ndq_series, freq="YS")
    assert out.attrs["units"] == ""


class Test_FA:
    def test_simple(self, ndq_series):
        out = streamflow.freq_analysis(
            ndq_series, mode="max", t=[2, 5], dist="gamma", season="DJF"
        )
        assert out.long_name == "N-year return period max winter 1-day flow"
        assert out.shape == (2, 2, 3)  # nrt, nx, ny

    def test_no_indexer(self, ndq_series):
        out = streamflow.freq_analysis(ndq_series, mode="max", t=[2, 5], dist="gamma")
        assert out.long_name == "N-year return period max annual 1-day flow"
        assert out.shape == (2, 2, 3)  # nrt, nx, ny

    def test_q27(self, ndq_series):
        out = streamflow.freq_analysis(
            ndq_series, mode="max", t=2, dist="gamma", window=7
        )
        assert out.shape == (1, 2, 3)


class TestStats:
    def test_simple(self, ndq_series):
        out = streamflow.stats(ndq_series, freq="YS", op="min", season="MAM")
        assert out.attrs["units"] == "m^3 s-1"

    def test_missing(self, ndq_series):
        a = ndq_series
        a = ndq_series.where(~((a.time.dt.dayofyear == 5) * (a.time.dt.year == 1902)))
        out = streamflow.stats(a, op="max", month=1)

        np.testing.assert_array_equal(out[1].isnull(), False)
        np.testing.assert_array_equal(out[2].isnull(), True)


class TestFit:
    def test_simple(self, ndq_series):
        ts = streamflow.stats(ndq_series, freq="YS", op="max")
        p = streamflow.fit(ts, dist="gumbel_r")
        assert p.attrs["estimator"] == "Maximum likelihood"


def test_qdoy_max(ndq_series, q_series):
    out = streamflow.doy_qmax(ndq_series, freq="YS", season="JJA")
    assert out.attrs["units"] == ""

    a = np.ones(450)
    a[100] = 2
    out = streamflow.doy_qmax(q_series(a), freq="YS")
    assert out[0] == 101

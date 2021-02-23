import numpy as np
import xarray as xr

from xclim import land, set_options


def test_base_flow_index(ndq_series):
    out = land.base_flow_index(ndq_series, freq="YS")
    assert out.attrs["units"] == ""
    assert isinstance(out, xr.DataArray)


def test_rb_flashiness_index(ndq_series):
    out = land.base_flow_index(ndq_series, freq="YS")
    assert out.attrs["units"] == ""
    assert isinstance(out, xr.DataArray)


class Test_FA:
    def test_simple(self, ndq_series):
        out = land.freq_analysis(
            ndq_series, mode="max", t=[2, 5], dist="gamma", season="DJF"
        )
        assert out.long_name == "N-year return period max winter 1-day flow"
        assert out.shape == (2, 2, 3)  # nrt, nx, ny
        np.testing.assert_array_equal(out.isnull(), False)

    def test_no_indexer(self, ndq_series):
        out = land.freq_analysis(ndq_series, mode="max", t=[2, 5], dist="gamma")
        assert out.long_name == "N-year return period max annual 1-day flow"
        assert out.shape == (2, 2, 3)  # nrt, nx, ny
        np.testing.assert_array_equal(out.isnull(), False)

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

    def test_nan(self, q_series):
        r = np.random.rand(22)
        r[0] = np.nan
        q = q_series(r)

        out = land.fit(q, dist="norm")
        assert not np.isnan(out.values[0])

    def test_ndim(self, ndq_series):
        out = land.fit(ndq_series, dist="norm")
        assert out.shape == (2, 2, 3)
        np.testing.assert_array_equal(out.isnull(), False)

    def test_options(self, q_series):
        q = q_series(np.random.rand(19))
        with set_options(missing_options={"at_least_n": {"n": 10}}):
            out = land.fit(q, dist="norm")
        np.testing.assert_array_equal(out.isnull(), False)


def test_qdoy_max(ndq_series, q_series):
    out = land.doy_qmax(ndq_series, freq="YS", season="JJA")
    assert out.attrs["units"] == ""

    a = np.ones(450)
    a[100] = 2
    out = land.doy_qmax(q_series(a), freq="YS")
    assert out[0] == 101

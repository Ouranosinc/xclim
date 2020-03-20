import numpy as np
import pytest
import xarray as xr

from xclim.downscaling import utils as u


def test_jitter_under_thresh():
    da = xr.DataArray([0.5, 2.1, np.nan])
    out = u.jitter_under_thresh(da, 1)

    assert da[0] != out[0]
    assert da[0] < 1
    assert da[0] > 0
    np.testing.assert_allclose(da[1:], out[1:])


def test_nodes():
    x = u.nodes(5)
    assert len(x) == 7
    d = np.diff(x)
    np.testing.assert_almost_equal(d[0], d[1] / 2, 3)

    x = u.nodes(1, eps=None)
    np.testing.assert_almost_equal(x[0], 0.5)


@pytest.mark.parametrize("method", ["nan", "constant"])
def test_extrapolate_qm(make_qm, method):
    qm = make_qm(np.arange(6).reshape(2, 3))
    xq = make_qm(np.arange(6).reshape(2, 3))

    q, x = u.extrapolate_qm(qm, xq, method=method)

    assert isinstance(q, xr.DataArray)
    assert isinstance(q, xr.DataArray)


class TestReindex:
    def test_month(self, make_qm):
        qm = make_qm(np.arange(6).reshape(2, 3))
        xq = make_qm(np.arange(6).reshape(2, 3))

        out = u.reindex(qm, xq, extrapolation="nan")
        np.testing.assert_array_equal(
            out.T,
            [
                [0, np.nan, np.nan],
                [1, 1, np.nan],
                [2, 2, 2],
                [3, 3, 3],
                [np.nan, 4, 4],
                [np.nan, np.nan, 5],
            ],
        )

        out = u.reindex(qm, xq, extrapolation="constant")
        np.testing.assert_array_equal(
            out.T,
            [
                [0, 1, 2],
                [0, 1, 2],
                [1, 1, 2],
                [2, 2, 2],
                [3, 3, 3],
                [3, 4, 4],
                [3, 4, 5],
                [3, 4, 5],
            ],
        )

        assert out.dims == ("month", "x")
        assert isinstance(out.attrs["quantile"], np.ndarray)

    def test_time(self, make_qm):
        qm = make_qm(np.arange(4), "time")
        xq = make_qm(np.arange(4), "time")

        out = u.reindex(qm, xq, extrapolation="nan")
        assert out.dims == ("x",)
        assert isinstance(out.attrs["quantile"], np.ndarray)

        np.testing.assert_array_equal(out.x, xq)

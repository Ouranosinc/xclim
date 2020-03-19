import numpy as np
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

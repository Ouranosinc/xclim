import numpy as np
import pandas as pd
import pytest
import xarray as xr

from xclim.sdba.base import Grouper
from xclim.sdba.processing import adapt_freq, jitter_over_thresh, jitter_under_thresh


def test_jitter_under_thresh():
    da = xr.DataArray([0.5, 2.1, np.nan])
    out = jitter_under_thresh(da, 1)

    assert da[0] != out[0]
    assert da[0] < 1
    assert da[0] > 0
    np.testing.assert_allclose(da[1:], out[1:])


def test_jitter_over_thresh():
    da = xr.DataArray([0.5, 2.1, np.nan])
    out = jitter_over_thresh(da, 2, 3)

    assert da[1] != out[1]
    assert da[1] < 3
    assert da[1] > 2
    np.testing.assert_allclose(da[[0, 2]], out[[0, 2]])


@pytest.mark.parametrize("use_dask", [True, False])
def test_adapt_freq(use_dask):
    time = pd.date_range("1990-01-01", "2020-12-31", freq="D")
    prvals = np.random.randint(0, 100, size=(time.size, 3))
    pr = xr.DataArray(
        prvals, coords={"time": time, "lat": [0, 1, 2]}, dims=("time", "lat")
    )

    if use_dask:
        pr = pr.chunk({"lat": 1})
    group = Grouper("time.month")

    prsim = xr.where(pr < 20, pr / 20, pr)
    prref = xr.where(pr < 10, pr / 20, pr)
    ds_ad = adapt_freq(prsim, prref, thresh=1, group=group)

    # Where the input is considered zero
    input_zeros = ds_ad.sim_ad.where(prsim <= 1)

    # The proportion of corrected values (time.size * 3 * 0.2 is the theoritical number of values under 1 in prsim)
    dP0_out = (input_zeros > 1).sum() / (time.size * 3 * 0.2)
    np.testing.assert_allclose(dP0_out, 0.5, atol=0.1)

    # Assert that corrected values were generated in the range ]1, 20 + tol[
    corrected = (
        input_zeros.where(input_zeros > 1)
        .stack(flat=["lat", "time"])
        .reset_index("flat")
        .dropna("flat")
    )
    assert ((corrected < 20.1) & (corrected > 1)).all()

    # Assert that non-corrected values are untouched
    # Again we add a 0.5 tol because of randomness.
    xr.testing.assert_equal(ds_ad.sim_ad.where(prsim > 20.1), prsim.where(prsim > 20.5))
    # Assert that Pth and dP0 are approx the good values
    np.testing.assert_allclose(ds_ad.pth, 20, rtol=0.05)
    np.testing.assert_allclose(ds_ad.dP0, 0.5, atol=0.14)

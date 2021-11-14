import numpy as np
import pandas as pd
import pytest
import xarray as xr

from xclim.sdba.base import Grouper
from xclim.sdba.processing import (
    adapt_freq,
    escore,
    jitter_over_thresh,
    jitter_under_thresh,
    reordering,
    standardize,
    unstandardize,
)


def test_jitter_under_thresh():
    da = xr.DataArray([0.5, 2.1, np.nan], attrs={"units": "K"})
    out = jitter_under_thresh(da, "1 K")

    assert da[0] != out[0]
    assert da[0] < 1
    assert da[0] > 0
    np.testing.assert_allclose(da[1:], out[1:])
    assert "jitter_under_thresh(<array>, '1 K') - xclim version" in out.attrs["history"]


def test_jitter_over_thresh():
    da = xr.DataArray([0.5, 2.1, np.nan], attrs={"units": "m"})
    out = jitter_over_thresh(da, "200 cm", "0.003 km")

    assert da[1] != out[1]
    assert da[1] < 3
    assert da[1] > 2
    np.testing.assert_allclose(da[[0, 2]], out[[0, 2]])
    assert out.units == "m"


@pytest.mark.parametrize("use_dask", [True, False])
def test_adapt_freq(use_dask):
    time = pd.date_range("1990-01-01", "2020-12-31", freq="D")
    prvals = np.random.randint(0, 100, size=(time.size, 3))
    pr = xr.DataArray(
        prvals,
        coords={"time": time, "lat": [0, 1, 2]},
        dims=("time", "lat"),
        attrs={"units": "mm d-1"},
    )

    if use_dask:
        pr = pr.chunk({"lat": 1})
    group = Grouper("time.month")
    with xr.set_options(keep_attrs=True):
        prsim = xr.where(pr < 20, pr / 20, pr)
        prref = xr.where(pr < 10, pr / 20, pr)
    sim_ad, pth, dP0 = adapt_freq(prref, prsim, thresh="1 mm d-1", group=group)

    # Where the input is considered zero
    input_zeros = sim_ad.where(prsim <= 1)

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
    xr.testing.assert_equal(
        sim_ad.where(prsim > 20.1),
        prsim.where(prsim > 20.5).transpose("lat", "time"),
    )
    # Assert that Pth and dP0 are approx the good values
    np.testing.assert_allclose(pth, 20, rtol=0.05)
    np.testing.assert_allclose(dP0, 0.5, atol=0.14)
    assert sim_ad.units == "mm d-1"
    assert sim_ad.attrs["references"].startswith("Themeßl")
    assert pth.units == "mm d-1"


@pytest.mark.parametrize("use_dask", [True, False])
def test_adapt_freq_add_dims(use_dask):
    time = pd.date_range("1990-01-01", "2020-12-31", freq="D")
    prvals = np.random.randint(0, 100, size=(time.size, 3))
    pr = xr.DataArray(
        prvals,
        coords={"time": time, "lat": [0, 1, 2]},
        dims=("time", "lat"),
        attrs={"units": "mm d-1"},
    )

    if use_dask:
        pr = pr.chunk()
    group = Grouper("time.month", add_dims=["lat"])
    with xr.set_options(keep_attrs=True):
        prsim = xr.where(pr < 20, pr / 20, pr)
        prref = xr.where(pr < 10, pr / 20, pr)
    sim_ad, pth, dP0 = adapt_freq(prref, prsim, thresh="1 mm d-1", group=group)
    assert set(sim_ad.dims) == set(prsim.dims)
    assert "lat" not in pth.dims

    group = Grouper("time.dayofyear", window=5)
    with xr.set_options(keep_attrs=True):
        prsim = xr.where(pr < 20, pr / 20, pr)
        prref = xr.where(pr < 10, pr / 20, pr)
    sim_ad, pth, dP0 = adapt_freq(prref, prsim, thresh="1 mm d-1", group=group)
    assert set(sim_ad.dims) == set(prsim.dims)


def test_escore():

    x = np.array([1, 4, 3, 6, 4, 7, 5, 8, 4, 5, 3, 7]).reshape(2, 6)
    y = np.array([6, 6, 3, 8, 5, 7, 3, 7, 3, 6, 4, 3]).reshape(2, 6)

    x = xr.DataArray(x, dims=("variables", "time"))
    y = xr.DataArray(y, dims=("variables", "time"))

    # Value taken from escore of Cannon's MBC R package.
    out = escore(x, y)
    np.testing.assert_allclose(out, 1.90018550338863)
    assert "escore(" in out.attrs["history"]
    assert out.attrs["references"].startswith("Skezely")


def test_standardize():
    x = np.random.standard_normal((2, 10000))
    x[0, 50] = np.NaN
    x = xr.DataArray(x, dims=("x", "y"), attrs={"units": "m"})

    xp, avg, std = standardize(x, dim="y")

    np.testing.assert_allclose(avg, 0, atol=4e-2)
    np.testing.assert_allclose(std, 1, atol=2e-2)

    xp, avg, std = standardize(x, mean=avg, dim="y")
    np.testing.assert_allclose(std, 1, atol=2e-2)

    y = unstandardize(xp, 0, 1)

    np.testing.assert_allclose(x, y, atol=0.1)
    assert avg.units == xp.units


def test_reordering():
    y = xr.DataArray(np.arange(1, 11), dims=("time",), attrs={"a": 1, "units": "K"})
    x = xr.DataArray(np.arange(10, 20)[::-1], dims=("time",))

    out = reordering(x, y, group="time")

    np.testing.assert_array_equal(out, np.arange(1, 11)[::-1])
    out.attrs.pop("history")
    assert out.attrs == y.attrs

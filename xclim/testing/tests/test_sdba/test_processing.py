from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from xclim.core.units import units
from xclim.sdba.adjustment import EmpiricalQuantileMapping
from xclim.sdba.base import Grouper
from xclim.sdba.processing import (
    adapt_freq,
    escore,
    from_additive_space,
    jitter,
    jitter_over_thresh,
    jitter_under_thresh,
    normalize,
    reordering,
    stack_variables,
    standardize,
    to_additive_space,
    unstack_variables,
    unstandardize,
)


def test_jitter_both():
    da = xr.DataArray([0.5, 2.1, np.nan], attrs={"units": "K"})
    out = jitter(da, lower="1 K", upper="2 K", maximum="3 K")

    assert da[0] != out[0]
    assert da[0] < 1
    assert da[0] > 0

    assert da[1] != out[1]
    assert da[1] < 3
    assert da[1] > 2


def test_jitter_under_thresh():
    da = xr.DataArray([0.5, 2.1, np.nan], attrs={"units": "K"})
    out = jitter_under_thresh(da, "1 K")

    assert da[0] != out[0]
    assert da[0] < 1
    assert da[0] > 0
    np.testing.assert_allclose(da[1:], out[1:])
    assert (
        "jitter(x=<array>, lower='1 K', upper=None, minimum=None, maximum=None) - xclim version"
        in out.attrs["history"]
    )


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

    # The proportion of corrected values (time.size * 3 * 0.2 is the theoretical number of values under 1 in prsim)
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
    np.testing.assert_allclose(dP0, 0.5, atol=0.15)
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
    assert out.attrs["references"].startswith("Székely")


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


def test_to_additive(pr_series, hurs_series):
    # log
    pr = pr_series(np.array([0, 1e-5, 1, np.e**10]))

    with units.context("hydro"):
        prlog = to_additive_space(pr, lower_bound="0 mm/d", trans="log")
    np.testing.assert_allclose(prlog, [-np.Inf, -11.512925, 0, 10])
    assert prlog.attrs["sdba_transform"] == "log"
    assert prlog.attrs["sdba_transform_units"] == "kg m-2 s-1"

    with xr.set_options(keep_attrs=True):
        pr1 = pr + 1
    with units.context("hydro"):
        prlog2 = to_additive_space(pr1, trans="log", lower_bound="1.0 kg m-2 s-1")
    np.testing.assert_allclose(prlog2, [-np.Inf, -11.512925, 0, 10])
    assert prlog2.attrs["sdba_transform_lower"] == 1.0

    # logit
    hurs = hurs_series(np.array([0, 1e-3, 90, 100]))

    hurslogit = to_additive_space(
        hurs, lower_bound="0 %", trans="logit", upper_bound="100 %"
    )
    np.testing.assert_allclose(
        hurslogit, [-np.Inf, -11.5129154649, 2.197224577, np.Inf]
    )
    assert hurslogit.attrs["sdba_transform"] == "logit"
    assert hurslogit.attrs["sdba_transform_units"] == "%"

    with xr.set_options(keep_attrs=True):
        hursscl = hurs * 4 + 200
    hurslogit2 = to_additive_space(
        hursscl, trans="logit", lower_bound="2", upper_bound="6"
    )
    np.testing.assert_allclose(
        hurslogit2, [-np.Inf, -11.5129154649, 2.197224577, np.Inf]
    )
    assert hurslogit2.attrs["sdba_transform_lower"] == 200.0
    assert hurslogit2.attrs["sdba_transform_upper"] == 600.0


def test_from_additive(pr_series, hurs_series):
    # log
    pr = pr_series(np.array([0, 1e-5, 1, np.e**10]))
    with units.context("hydro"):
        pr2 = from_additive_space(
            to_additive_space(pr, lower_bound="0 mm/d", trans="log")
        )
    np.testing.assert_allclose(pr[1:], pr2[1:])
    pr2.attrs.pop("history")
    assert pr.attrs == pr2.attrs

    # logit
    hurs = hurs_series(np.array([0, 1e-5, 0.9, 1]))
    hurs2 = from_additive_space(
        to_additive_space(hurs, lower_bound="0 %", trans="logit", upper_bound="100 %")
    )
    np.testing.assert_allclose(hurs[1:-1], hurs2[1:-1])


def test_normalize(tas_series):
    tas = tas_series(
        np.random.standard_normal((int(365.25 * 36),)) + 273.15, start="2000-01-01"
    )

    xp, norm = normalize(tas, group="time.dayofyear")
    np.testing.assert_allclose(norm, 273.15, atol=1)

    xp2, norm = normalize(tas, norm=norm, group="time.dayofyear")
    np.testing.assert_allclose(xp, xp2)


def test_stack_variables(open_dataset):
    ds1 = open_dataset("sdba/CanESM2_1950-2100.nc")
    ds2 = open_dataset("sdba/ahccd_1950-2013.nc")

    da1 = stack_variables(ds1)
    da2 = stack_variables(ds2)

    assert list(da1.multivar.values) == ["pr", "tasmax"]
    assert da1.multivar.attrs["_standard_name"] == [
        "precipitation_flux",
        "air_temperature",
    ]
    assert da2.multivar.attrs["is_variables"]
    assert da1.multivar.equals(da2.multivar)

    da1p = da1.sortby("multivar", ascending=False)

    with pytest.raises(ValueError, match="Inputs have different multivariate"):
        EmpiricalQuantileMapping.train(da1p, da2)

    ds1p = unstack_variables(da1)

    xr.testing.assert_equal(ds1, ds1p)

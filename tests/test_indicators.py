#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Tests for the Indicator objects
import dask
import numpy as np
import pytest
import xarray as xr

from xclim import __version__
from xclim import atmos
from xclim.core.formatting import AttrFormatter
from xclim.core.formatting import default_formatter
from xclim.core.formatting import merge_attributes
from xclim.core.formatting import parse_doc
from xclim.core.formatting import update_history
from xclim.core.indicator import Indicator
from xclim.core.units import units
from xclim.indices import tg_mean


class UniIndTemp(Indicator):
    identifier = "tmin"
    var_name = "tmin{thresh}"
    units = "K"
    long_name = "{freq} mean surface temperature"
    standard_name = "{freq} mean temperature"
    cell_methods = "time: mean within {freq:noun}"

    @staticmethod
    def compute(da, thresh=0.0, freq="YS"):
        """Docstring"""
        out = da
        out -= thresh
        return out.resample(time=freq).mean(keep_attrs=True)


class UniIndPr(Indicator):
    identifier = "prmax"
    units = "mm/s"
    context = "hydro"

    @staticmethod
    def compute(da, freq):
        """Docstring"""
        return da.resample(time=freq).mean(keep_attrs=True)


def test_attrs(tas_series):
    import datetime as dt

    a = tas_series(np.arange(360.0))
    ind = UniIndTemp()
    txm = ind(a, thresh=5, freq="YS")
    assert txm.cell_methods == "time: mean within days time: mean within years"
    assert f"{dt.datetime.now():%Y-%m-%d %H}" in txm.attrs["history"]
    assert "tmin(da, thresh=5, freq='YS')" in txm.attrs["history"]
    assert f"xclim version: {__version__}." in txm.attrs["history"]
    assert txm.name == "tmin5"


def test_temp_unit_conversion(tas_series):
    a = tas_series(np.arange(360.0))
    ind = UniIndTemp()
    txk = ind(a, freq="YS")

    ind.units = "degC"
    txc = ind(a, freq="YS")

    np.testing.assert_array_almost_equal(txk, txc + 273.15)


def test_json(pr_series):
    ind = UniIndPr()
    meta = ind.json()

    expected = {
        "identifier",
        "var_name",
        "units",
        "long_name",
        "standard_name",
        "cell_methods",
        "keywords",
        "abstract",
        "parameters",
        "description",
        "history",
        "references",
        "comment",
        "notes",
    }

    assert set(meta.keys()).issubset(expected)


def test_signature():
    from inspect import signature

    ind = UniIndTemp()
    assert signature(ind.compute) == signature(ind.__call__)


def test_doc():
    ind = UniIndTemp()
    assert ind.__call__.__doc__ == ind.compute.__doc__


def test_delayed(tasmax_series):
    tasmax = tasmax_series(np.arange(360.0)).chunk({"time": 5})

    tx = UniIndTemp()
    txk = tx(tasmax)

    # Check that the calculations are delayed
    assert isinstance(txk.data, dask.array.core.Array)

    # Same with unit conversion
    tx.required_units = ("C",)
    tx.units = "C"
    txc = tx(tasmax)

    assert isinstance(txc.data, dask.array.core.Array)


def test_identifier():
    with pytest.warns(UserWarning):
        UniIndPr(identifier="t_{}")


def test_formatting(pr_series):
    out = atmos.wetdays(pr_series(np.arange(366)), thresh=1.0 * units.mm / units.day)
    # pint 0.10 now pretty print day as d.
    assert out.attrs["long_name"] in [
        "Number of wet days (precip >= 1 mm/day)",
        "Number of wet days (precip >= 1 mm/d)",
    ]
    out = atmos.wetdays(pr_series(np.arange(366)), thresh=1.5 * units.mm / units.day)
    assert out.attrs["long_name"] in [
        "Number of wet days (precip >= 1.5 mm/day)",
        "Number of wet days (precip >= 1.5 mm/d)",
    ]


# TODO Add a meaningful test
def test_parse_doc():
    parse_doc(tg_mean.__doc__)


def test_default_formatter():
    assert default_formatter.format("{freq}", freq="YS") == "annual"
    assert default_formatter.format("{freq:noun}", freq="MS") == "months"
    assert default_formatter.format("{month}", month="m3") == "march"


def test_AttrFormatter():
    fmt = AttrFormatter(
        mapping={"evil": ["méchant", "méchante"], "nice": ["beau", "belle"]},
        modifiers=["m", "f"],
    )
    # Normal cases
    assert fmt.format("{adj:m}", adj="evil") == "méchant"
    assert fmt.format("{adj:f}", adj="nice") == "belle"
    # Missing mod:
    assert fmt.format("{adj}", adj="evil") == "méchant"
    # Mod with unknown value
    with pytest.raises(ValueError):
        fmt.format("{adj:m}", adj="funny")


@pytest.mark.parametrize("new_line", ["<>", "\n"])
@pytest.mark.parametrize("missing_str", ["<Missing>", None])
def test_merge_attributes(missing_str, new_line):
    a = xr.DataArray([0], attrs={"text": "Text1"}, name="a")
    b = xr.DataArray([0], attrs={})
    c = xr.Dataset(attrs={"text": "Text3"})

    merged = merge_attributes(
        "text", a, missing_str=missing_str, new_line=new_line, b=b, c=c
    )

    assert merged.startswith("a: Text1")

    if missing_str is not None:
        assert merged.count(new_line) == 2
        assert f"b: {missing_str}" in merged
    else:
        assert merged.count(new_line) == 1
        assert "b:" not in merged


def test_update_history():
    a = xr.DataArray([0], attrs={"history": "Text1"}, name="a")
    b = xr.DataArray([0], attrs={"history": "Text2"})
    c = xr.Dataset(attrs={"history": "Text3"})

    merged = update_history("text", a, new_name="d", b=b, c=c)

    assert "d: text" in merged.split("\n")[-1]
    assert merged.startswith("a: Text1")

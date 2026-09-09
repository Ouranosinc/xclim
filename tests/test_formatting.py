from __future__ import annotations

import datetime as dt
import re

import pytest
import xarray as xr

from xclim import __version__
from xclim.core import formatting as fmt


def test_prefix_attrs():
    source = {"units": "mm/s", "name": "pr"}
    dest = fmt.prefix_attrs(source, ["units"], "original_")
    assert "original_units" in dest

    out = fmt.unprefix_attrs(dest, ["units"], "original_")
    assert out == source

    # Check that the "naked" units will be overwritten.
    dest["units"] = ""

    out = fmt.unprefix_attrs(dest, ["units"], "original_")
    assert out == source


def test_update_history():
    a = xr.DataArray([0], attrs={"history": "Text1"}, name="a")
    b = xr.DataArray([0], attrs={"history": "Text2"})
    c = xr.Dataset(attrs={"history": "Text3"})

    merged = fmt.update_history("text", a, new_name="d", b=b, c=c)

    assert "d: text" in merged.split("\n")[0]
    assert "a: Text1" in merged


def test_update_xclim_history(atmosds):
    @fmt.update_xclim_history
    def func(da, arg1, arg2=None, arg3=None):  # noqa: F841
        return da

    out = func(atmosds.tas, 1, arg2=[1, 2], arg3=None)

    matches = re.match(
        r"\[([0-9-:\s]*)]\s(\w*):\s(\w*)\((.*)\)\s-\sxclim\sversion:\s(\d*\.\d*\.\d*[a-zA-Z-]*(\.\d*)?)",
        out.attrs["history"],
    ).groups()

    date = dt.datetime.fromisoformat(matches[0])
    assert dt.timedelta(0) < (dt.datetime.now() - date) < dt.timedelta(seconds=3)
    assert matches[1] == "tas"
    assert matches[2] == "func"
    assert matches[3] == "da=tas, arg1=1, arg2=[1, 2], arg3=None"
    assert matches[4] == __version__


def test_default_formatter():
    assert fmt.default_formatter.format("{freq}", freq="YS") == "annual"
    assert fmt.default_formatter.format("{freq:noun}", freq="MS") == "months"
    assert fmt.default_formatter.format("{month}", month="m3") == "march"


def test_AttrFormatter():
    ft = fmt.AttrFormatter(
        mapping={"evil": ["méchant", "méchante"], "nice": ["beau", "belle"]},
        modifiers=["m", "f"],
    )
    # Normal cases
    assert ft.format("{adj:m}", adj="evil") == "méchant"
    assert ft.format("{adj:f}", adj="nice") == "belle"
    # Missing mod:
    assert ft.format("{adj}", adj="evil") == "méchant"
    # Mod with unknown value
    with pytest.warns(match="Requested formatting `m` for unknown string `funny`."):
        ft.format("{adj:m}", adj="funny")


@pytest.mark.parametrize("new_line", ["<>", "\n"])
@pytest.mark.parametrize("missing_str", ["<Missing>", None])
def test_merge_attributes(missing_str, new_line):
    a = xr.DataArray([0], attrs={"text": "Text1"}, name="a")
    b = xr.DataArray([0], attrs={})
    c = xr.Dataset(attrs={"text": "Text3"})

    merged = fmt.merge_attributes("text", a, missing_str=missing_str, new_line=new_line, b=b, c=c)

    assert merged.startswith("a: Text1")

    if missing_str is not None:
        assert merged.count(new_line) == 2
        assert f"b: {missing_str}" in merged
    else:
        assert merged.count(new_line) == 1
        assert "b:" not in merged

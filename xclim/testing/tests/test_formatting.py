from __future__ import annotations

import datetime as dt
import re

from xclim import __version__
from xclim.core import formatting as fmt
from xclim.indicators.atmos import degree_days_exceedance_date, heat_wave_frequency


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


def test_indicator_docstring():
    doc = heat_wave_frequency.__doc__.split("\n")
    assert doc[0] == "Heat wave frequency (realm: atmos)"
    assert (
        doc[5]
        == "Based on indice :py:func:`~xclim.indices._multivariate.heat_wave_frequency`."
    )
    assert doc[6] == "Keywords : health,."
    assert doc[12] == "  Default : `ds.tasmin`. [Required units : [temperature]]"
    assert (
        doc[41]
        == "  Total number of series of at least {window} consecutive days with daily minimum temperature above "
        "{thresh_tasmin} and daily maximum temperature above {thresh_tasmax} (heat_wave_events), "
        "with additional attributes: **description**: {freq} number of heat wave events within a given period. "
        "A heat wave occurs when daily minimum and maximum temperatures exceed {thresh_tasmin} and {thresh_tasmax}, "
        "respectively, over at least {window} days."
    )

    doc = degree_days_exceedance_date.__doc__.split("\n")
    assert doc[20] == "  Default : >. "


def test_update_xclim_history(atmosds):
    @fmt.update_xclim_history
    def func(da, arg1, arg2=None, arg3=None):
        return da

    out = func(atmosds.tas, 1, arg2=[1, 2], arg3=None)

    matches = re.match(
        r"\[([0-9-:\s]*)\]\s(\w*):\s(\w*)\((.*)\)\s-\sxclim\sversion:\s(\d*\.\d*\.\d*[a-zA-Z-]*)",
        out.attrs["history"],
    ).groups()

    date = dt.datetime.fromisoformat(matches[0])
    assert dt.timedelta(0) < (dt.datetime.now() - date) < dt.timedelta(seconds=3)
    assert matches[1] == "tas"
    assert matches[2] == "func"
    assert matches[3] == "da=tas, arg1=1, arg2=[1, 2], arg3=None"
    assert matches[4] == __version__

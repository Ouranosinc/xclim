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
    assert doc[0] == "Heat wave frequency. (realm: atmos)"
    assert (
        doc[5]
        == "Based on indice :py:func:`xclim.indices._multivariate.heat_wave_frequency`."
    )
    assert doc[6] == "Keywords : health,."
    assert doc[12] == "  Default : `ds.tasmin`. [Required units : [temperature]]"
    assert (
        doc[35]
        == "  Number of heat wave events (Tmin > {thresh_tasmin} and Tmax > {thresh_tasmax} for >= {window} days) (heat_wave_events)"
    )

    doc = degree_days_exceedance_date.__doc__.split("\n")
    assert doc[20] == "  Default : >. "

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

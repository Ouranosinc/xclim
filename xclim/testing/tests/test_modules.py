import sys
from pathlib import Path

import pytest
import xarray as xr

from xclim import indicators
from xclim.core.indicator import build_indicator_module_from_yaml
from xclim.core.options import set_options
from xclim.core.utils import InputKind
from xclim.testing import open_dataset


def all_virtual_indicators():
    for mod in ["anuclim", "cf", "icclim"]:
        for name, ind in getattr(indicators, mod).iter_indicators():
            yield pytest.param((mod, name, ind), id=f"{mod}.{name}")


@pytest.fixture(params=all_virtual_indicators())
def virtual_indicator(request):
    return request.param


def test_default_modules_exist():
    from xclim.indicators import anuclim, cf, icclim  # noqa

    assert hasattr(icclim, "TG")

    assert hasattr(anuclim, "P1_AnnMeanTemp")
    assert hasattr(anuclim, "P19_PrecipColdestQuarter")

    assert hasattr(cf, "fg")

    assert len(list(icclim.iter_indicators())) == 55
    assert len(list(anuclim.iter_indicators())) == 19
    # Not testing cf because many indices are waiting to be implemented.


@pytest.mark.slow
def test_virtual_modules(virtual_indicator, atmosds):
    with set_options(cf_compliance="warn"):
        # skip when missing default values
        kws = {}
        mod, indname, ind = virtual_indicator
        for name, param in ind.parameters.items():
            if name == "src_timestep":
                kws["src_timestep"] = "D"
            if param["kind"] is not InputKind.DATASET and (
                param["default"] is None
                or (param["default"] == name and name not in atmosds)
            ):

                pytest.skip(f"Indicator {mod}.{indname} has no default for {name}.")
        ind(ds=atmosds)


@pytest.mark.requires_docs
def test_custom_indices():
    # Use the example in the Extending Xclim notebook for testing.
    pr = open_dataset("ERA5/daily_surface_cancities_1990-1993.nc").pr

    nbpath = Path(__file__).parent.parent.parent.parent / "docs" / "notebooks"
    sys.path.insert(1, str(nbpath.absolute()))

    import example  # noqa

    # Fron module
    ex1 = build_indicator_module_from_yaml(
        nbpath / "example.yml", name="ex1", indices=example
    )

    # From mapping
    exinds = {"extreme_precip_accumulation": example.extreme_precip_accumulation}
    ex2 = build_indicator_module_from_yaml(
        nbpath / "example.yml", name="ex2", indices=exinds
    )

    assert ex1.R95p.__doc__ == ex2.R95p.__doc__  # noqa

    out1 = ex1.R95p(pr=pr)  # noqa
    out2 = ex2.R95p(pr=pr)  # noqa

    xr.testing.assert_equal(out1, out2)

    # Check that missing was not modified even with injecting `freq`.
    assert ex1.RX5day.missing == indicators.atmos.max_n_day_precipitation_amount.missing


# It's not really slow, but this is an unstable test (when it fails) and we might not want to execute it on all builds
@pytest.mark.slow
def test_encoding():
    import sys

    import _locale

    # remove xclim
    del sys.modules["xclim"]

    # patch so that the default encoding is not UTF-8
    old = _locale.nl_langinfo
    _locale.nl_langinfo = lambda x: "GBK"

    try:
        import xclim  # noqa
    finally:
        # Put the correct function back
        _locale.nl_langinfo = old

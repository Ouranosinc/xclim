import sys
from pathlib import Path

import pytest
import xarray as xr

from xclim.core.indicator import build_indicator_module_from_yaml, registry
from xclim.core.options import set_options
from xclim.core.utils import InputKind
from xclim.testing import open_dataset


def test_default_modules_exist():
    from xclim.indicators import anuclim, cf, icclim

    assert hasattr(icclim, "TG")

    assert hasattr(anuclim, "P1_AnnMeanTemp")
    assert hasattr(anuclim, "P19_PrecipColdestQuarter")

    assert hasattr(cf, "fg")

    assert len(list(icclim.iter_indicators())) == 48
    assert len(list(anuclim.iter_indicators())) == 19
    # Not testing cf because many indices are waiting to be implemented.


@pytest.mark.slow
@pytest.mark.parametrize(
    "indname", [name for name in registry.keys() if name.startswith("cf.")]
)
def test_cf(indname, atmosds):
    with set_options(cf_compliance="warn"):
        # skip when missing default values
        ind = registry[indname].get_instance()
        for name, param in ind.parameters.items():
            if param["kind"] is not InputKind.DATASET and param["default"] in (
                None,
                name,
            ):
                pytest.skip(f"Indicator {ind.identifier} has no default for {name}.")
        ind(ds=atmosds)


@pytest.mark.slow
@pytest.mark.parametrize(
    "indname", [name for name in registry.keys() if name.startswith("icclim.")]
)
def test_icclim(indname, atmosds):
    # skip when missing default values
    ind = registry[indname].get_instance()
    for name, param in ind.parameters.items():
        if param["kind"] is not InputKind.DATASET and param["default"] in (None, name):
            pytest.skip(f"Indicator {ind.identifier} has no default for {name}.")
    ind(ds=atmosds)


@pytest.mark.slow
@pytest.mark.parametrize(
    "indname", [name for name in registry.keys() if name.startswith("anuclim.")]
)
def test_anuclim(indname, atmosds):
    # skip when missing default values
    ind = registry[indname].get_instance()
    kws = {}
    for name, param in ind.parameters.items():
        if name == "src_timestep":
            kws["src_timestep"] = "D"
        elif param["kind"] is not InputKind.DATASET and param["default"] in (
            None,
            name,
        ):
            pytest.skip(f"Indicator {ind.identifier} has no default for {name}.")
    ind(ds=atmosds, **kws)


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

    assert ex1.R95p.__doc__ == ex2.R95p.__doc__

    out1 = ex1.R95p(pr=pr)
    out2 = ex2.R95p(pr=pr)

    xr.testing.assert_equal(out1, out2)

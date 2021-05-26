import pytest

from xclim.core.indicator import registry
from xclim.core.options import set_options
from xclim.core.utils import InputKind


def test_default_modules_exist():
    from xclim.indicators import anuclim, cf, icclim

    assert hasattr(icclim, "TG")

    assert hasattr(anuclim, "P1_AnnMeanTemp")
    assert hasattr(anuclim, "P19_PrecipColdestQuarter")

    assert hasattr(cf, "fg")

    assert len(list(icclim.iter_indicators())) == 47
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
            if param["kind"] is not InputKind.DATASET and param["default"] is None:
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
        if param["kind"] is not InputKind.DATASET and param["default"] is None:
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
        elif param["kind"] is not InputKind.DATASET and param["default"] is None:
            pytest.skip(f"Indicator {ind.identifier} has no default for {name}.")
    ind(ds=atmosds, **kws)

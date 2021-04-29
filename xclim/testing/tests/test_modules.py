import pytest

from xclim.core.indicator import registry
from xclim.core.options import set_options
from xclim.core.utils import InputKind


def test_default_modules_exist():
    from xclim.indicators import anuclim, cf, icclim

    assert getattr(icclim, "TG", None) is not None

    assert getattr(anuclim, "P1_AnnMeanTemp", None) is not None
    assert getattr(anuclim, "P19_PrecipColdestQuarter", None) is not None

    assert getattr(cf, "fg", None) is not None


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

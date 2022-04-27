from inspect import _empty
from pathlib import Path

import pytest
import xarray as xr
import yamale

from xclim import indicators
from xclim.core.indicator import build_indicator_module_from_yaml
from xclim.core.locales import read_locale_file
from xclim.core.options import set_options
from xclim.core.utils import InputKind, load_module
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
        mod, indname, ind = virtual_indicator
        for name, param in ind.parameters.items():
            if param.kind is not InputKind.DATASET and (
                param.default in (None, _empty)
                or (param.default == name and name not in atmosds)
            ):

                pytest.skip(f"Indicator {mod}.{indname} has no default for {name}.")
        ind(ds=atmosds)


@pytest.mark.requires_docs
def test_custom_indices():
    # Use the example in the Extending Xclim notebook for testing.
    nbpath = Path(__file__).parent.parent.parent.parent / "docs" / "notebooks"

    schema = yamale.make_schema(
        Path(__file__).parent.parent.parent / "data" / "schema.yml"
    )
    data = yamale.make_data(nbpath / "example.yml")
    yamale.validate(schema, data)

    pr = open_dataset("ERA5/daily_surface_cancities_1990-1993.nc").pr

    # This tests load_module with a python file that is _not_ on the PATH
    example = load_module(nbpath / "example.py")

    # Fron module
    ex1 = build_indicator_module_from_yaml(
        nbpath / "example.yml", name="ex1", indices=example
    )

    # From mapping
    exinds = {
        "extreme_precip_accumulation_and_days": example.extreme_precip_accumulation_and_days
    }
    ex2 = build_indicator_module_from_yaml(
        nbpath / "example.yml", name="ex2", indices=exinds
    )

    assert ex1.R95p.__doc__ == ex2.R95p.__doc__  # noqa

    out1 = ex1.R95p(pr=pr)  # noqa
    out2 = ex2.R95p(pr=pr)  # noqa

    xr.testing.assert_equal(out1[0], out2[0])

    # Check that missing was not modified even with injecting `freq`.
    assert ex1.RX5day.missing == indicators.atmos.max_n_day_precipitation_amount.missing

    # Error when missing
    with pytest.raises(ImportError, match="extreme_precip_accumulation_and_days"):
        build_indicator_module_from_yaml(nbpath / "example.yml", name="ex3")
    build_indicator_module_from_yaml(nbpath / "example.yml", name="ex4", mode="ignore")


@pytest.mark.requires_docs
def test_indicator_module_translations():
    # Use the example in the Extending Xclim notebook for testing.
    nbpath = Path(__file__).parent.parent.parent.parent / "docs" / "notebooks"

    ex = build_indicator_module_from_yaml(nbpath / "example", name="ex_trans")
    assert ex.RX5day.translate_attrs("fr")["cf_attrs"][0]["long_name"].startswith(
        "Cumul maximal"
    )
    assert indicators.atmos.max_n_day_precipitation_amount.translate_attrs("fr")[
        "cf_attrs"
    ][0]["long_name"].startswith("Maximum du cumul")


@pytest.mark.requires_docs
def test_build_indicator_module_from_yaml_edge_cases():
    # Use the example in the Extending Xclim notebook for testing.
    nbpath = Path(__file__).parent.parent.parent.parent / "docs" / "notebooks"

    # All from paths but one
    ex5 = build_indicator_module_from_yaml(
        nbpath / "example.yml",
        indices=nbpath / "example.py",
        translations={
            "fr": nbpath / "example.fr.json",
            "ru": str(nbpath / "example.fr.json"),
            "eo": read_locale_file(nbpath / "example.fr.json", module="ex5"),
        },
        name="ex5",
    )
    assert hasattr(indicators, "ex5")
    assert ex5.R95p.translate_attrs("fr")["cf_attrs"][0]["description"].startswith(
        "Épaisseur équivalente"
    )
    assert ex5.R95p.translate_attrs("ru")["cf_attrs"][0]["description"].startswith(
        "Épaisseur équivalente"
    )
    assert ex5.R95p.translate_attrs("eo")["cf_attrs"][0]["description"].startswith(
        "Épaisseur équivalente"
    )


class TestOfficalYaml(yamale.YamaleTestCase):
    base_dir = str(Path(__file__).parent.parent.parent / "data")
    schema = "schema.yml"
    yaml = ["cf.yml", "anuclim.yml", "icclim.yml"]

    def test_all(self):
        assert self.validate()


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

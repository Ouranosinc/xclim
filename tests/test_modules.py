from __future__ import annotations

import platform
from importlib.util import find_spec
from inspect import _empty  # noqa
from pathlib import Path

import pytest
import xarray as xr
import yamale
from yaml import safe_load

from xclim import indicators
from xclim.core import VARIABLES
from xclim.core.indicator import build_indicator_module_from_yaml
from xclim.core.locales import read_locale_file
from xclim.core.options import set_options
from xclim.core.utils import InputKind, adapt_clix_meta_yaml, load_module


def all_virtual_indicators():
    for mod in ["anuclim", "cf", "icclim"]:
        for name, ind in getattr(indicators, mod).iter_indicators():
            yield pytest.param((mod, name, ind), id=f"{mod}.{name}")


@pytest.fixture(params=all_virtual_indicators())
def virtual_indicator(request):
    return request.param


def test_default_modules_exist():
    from xclim.indicators import anuclim  # noqa
    from xclim.indicators import cf  # noqa
    from xclim.indicators import icclim  # noqa

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
            if param.kind not in [InputKind.DATASET, InputKind.KWARGS] and (
                param.default in (None, _empty) or (param.default == name and name not in atmosds)
            ):
                pytest.skip(f"Indicator {mod}.{indname} has no default for {name}.")
        ind(ds=atmosds)


@pytest.mark.requires_docs
def test_custom_indices(open_dataset):
    # Use the example data used in the Extending Xclim notebook for testing.
    example_path = Path(__file__).parent.parent / "docs" / "notebooks" / "example"

    pr = open_dataset("ERA5/daily_surface_cancities_1990-1993.nc").pr

    # This tests load_module with a python file that is _not_ on the PATH
    example = load_module(example_path / "example.py")

    # From module
    ex1 = build_indicator_module_from_yaml(example_path / "example.yml", name="ex1", indices=example)

    # Did this register the new variable?
    assert "prveg" in VARIABLES

    # From mapping
    extreme_inds = {"extreme_precip_accumulation_and_days": example.extreme_precip_accumulation_and_days}
    ex2 = build_indicator_module_from_yaml(example_path / "example.yml", name="ex2", indices=extreme_inds)

    assert ex1.R95p.__doc__ == ex2.R95p.__doc__  # noqa

    out1 = ex1.R95p(pr=pr)  # noqa
    out2 = ex2.R95p(pr=pr)  # noqa

    xr.testing.assert_equal(out1[0], out2[0])

    # Check that missing was not modified even with injecting `freq`.
    assert ex1.RX5day_canopy.missing == indicators.atmos.max_n_day_precipitation_amount.missing

    # Error when missing
    with pytest.raises(ImportError, match="extreme_precip_accumulation_and_days"):
        build_indicator_module_from_yaml(example_path / "example.yml", name="ex3")
    build_indicator_module_from_yaml(example_path / "example.yml", name="ex4", mode="ignore")

    # Check that indexer was added and injected correctly
    assert "indexer" not in ex1.RX1day_summer.parameters
    assert ex1.RX1day_summer.injected_parameters["indexer"] == {"month": [5, 6, 7, 8, 9]}


@pytest.mark.requires_docs
def test_indicator_module_translations():
    # Use the example data used in the Extending Xclim notebook for testing.
    example_path = Path(__file__).parent.parent / "docs" / "notebooks" / "example"

    ex = build_indicator_module_from_yaml(example_path / "example", name="ex_trans")
    assert ex.RX5day_canopy.translate_attrs("fr")["cf_attrs"][0]["long_name"].startswith("Cumul maximal")
    assert indicators.atmos.max_n_day_precipitation_amount.translate_attrs("fr")["cf_attrs"][0]["long_name"].startswith(
        "Maximum du cumul"
    )


@pytest.mark.requires_docs
def test_indicator_module_input_mapping(atmosds):
    example_path = Path(__file__).parent.parent / "docs" / "notebooks" / "example"
    ex = build_indicator_module_from_yaml(example_path / "example", name="ex_input")
    prveg = atmosds.pr.rename("prveg").assign_attrs(standard_name="precipitation_flux_onto_canopy")

    out = ex.RX5day_canopy(prveg=prveg)
    assert "RX5DAY_CANOPY(prveg=prveg)" in out.attrs["history"]


@pytest.mark.requires_docs
def test_build_indicator_module_from_yaml_edge_cases():
    # Use the example data used in the Extending Xclim notebook for testing.
    example_path = Path(__file__).parent.parent / "docs" / "notebooks" / "example"

    # All from paths but one
    ex5 = build_indicator_module_from_yaml(
        example_path / "example.yml",
        indices=example_path / "example.py",
        translations={
            "fr": example_path / "example.fr.json",
            "ru": str(example_path / "example.fr.json"),
            "eo": read_locale_file(example_path / "example.fr.json", module="ex5"),
        },
        name="ex5",
    )
    assert hasattr(indicators, "ex5")
    assert ex5.R95p.translate_attrs("fr")["cf_attrs"][0]["description"].startswith("Épaisseur équivalente")
    assert ex5.R95p.translate_attrs("ru")["cf_attrs"][0]["description"].startswith("Épaisseur équivalente")
    assert ex5.R95p.translate_attrs("eo")["cf_attrs"][0]["description"].startswith("Épaisseur équivalente")


class TestClixMeta:
    cdd = """
indices:
  cdd:
    reference: ETCCDI
    default_period: annual
    output:
      var_name: "cdd"
      standard_name: spell_length_of_days_with_lwe_thickness_of_precipitation_amount_below_threshold
      proposed_standard_name: spell_length_with_lwe_thickness_of_precipitation_amount_below_threshold
      long_name: "Maximum consecutive dry days (Precip < 1mm)"
      units: "day"
      cell_methods:
        - time: sum within days
        - time: sum over days
    input:
      data: pr
    index_function:
      name: spell_length
      parameters:
        threshold:
          kind: quantity
          standard_name: lwe_precipitation_rate
          long_name: "Wet day threshold"
          data: 1
          units: "mm day-1"
        condition:
          kind: operator
          operator: "<"
        reducer:
          kind: reducer
          reducer: max
    ET:
      short_name: "cdd"
      long_name: "Consecutive dry days"
      definition: "Maximum number of consecutive days with P<1mm"
      comment: "maximum consecutive days when daily total precipitation is below 1 mm"
"""

    def test_simple_clix_meta_adaptor(self, tmp_path):
        test_yaml = tmp_path.joinpath("test.yaml")

        adapt_clix_meta_yaml(self.cdd, test_yaml)

        converted = safe_load(Path(test_yaml).open())
        assert "cdd" in converted["indicators"]


def test_realm(tmp_path):
    # Regression test for #1425
    yml = """
    realm: land

    indicators:
      ice_extent:
        base: sea_ice_extent
        realm: ocean
    """
    fh = tmp_path / "test.yml"

    fh.write_text(yml)
    mod = build_indicator_module_from_yaml(fh, name="test")
    assert mod.ice_extent.realm == "ocean"


def test_validate(tmp_path):
    # Regression test for #1425
    yml = """
    realm: land

    indicators:
      ice_extent:
        base: sea_ice_extent
        realm: ocean
        this_is_not_accepted: True
    """
    fh = tmp_path / "test.yml"
    fh.write_text(yml)

    with pytest.raises(yamale.YamaleError):
        build_indicator_module_from_yaml(fh, name="test")

    build_indicator_module_from_yaml(fh, name="test2", validate=False)

    sch = r"""
realm: str(required=False)
indicators: map(include('indicator'), key=regex(r'^[-\w]+$'))
---
indicator:
  base: str(required=False)
  realm: str(required=False)
  this_is_not_accepted: bool()
"""
    fsch = tmp_path / "schema.yml"
    fsch.write_text(sch)
    build_indicator_module_from_yaml(fh, name="test3", validate=fsch)


class TestOfficialYaml(yamale.YamaleTestCase):
    base_dir = str(Path(find_spec("xclim").origin).parent.joinpath("data"))
    schema = "schema.yml"
    yaml = ["cf.yml", "anuclim.yml", "icclim.yml"]

    def test_all(self):
        assert self.validate()


@pytest.mark.xfail(reason="This test is relatively unstable.", strict=False)
@pytest.mark.skipif(platform.system() == "Windows", reason="nl_langinfo not available on Windows.")
def test_encoding():
    import _locale
    import sys

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

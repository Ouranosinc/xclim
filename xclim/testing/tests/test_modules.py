import pytest

from xclim.core.indicator import (
    build_indicator_module,
    build_indicator_module_from_yaml,
)


def test_default_modules_exist():
    from xclim.indicators import anuclim, cf, icclim

    assert getattr(icclim, "TG", None) is not None

    assert getattr(anuclim, "P1_AnnMeanTemp", None) is not None
    assert getattr(anuclim, "P19_PrecipColdestQuarter", None) is not None

    assert getattr(cf, "fd", None) is not None

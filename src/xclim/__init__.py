"""Climate indices computation package based on Xarray."""

from __future__ import annotations

import importlib.resources as _resources

from xclim import compute, ensembles
from xclim import indicators as _indicators
from xclim.core import calendar, units
from xclim.core.collection import IndicatorCollection
from xclim.core.locales import load_locale as _load_locale
from xclim.core.options import set_options
from xclim.indicators import atmos, convert, generic, land, seaIce

__author__ = """Travis Logan"""
__email__ = "logan.travis@ouranos.ca"
__version__ = "0.99.0-dev.24"

with _resources.as_file(_resources.files("xclim.data")) as _module_data:
    # Load official locales
    for _filename in _module_data.glob("??.json"):
        # Only select <locale>.json and not <module>.<locale>.json
        _load_locale(_filename, _filename.stem)

    # Virtual modules creation:
    _indicators.icclim = IndicatorCollection.from_yaml(_module_data / "icclim", mode="raise", register=True)
    _indicators.anuclim = IndicatorCollection.from_yaml(_module_data / "anuclim", mode="raise", register=True)
    _indicators.cf = IndicatorCollection.from_yaml(_module_data / "cf", mode="raise", register=True)

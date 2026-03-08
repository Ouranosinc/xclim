"""Climate indices computation package based on Xarray."""

from __future__ import annotations

import importlib.resources as _resources

from xclim import compute, ensembles
from xclim.core import calendar, units  # noqa
from xclim.core.indicator import build_indicator_module_from_yaml
from xclim.core.locales import load_locale as _load_locale
from xclim.core.options import set_options  # noqa
from xclim.indicators import atmos, convert, generic, land, seaIce  # noqa

__author__ = """Travis Logan"""
__email__ = "logan.travis@ouranos.ca"
__version__ = "0.99.0-dev.4"


with _resources.as_file(_resources.files("xclim.data")) as _module_data:
    # Load official locales
    for _filename in _module_data.glob("??.json"):
        # Only select <locale>.json and not <module>.<locale>.json
        _load_locale(_filename, _filename.stem)

    # Virtual modules creation:
    build_indicator_module_from_yaml(_module_data / "icclim", mode="raise")
    build_indicator_module_from_yaml(_module_data / "anuclim", mode="raise")
    build_indicator_module_from_yaml(_module_data / "cf", mode="raise")

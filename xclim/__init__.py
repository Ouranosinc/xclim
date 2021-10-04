# -*- coding: utf-8 -*-
"""Climate indices computation package based on Xarray."""
from importlib.resources import contents, path

from xclim.core import units  # noqa
from xclim.core.indicator import build_indicator_module_from_yaml
from xclim.core.locales import load_locale
from xclim.core.options import set_options  # noqa
from xclim.indicators import atmos, land, seaIce  # noqa

__author__ = """Travis Logan"""
__email__ = "logan.travis@ouranos.ca"
__version__ = "0.30.1"


# Load official locales
for filename in contents("xclim.data"):
    # Only select <locale>.json and not <module>.<locale>.json
    if filename.endswith(".json") and filename.count(".") == 1:
        locale = filename.split(".")[0]
        with path("xclim.data", filename) as f:
            load_locale(f, locale)


# Virtual modules creation:
with path("xclim.data", "icclim.yml") as f:
    build_indicator_module_from_yaml(f.with_suffix(""), mode="raise")
with path("xclim.data", "anuclim.yml") as f:
    build_indicator_module_from_yaml(f.with_suffix(""), mode="raise")
with path("xclim.data", "cf.yml") as f:
    # ignore because some generic function are missing.
    build_indicator_module_from_yaml(f.with_suffix(""), mode="ignore")

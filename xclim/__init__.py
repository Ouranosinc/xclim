# -*- coding: utf-8 -*-
"""Climate indices computation package based on Xarray."""
from importlib.resources import path

from xclim.core import units  # noqa
from xclim.core.indicator import build_indicator_module_from_yaml
from xclim.core.options import set_options  # noqa
from xclim.indicators import atmos, land, seaIce  # noqa

__author__ = """Travis Logan"""
__email__ = "logan.travis@ouranos.ca"
__version__ = "0.26.2-beta"


# Virtual modules creation:
with path("xclim.data", "icclim.yml") as f:
    build_indicator_module_from_yaml(f, mode="raise")
with path("xclim.data", "anuclim.yml") as f:
    build_indicator_module_from_yaml(f, mode="raise")
with path("xclim.data", "cf.yml") as f:
    # ignore because some generic function are missing.
    build_indicator_module_from_yaml(f, mode="ignore")

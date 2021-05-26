# -*- coding: utf-8 -*-
"""Climate indices computation package based on Xarray."""
from importlib.resources import path

from boltons.funcutils import wraps

from xclim.core import units  # noqa
from xclim.core.indicator import build_indicator_module_from_yaml
from xclim.core.options import set_options  # noqa
from xclim.indicators import atmos, land, seaIce  # noqa

__author__ = """Travis Logan"""
__email__ = "logan.travis@ouranos.ca"
__version__ = "0.26.4-beta"


# Virtual modules creation:
with path("xclim.data", "icclim.yml") as f:
    build_indicator_module_from_yaml(f, mode="raise")
with path("xclim.data", "anuclim.yml") as f:
    build_indicator_module_from_yaml(f, mode="raise")
with path("xclim.data", "cf.yml") as f:
    # ignore because some generic function are missing.
    build_indicator_module_from_yaml(f, mode="ignore")

try:
    import metpy.calc
except ImportError:
    pass
else:
    # MetPy returns DataArrays with pint arrays, instead of numpy. This function and fake mapping wrap indices on-the-fly.

    def metpy_calc_wrapper(func):
        """Given a function from metpy's "calc" submodule wraps it to convert the pint output to xclim-compliant output."""

        # MetPy makes extensive use of wrapper (like us), but doesn't use boltons.funcutils, thus the signature is not propagated. This causes problems in Indicator.from_dict.
        @wraps(func.__wrapped__.__wrapped__)
        def _metpy_wrap(*args, **kwargs):
            out = func(*args, **kwargs)
            outxc = out.copy(data=out.data.magnitude)
            outxc.attrs["units"] = units.pint2cfunits(out.data.units)
            return outxc

        return _metpy_wrap

    class MetPyWrapper(object):
        """Dict-like object wrapping MetPy indices on-the-fly."""

        def __getitem__(self, indice):
            return metpy_calc_wrapper(getattr(metpy.calc, indice))

    fake_metpy = MetPyWrapper()

    with path("xclim.data", "metpy.yml") as f:
        build_indicator_module_from_yaml(f, indices=fake_metpy, mode="raise")

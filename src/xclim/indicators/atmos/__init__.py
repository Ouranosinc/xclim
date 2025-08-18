"""
Atmospheric Indicators
======================

While the `indices` module stores the computing functions, this module defines Indicator classes and instances that
include a number of functionalities, such as input validation, unit conversion, output meta-data handling,
and missing value masking.

The concept followed here is to define Indicator subclasses for each input variable, then create instances
for each indicator.
"""

from __future__ import annotations

import functools
import warnings

from xclim.indicators import convert
from xclim.indicators.convert._conversion import __all__ as _conversion_all  # noqa: F401

from ._precip import *
from ._precip import __all__ as _precip_all
from ._synoptic import *
from ._synoptic import __all__ as _synoptic_all
from ._temperature import *
from ._temperature import __all__ as _temperature_all
from ._wind import *
from ._wind import __all__ as _wind_all

__all__ = _precip_all + _synoptic_all + _temperature_all + _wind_all


def _deprecated_alias(func_name):
    """Factory to create a deprecated alias for a function in new_module."""
    new_func = getattr(convert, func_name)

    @functools.wraps(new_func)
    def wrapper(*args, **kwargs):
        warnings.warn(
            f"{func_name} is deprecated and will be removed in a future release. "
            f"Use xclim.convert.{func_name} instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return new_func(*args, **kwargs)

    return wrapper


for _name in _conversion_all:
    # Exclude snd_to_snw and snw_to_snd as they were never in atmos module
    if _name not in ["snd_to_snw", "snw_to_snd"]:
        continue
    _obj = getattr(convert, _name)
    if callable(_obj):
        globals()[_name] = _deprecated_alias(_name)

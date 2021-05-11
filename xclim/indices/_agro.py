# noqa: D100
from typing import Optional

import numpy as np
import xarray

from xclim.core.calendar import resample_doy
from xclim.core.units import (
    convert_units_to,
    declare_units,
    pint2cfunits,
    rate2amount,
    str2pint,
    to_agg_units,
)

from . import run_length as rl
from ._conversion import rain_approximation, snowfall_approximation
from .generic import select_resample_op, threshold_count

# Frequencies : YS: year start, QS-DEC: seasons starting in december, MS: month start
# See http://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases

# -------------------------------------------------- #
# ATTENTION: ASSUME ALL INDICES WRONG UNTIL TESTED ! #
# -------------------------------------------------- #

__all__ = [
    "corn_heat_units"
]

@declare_units(tasmin="[temperature]", tasmax="[temperature]", thresh_tasmin="[temperature]", thresh_tasmax="[temperature]")
def corn_heat_units(tasmin: xarray.DataArray, tasmax: xarray.DataArray, thresh_tasmin: str, thresh_tasmax: str
) -> xarray.DataArray:

    thresh_tasmax = convert_units_to(thresh_tasmax, tasmax)
    thresh_tasmin = convert_units_to(thresh_tasmin, tasmin)

    mask_tasmax = tasmax > thresh_tasmax

    chu = mask_tasmax
    return chu

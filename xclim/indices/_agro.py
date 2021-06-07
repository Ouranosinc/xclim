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
from ._threshold import first_day_above, first_day_below
from .generic import aggregate_between_dates

# Frequencies : YS: year start, QS-DEC: seasons starting in december, MS: month start
# See http://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases

# -------------------------------------------------- #
# ATTENTION: ASSUME ALL INDICES WRONG UNTIL TESTED ! #
# -------------------------------------------------- #

__all__ = [
  "corn_heat_units",
  "biologically_effective_degree_days"
]


@declare_units(
    tasmin="[temperature]",
    tasmax="[temperature]",
    thresh_tasmin="[temperature]",
    thresh_tasmax="[temperature]",
)
def corn_heat_units(
    tasmin: xarray.DataArray,
    tasmax: xarray.DataArray,
    thresh_tasmin: str = "4.44 degC",
    thresh_tasmax: str = "10 degC",
) -> xarray.DataArray:
    r"""Corn heat units.

    Temperature-based index used to estimate the development of corn crops.

    Parameters
    ----------
    tasmin : xarray.DataArray
      Minimum daily temperature.
    tasmax : xarray.DataArray
      Maximum daily temperature.
    thresh_tasmin : str
      The minimum temperature threshold needed for corn growth.
    thresh_tasmax : str
      The maximum temperature threshold needed for corn growth.

    Returns
    -------
    xarray.DataArray, [dimensionless]
      Daily corn heat units.

    Notes
    -----
    The thresholds of 4.44°C for minimum temperatures and 10°C for maximum temperatures were selected following
    the assumption that no growth occurs below these values.

    Let :math:`TX_{i}` and :math:`TN_{i}` be the daily maximum and minimum temperature at day :math:`i`. Then the daily
    corn heat unit is:

    .. math::
        CHU_i = \frac{YX_{i} + YN_{i}}{2}

    with

    .. math::

        YX_i & = 3.33(TX_i -10) - 0.084(TX_i -10)^2, &\text{if } TX_i > 10°C

        YN_i & = 1.8(TN_i -4.44), &\text{if } TN_i > 4.44°C

    where :math:`YX_{i}` and :math:`YN_{i}` is 0 when :math:`TX_i \leq 10°C` and :math:`TN_i \leq 4.44°C`, respectively.

    References
    ----------
    Equations from Bootsma, A., G. Tremblay et P. Filion. 1999: Analyse sur les risques associés aux unités thermiques
    disponibles pour la production de maïs et de soya au Québec. Centre de recherches de l’Est sur les céréales et
    oléagineux, Ottawa, 28 p.

    Can be found in Audet, R., Côté, H., Bachand, D. and Mailhot, A., 2012: Atlas agroclimatique du Québec. Évaluation
    des opportunités et des risques agroclimatiques dans un climat en évolution.
    """

    tasmin = convert_units_to(tasmin, "degC")
    tasmax = convert_units_to(tasmax, "degC")
    thresh_tasmin = convert_units_to(thresh_tasmin, "degC")
    thresh_tasmax = convert_units_to(thresh_tasmax, "degC")

    mask_tasmin = tasmin > thresh_tasmin
    mask_tasmax = tasmax > thresh_tasmax

    chu = (
        xarray.where(mask_tasmin, 1.8 * (tasmin - thresh_tasmin), 0)
        + xarray.where(
            mask_tasmax,
            (3.33 * (tasmax - thresh_tasmax) - 0.084 * (tasmax - thresh_tasmax) ** 2),
            0,
        )
    ) / 2

    chu.attrs["units"] = ""
    return chu


@declare_units(
    tasmin="[temperature]",
    tasmax="[temperature]",
    thresh_tasmin="[temperature]",
    thresh_tasmax="[temperature]",
)
def biologically_effective_degree_days(
    tasmin: xarray.DataArray,
    tasmax: xarray.DataArray,
    thresh_tasmin: str = "10 degC",
    thresh_tasmax: str = "19 degC",
    lat_dim: str = "lat",
    hemisphere: str = "north",
) -> xarray.DataArray:
    """

    Parameters
    ----------
    tasmin: str
      Minimum daily temperature.
    tasmax: str
      Maximum daily temperature.
    thresh_tasmin: str
      The minimum temperature threshold.
    thresh_tasmax: str
      The maximum temperature threshold.
    lat_dim: str
    hemisphere: {"north", "south"}
      The hemisphere-based growing season to consider (north = April - October, south = October - April).

    Returns
    -------
    xarray.DataArray
    """
    tasmin = convert_units_to(tasmin, "degC")
    tasmax = convert_units_to(tasmax, "degC")
    thresh_tasmin = convert_units_to(thresh_tasmin, "degC")
    thresh_tasmax = convert_units_to(thresh_tasmax, "degC")

    mask_tasmin = tasmin > thresh_tasmin

    lat_mask = (abs(tasmin[lat_dim]) >= 40) & (abs(tasmin[lat_dim]) <= 50)
    lat_constant = xarray.where(lat_mask, (abs(tasmin[lat_dim]) / 50) * 0.06, 0)

    def sel_months(time):
        if hemisphere.lower() == "north":
            return (time >= 4) & (time <= 7)
        elif hemisphere.lower() == "south":
            raise NotImplementedError()
            # This needs to cross the year-line for consistent growing seasons
            # return (time>=10) & (time <=4)

    def tasmax_limited(tasmax, thresh_tasmax):
        return xarray.where((tasmax > thresh_tasmax).any(), thresh_tasmax, tasmax)

    date_mask = tasmin.sel(time=sel_months(tasmin["time.month"]))

    bedd = (
        (
            (
                xarray.where(mask_tasmin, (tasmin - thresh_tasmin), 0)
                + xarray.where(
                    mask_tasmin,
                    tasmax_limited(tasmax, thresh_tasmax) - thresh_tasmin,
                    0,
                )
            )
            / 2
        )
        * (1 + lat_constant)
        * date_mask
    )

    bedd.attrs["units"] = "degC"
    return bedd

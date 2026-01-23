"""
Classify Submodule
==================

This submodule defines tools to perform classification, also know in the GIS world as a « multi-criteria analysis ».
The process is understood here as an operation that takes spatially varying variables and assigns
a "class" or "category" to each spatial element (usually gridpoints). The classification process usually doesn't
reduce any dimension.
"""

from __future__ import annotations

import warnings

import numpy as np
import xarray as xr
from pint import Quantity

from xclim.core.units import convert_units_to, pint2cfunits, str2pint


def _get_zone_bins(
    zone_min: Quantity,
    zone_max: Quantity,
    zone_step: Quantity,
):
    """
    Bin boundary values as defined by zone parameters.

    Parameters
    ----------
    zone_min : Quantity
        Left boundary of the first zone.
    zone_max : Quantity
        Right boundary of the last zone.
    zone_step: Quantity
        Size of zones.

    Returns
    -------
    xr.DataArray, [units of `zone_step`]
        Array of values corresponding to each zone: [zone_min, zone_min+step, ..., zone_max].
    """
    units = pint2cfunits(str2pint(zone_step))
    mn, mx, step = (convert_units_to(str2pint(z), units) for z in [zone_min, zone_max, zone_step])
    bins = np.arange(mn, mx + step, step)
    if (mx - mn) % step != 0:
        warnings.warn("`zone_max` - `zone_min` is not an integer multiple of `zone_step`. Last zone will be smaller.")
        bins[-1] = mx
    return xr.DataArray(bins, attrs={"units": units})


def get_zones(
    da: xr.DataArray,
    zone_min: Quantity | None = None,
    zone_max: Quantity | None = None,
    zone_step: Quantity | None = None,
    bins: xr.DataArray | list[Quantity] | None = None,
    exclude_boundary_zones: bool = True,
    close_last_zone_right_boundary: bool = True,
) -> xr.DataArray:
    r"""
    Divide data into zones and attribute a zone coordinate to each input value.

    Divide values into zones corresponding to bins of width zone_step beginning at zone_min and ending at zone_max.
    Bins are inclusive on the left values and exclusive on the right values.

    Parameters
    ----------
    da : xr.DataArray
        Input data.
    zone_min : Quantity, optional
        Left boundary of the first zone.
    zone_max : Quantity, optional
        Right boundary of the last zone.
    zone_step : Quantity, optional
        Size of zones.
    bins : xr.DataArray or list of Quantity, optional
        Zones to be used, either as a DataArray with appropriate units or a list of Quantity.
    exclude_boundary_zones : bool
        Determines whether a zone value is attributed for values in ]`-np.inf`,
        `zone_min`[ and [`zone_max`, `np.inf`\ [.
    close_last_zone_right_boundary : bool
        Determines if the right boundary of the last zone is closed.

    Returns
    -------
    xr.DataArray, [dimensionless]
        Zone index for each value in `da`. Zones are returned as an integer range, starting from `0`.
    """
    # Check compatibility of arguments
    zone_params = np.array([zone_min, zone_max, zone_step])
    if bins is None:
        if (zone_params == [None] * len(zone_params)).any():
            raise ValueError(
                "`bins` is `None` as well as some or all of [`zone_min`, `zone_max`, `zone_step`]. "
                "Expected defined parameters in one of these cases."
            )
    elif set(zone_params) != {None}:
        warnings.warn("Expected either `bins` or [`zone_min`, `zone_max`, `zone_step`], got both. `bins` will be used.")

    # Get zone bins (if necessary)
    bins = bins if bins is not None else _get_zone_bins(zone_min, zone_max, zone_step)
    if isinstance(bins, list):
        bins = sorted([convert_units_to(b, da) for b in bins])
    else:
        bins = convert_units_to(bins, da)

    def _get_zone(_da):
        return np.digitize(_da, bins) - 1

    zones = xr.apply_ufunc(_get_zone, da, dask="parallelized")

    if close_last_zone_right_boundary:
        zones = zones.where(da != bins[-1], _get_zone(bins[-2]))
    if exclude_boundary_zones:
        zones = zones.where((zones != _get_zone(bins[0] - 1)) & (zones != _get_zone(bins[-1])))

    return zones

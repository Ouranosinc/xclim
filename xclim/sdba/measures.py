"""
Measures submodule
==================
SDBA diagnostic tests are made up of properties and measures. Measures compare adjusted simulations to a reference,
through statistical properties or directly.
This framework for the diagnostic tests was inspired by the [VALUE]_ project.

 .. [VALUE] http://www.value-cost.eu/
"""
from typing import Callable
from warnings import warn

import numpy as np
import xarray as xr
from boltons.funcutils import wraps

from xclim import sdba
from xclim.core.formatting import update_xclim_history
from xclim.core.units import convert_units_to, units2pint


def check_same_units_and_convert(func) -> Callable:
    """Verify that the simulation and the reference have the same units.
    If not, it converts the simulation to the units of the reference"""

    @wraps(
        func
    )  # in order to keep the docstring of the function where the decorator is applied
    def _check_same_units(*args):
        sim = args[0]
        ref = args[1]
        units_sim = units2pint(sim.units)
        units_ref = units2pint(ref.units)

        if units_sim != units_ref:
            warn(
                f" sim({units_sim}) and ref({units_ref}) don't have the same units."
                f" sim will be converted to {units_ref}."
            )
            sim = convert_units_to(sim, ref)
        out = func(sim, ref, *args[2:])
        return out

    return _check_same_units


@check_same_units_and_convert
@update_xclim_history
def bias(sim: xr.DataArray, ref: xr.DataArray) -> xr.DataArray:
    """Bias.

    The bias is the simulation minus the reference.

    Parameters
    ----------
    sim : xr.DataArray
      data from the simulation (one value for each grid-point)
    ref : xr.DataArray
      data from the reference (observations) (one value for each grid-point)

    Returns
    -------
    xr.DataArray,
      Bias between the simulation and the reference

    """
    out = sim - ref
    out.attrs.update(sim.attrs)
    out.attrs["long_name"] = "Bias"
    out.attrs["units"] = sim.attrs["units"]
    return out


@check_same_units_and_convert
@update_xclim_history
def relative_bias(sim: xr.DataArray, ref: xr.DataArray) -> xr.DataArray:
    """Relative Bias.

    The relative bias is the simulation minus reference, divided by the reference.

    Parameters
    ----------
    sim : xr.DataArray
      data from the simulation (one value for each grid-point)
    ref : xr.DataArray
      data from the reference (observations) (one value for each grid-point)

    Returns
    -------
    xr.DataArray,
      Relative bias between the simulation and the reference

    """
    out = (sim - ref) / ref
    out.attrs.update(sim.attrs)
    out.attrs["long_name"] = "Relative bias"
    out.attrs["units"] = ""
    return out


@check_same_units_and_convert
@update_xclim_history
def circular_bias(sim: xr.DataArray, ref: xr.DataArray) -> xr.DataArray:
    """Circular bias.

    Bias considering circular time series.
    E.g. The bias between doy 365 and doy 1 is 364, but the circular bias is -1.

    Parameters
    ----------
    sim : xr.DataArray
      data from the simulation (one value for each grid-point)
    ref : xr.DataArray
      data from the reference (observations) (one value for each grid-point)

    Returns
    -------
    xr.DataArray,
      Circular bias between the simulation and the reference

    """
    out = (sim - ref) % 365
    out = out.where(
        out <= 365 / 2, 365 - out
    )  # when condition false, replace by 2nd arg
    out = out.where(ref >= sim, out * -1)  # when condition false, replace by 2nd arg
    out.attrs.update(sim.attrs)
    out.attrs["long_name"] = "Circular bias"
    return out


@check_same_units_and_convert
@update_xclim_history
def ratio(sim: xr.DataArray, ref: xr.DataArray) -> xr.DataArray:
    """Ratio.

    The ratio is the quotient of the simulation over the reference.

    Parameters
    ----------
    sim : xr.DataArray
      data from the simulation (one value for each grid-point)
    ref : xr.DataArray
      data from the reference (observations) (one value for each grid-point)

    Returns
    -------
    xr.DataArray,
      Ratio between the simulation and the reference

    """
    out = sim / ref
    out.attrs.update(sim.attrs)
    out.attrs["long_name"] = "Ratio"
    out.attrs["units"] = ""
    return out


@check_same_units_and_convert
@update_xclim_history
def rmse(sim: xr.DataArray, ref: xr.DataArray) -> xr.DataArray:
    """Root mean square error.

    The root mean square error on the time dimension between the simulation and the reference.

    Parameters
    ----------
    sim : xr.DataArray
      Data from the simulation (a time-series for each grid-point)
    ref : xr.DataArray
      Data from the reference (observations) (a time-series for each grid-point)

    Returns
    -------
    xr.DataArray,
      Root mean square error between the simulation and the reference
    """

    def _rmse(sim, ref):
        return np.sqrt(np.mean((sim - ref) ** 2, axis=-1))

    out = xr.apply_ufunc(
        _rmse,
        sim,
        ref,
        input_core_dims=[["time"], ["time"]],
        dask="parallelized",
    )
    out.attrs.update(sim.attrs)
    out.attrs["long_name"] = "Root mean square"
    return out


@check_same_units_and_convert
@update_xclim_history
def mae(sim: xr.DataArray, ref: xr.DataArray) -> xr.DataArray:
    """Mean absolute error.

    The mean absolute error on the time dimension between the simulation and the reference.

    Parameters
    ----------
    sim : xr.DataArray
      data from the simulation (a time-series for each grid-point)
    ref : xr.DataArray
      data from the reference (observations) (a time-series for each grid-point)

    Returns
    -------
    xr.DataArray,
      Mean absolute error between the simulation and the reference
    """

    def _mae(sim, ref):
        return np.mean(np.abs(sim - ref), axis=-1)

    out = xr.apply_ufunc(
        _mae,
        sim,
        ref,
        input_core_dims=[["time"], ["time"]],
        dask="parallelized",
    )
    out.attrs.update(sim.attrs)
    out.attrs["long_name"] = "Mean absolute error"
    return out


@check_same_units_and_convert
@update_xclim_history
def annual_cycle_correlation(sim, ref, window: int = 15):
    """Annual cycle correlation.

    Pearson correlation coefficient between the smooth day-of-year averaged annual cycles of the simulation and
    the reference. In the smooth day-of-year averaged annual cycles, each day-of-year is averaged over all years
    and over a window of days around that day.

    Parameters
    ----------
    sim : xr.DataArray
      data from the simulation (a time-series for each grid-point)
    ref : xr.DataArray
      data from the reference (observations) (a time-series for each grid-point)
    window: int
      Size of window around each day of year around which to take the mean.
      E.g. If window=31, Jan 1st is averaged over from December 17th to January 16th.

    Returns
    -------
    xr.DataArray,
      Annual cycle correlation between the simulation and the reference

    """
    # group by day-of-year and window around each doy
    grouper_test = sdba.base.Grouper("time.dayofyear", window=window)
    # for each day, mean over X day window and over all years to create a smooth avg annual cycle
    sim_annual_cycle = grouper_test.apply("mean", sim)
    ref_annual_cycle = grouper_test.apply("mean", ref)
    out = xr.corr(ref_annual_cycle, sim_annual_cycle, dim="dayofyear")
    out.attrs.update(sim.attrs)
    out.attrs["long_name"] = "Correlation of the annual cycle"
    return out

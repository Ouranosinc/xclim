"""
Measures submodule
=================
To compare adjusted simulations to observations, through statistical properties or directly
SDBA diagnostic tests are made up of properties and measures.
This framework was inspired by the VALUE project (www.value-cost.eu/).
"""
from typing import Callable
from warnings import warn

import numpy as np
import xarray
from sklearn import metrics

from xclim import sdba
from xclim.core.formatting import update_xclim_history
from xclim.core.units import convert_units_to, units2pint


def check_same_units_and_convert(func) -> Callable:
    def _check_same_units(sim, ref, **kwargs):
        units_sim = units2pint(sim.units)
        units_ref = units2pint(ref.units)

        if units_sim != units_ref:
            warn(
                f" sim({units_sim}) and ref({units_ref}) don't have the same units."
                f" sim will be converted to {units_ref}."
            )
            sim = convert_units_to(sim, ref)
        out = func(sim, ref)
        return out

    return _check_same_units


@check_same_units_and_convert
@update_xclim_history
def bias(sim: xarray.DataArray, ref: xarray.DataArray) -> xarray.DataArray:
    r"""Bias.

    The bias is the simulation minus reference.

    Parameters
    ----------
    sim : xarray.DataArray
      data from the simulation (one value for each grid-point)
    ref : xarray.DataArray
      data from the reference (observations) (one value for each grid-point)

    Returns
    -------
    xarray.DataArray,
      Bias between the simulation and the reference

    Examples
    --------
    >>> from xclim.testing import open_dataset
    >>> tasmax = open_dataset(path_to_tasmax_file).tasmax.isel(time=0)
    >>> tasmin = open_dataset(path_to_tasmin_file).tasmin.isel(time=0)
    >>> bias(tasmax, tasmin)

    """
    out = sim - ref
    out.attrs.update(sim.attrs)
    out.attrs["long_name"] = f"Bias of the {sim.attrs['long_name']}"
    out.attrs["units"] = sim.attrs["units"]
    return out


@check_same_units_and_convert
@update_xclim_history
def relative_bias(sim: xarray.DataArray, ref: xarray.DataArray) -> xarray.DataArray:
    r"""Relative Bias.

    The relative bias is the simulation minus reference, divided by the reference.

    Parameters
    ----------
    sim : xarray.DataArray
      data from the simulation (one value for each grid-point)
    ref : xarray.DataArray
      data from the reference (observations) (one value for each grid-point)

    Returns
    -------
    xarray.DataArray,
      Relative bias between the simulation and the reference

    Examples
    --------
    >>> from xclim.testing import open_dataset
    >>> tasmax = open_dataset(path_to_tasmax_file).tasmax.isel(time=0)
    >>> tasmin = open_dataset(path_to_tasmin_file).tasmin.isel(time=0)
    >>> relative_bias(tasmax, tasmin)

    """
    out = (sim - ref) / ref
    out.attrs.update(sim.attrs)
    out.attrs["long_name"] = f"Relative bias of the {sim.attrs['long_name']}"
    out.attrs["units"] = ""
    return out


@check_same_units_and_convert
@update_xclim_history
def circular_bias(sim: xarray.DataArray, ref: xarray.DataArray) -> xarray.DataArray:
    r"""Ratio.

    Bias considering circular time series.
    Eg. The bias between doy 1 and doy 365 is 364, but the circular bias is 1.

    Parameters
    ----------
    sim : xarray.DataArray
      data from the simulation (one value for each grid-point)
    ref : xarray.DataArray
      data from the reference (observations) (one value for each grid-point)

    Returns
    -------
    xarray.DataArray,
      Circular bias between the simulation and the reference

    Examples
    --------
    >>> from xclim.testing import open_dataset
    >>> tasmax = open_dataset(path_to_tasmax_file).tasmax.isel(time=0)
    >>> tasmin = open_dataset(path_to_tasmin_file).tasmin.isel(time=0)
    >>> circular_bias(tasmax, tasmin)

    """
    out = (sim - ref) % 365
    out = out.where(
        out <= 365 / 2, 365 - out
    )  # when condition false, replace by 2nd arg
    out = out.where(ref >= sim, out * -1)
    out.attrs.update(sim.attrs)
    out.attrs["long_name"] = f"Circular bias of the {sim.attrs['long_name']}"
    return out


@check_same_units_and_convert
@update_xclim_history
def ratio(sim: xarray.DataArray, ref: xarray.DataArray) -> xarray.DataArray:
    r"""Ratio.

    The ration is the quotient of the simulation over the reference.

    Parameters
    ----------
    sim : xarray.DataArray
      data from the simulation (one value for each grid-point)
    ref : xarray.DataArray
      data from the reference (observations) (one value for each grid-point)

    Returns
    -------
    xarray.DataArray,
      Ratio between the simulation and the reference

    Examples
    --------
    >>> from xclim.testing import open_dataset
    >>> tasmax = open_dataset(path_to_tasmax_file).tasmax.isel(time=0)
    >>> tasmin = open_dataset(path_to_tasmin_file).tasmin.isel(time=0)
    >>> ratio(tasmax, tasmin)

    """
    out = sim / ref
    out.attrs.update(sim.attrs)
    out.attrs["long_name"] = f"Ratio of the {sim.attrs['long_name']}"
    out.attrs["units"] = ""
    return out


@check_same_units_and_convert
@update_xclim_history
def rmse(sim: xarray.DataArray, ref: xarray.DataArray) -> xarray.DataArray:
    r"""Root mean square error.

    The root mean square error on the time dimension between the simulation and the reference.

    Parameters
    ----------
    sim : xarray.DataArray
      data from the simulation (a time-series for each grid-point)
    ref : xarray.DataArray
      data from the reference (observations) (a time-series fro each grid-point)

    Returns
    -------
    xarray.DataArray,
      root mean square error between the simulation and the reference

    Notes
    -------
    See sklearn.metrics.mean_squared_error

    Examples
    --------
    >>> from xclim.testing import open_dataset
    >>> tasmax = open_dataset(path_to_tasmax_file).tasmax
    >>> tasmin = open_dataset(path_to_tasmin_file).tasmin
    >>> rmse(tasmax, tasmin)
    """

    def nan_sklearn(sim, ref):
        if np.isnan(sim[0]):  # sklearn can't handle the NaNs
            return np.nan
        return metrics.mean_squared_error(sim, ref, squared=False)

    out = xarray.apply_ufunc(
        nan_sklearn,
        sim,
        ref,
        input_core_dims=[["time"], ["time"]],
        vectorize=True,
        dask="parallelized",
    )
    out.attrs.update(sim.attrs)
    out.attrs["long_name"] = f"Root mean square of the {sim.attrs['long_name']}"
    return out


@check_same_units_and_convert
@update_xclim_history
def mae(sim: xarray.DataArray, ref: xarray.DataArray) -> xarray.DataArray:
    r"""Mean absolute error.

    The mean absolute error on the time dimension between the simulation and the reference.

    Parameters
    ----------
    sim : xarray.DataArray
      data from the simulation (a time-series for each grid-point)
    ref : xarray.DataArray
      data from the reference (observations)(a time-series fro each grid-point)

    Returns
    -------
    xarray.DataArray,
      Mean absolute error between the simulation and the reference

    Notes
    -------
    See sklearn.metrics.mean_absolute_error

    Examples
    --------
    >>> from xclim.testing import open_dataset
    >>> tasmax = open_dataset(path_to_tasmax_file).tasmax
    >>> tasmin = open_dataset(path_to_tasmin_file).tasmin
    >>> mae(tasmax, tasmin)
    """

    def nan_sklearn(sim, ref):
        if np.isnan(sim[0]):  # sklearn can't handle the NaNs
            return np.nan
        return metrics.mean_absolute_error(sim, ref)

    out = xarray.apply_ufunc(
        nan_sklearn,
        sim,
        ref,
        input_core_dims=[["time"], ["time"]],
        vectorize=True,
        dask="parallelized",
    )
    out.attrs.update(sim.attrs)
    out.attrs["long_name"] = f"Mean absolute error of the {sim.attrs['long_name']}"
    return out


@check_same_units_and_convert
@update_xclim_history
def annual_cycle_correlation(sim, ref, window: int = 15):
    r"""Annual cycle correlation.

    Pearson correlation coefficient between the smooth day-of-year averaged annual cycles of the simulation and
    the reference. In the smooth day-of-year averaged annual cycles, each day-of-year is average over all years
     and over a window of days around that day.

    Parameters
    ----------
    sim : xarray.DataArray
      data from the simulation (a time-series for each grid-point)
    ref : xarray.DataArray
      data from the reference (observations) (a time-series fro each grid-point)

    window: str
      Size of window around each day of year around which to take the mean.
      Eg. If window=31, Jan 1st is averaged over from December 17th to January 16th.

    Returns
    -------
    xarray.DataArray,
      Annual cycle correlation between the simulation and the reference

    Examples
    --------
    >>> from xclim.testing import open_dataset
    >>> tasmax = open_dataset(path_to_tasmax_file).tasmax
    >>> tasmin = open_dataset(path_to_tasmin_file).tasmin
    >>> annual_cycle_correlation(tasmax, tasmin)

    """
    # group by day-of-year and window around each doy
    grouper_test = sdba.base.Grouper("time.dayofyear", window=window)
    # for each day, mean over X day window and over all years to create a smooth avg annual cycle
    sim_annual_cycle = grouper_test.apply("mean", sim)
    ref_annual_cycle = grouper_test.apply("mean", ref)
    out = xarray.corr(ref_annual_cycle, sim_annual_cycle, dim="dayofyear")
    out.attrs.update(sim.attrs)
    out.attrs[
        "long_name"
    ] = f"Correlation of the annual cycle of the {sim.attrs['long_name']}"
    return out

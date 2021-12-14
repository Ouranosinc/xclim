"""
Measures submodule
=================
 To compare adjusted simulations to observations, through statistical properties or directly
 SDBA diagnostics are made up of properties and measures.
"""
import numpy as np
import xarray
from xclim.core.units import convert_units_to, check_same_units
from sklearn import metrics


@check_same_units
def bias(sim: xarray.DataArray, ref: xarray.DataArray) -> xarray.DataArray:
    r"""Bias.

    The bias is the simulation minus reference.

    Parameters
    ----------
    sim : xarray.DataArray
      data from the simulation
    ref : xarray.DataArray
      data from the reference (observations)

    Returns
    -------
    xarray.DataArray,
      Bias between the simulation and the reference

    """
    out = sim - ref
    out.attrs.update(sim.attrs)
    out.attrs["long_name"] = f"Bias of the {sim.attrs['standard_name']}"
    out.attrs["units"] = sim.attrs["units"]
    return out


@check_same_units
def relative_bias(sim: xarray.DataArray, ref: xarray.DataArray) -> xarray.DataArray:
    r""" Relative Bias.

    The relative bias is the simulation minus reference, divided by the reference.

    Parameters
    ----------
    sim : xarray.DataArray
      data from the simulation
    ref : xarray.DataArray
      data from the reference (observations)

    Returns
    -------
    xarray.DataArray,
      Relative bias between the simulation and the reference

    """
    out = (sim - ref) / ref
    out.attrs.update(sim.attrs)
    out.attrs["long_name"] = f"Relative bias of the {sim.attrs['standard_name']}"
    out.attrs["units"] = ''
    return out


@check_same_units
def circular_bias(sim: xarray.DataArray, ref: xarray.DataArray) -> xarray.DataArray:
    r""" Ratio.

    Bias considering circular time series.
    Eg. The bias between doy 1 and doy 365 is 364, but the circular bias is 1.

    Parameters
    ----------
    sim : xarray.DataArray
      data from the simulation
    ref : xarray.DataArray
      data from the reference (observations)

    Returns
    -------
    xarray.DataArray,
      Circular bias between the simulation and the reference

    """
    out = (sim - ref) % 365
    out = out.where(out <= 365/2, 365 - out)  # when condition false, replace by 2nd arg
    out = out.where(ref >= sim, out * -1)
    out.attrs.update(sim.attrs)
    out.attrs["long_name"] = f"Circular bias of the {sim.attrs['standard_name']}"
    return out

@check_same_units
def ratio(sim: xarray.DataArray, ref: xarray.DataArray) -> xarray.DataArray:
    r""" Ratio.

    The ration is the quotient of the simulation over the reference.

    Parameters
    ----------
    sim : xarray.DataArray
      data from the simulation
    ref : xarray.DataArray
      data from the reference (observations)

    Returns
    -------
    xarray.DataArray,
      Ratio between the simulation and the reference

    """
    out = sim / ref
    out.attrs.update(sim.attrs)
    out.attrs["long_name"] = f"Ratio of the {sim.attrs['standard_name']}"
    out.attrs["units"] = ''
    return out

@check_same_units
def rmse(sim: xarray.DataArray, ref: xarray.DataArray) -> xarray.DataArray:
    r""" Root mean square error.

    The root mean square error on the time dimension between the simulation and the reference.

    Parameters
    ----------
    sim : xarray.DataArray
      data from the simulation
    ref : xarray.DataArray
      data from the reference (observations)

    Returns
    -------
    xarray.DataArray,
      root mean square error between the simulation and the reference

    Notes
    -------
    See sklearn.metrics.mean_squared_error
    """
    def nan_sklearn(sim, ref):
        if np.isnan(sim[0]):  # sklearn can't handle the NaNs
            return np.nan
        else:
            return metrics.mean_squared_error(sim, ref, squared='False')

    out = xarray.apply_ufunc(nan_sklearn, sim, ref,input_core_dims=[["time"], ["time"]], vectorize=True,
                             dask='parallelized')
    out.attrs.update(sim.attrs)
    out.attrs["long_name"] = f"Root mean square of the {sim.attrs['standard_name']}"
    return out


@check_same_units
def mae(sim: xarray.DataArray, ref: xarray.DataArray) -> xarray.DataArray:
    r""" Mean absolute error.

    The mean absolute error on the time dimension between the simulation and the reference.

    Parameters
    ----------
    sim : xarray.DataArray
      data from the simulation
    ref : xarray.DataArray
      data from the reference (observations)

    Returns
    -------
    xarray.DataArray,
      Mean absolute error between the simulation and the reference

    Notes
    -------
    See sklearn.metrics.mean_absolute_error
    """
    def nan_sklearn(sim, ref):
        if np.isnan(sim[0]):  # sklearn can't handle the NaNs
            return np.nan
        else:
            return metrics.mean_absolute_error(sim, ref)
    out = xarray.apply_ufunc(nan_sklearn, sim, ref,input_core_dims=[["time"], ["time"]], vectorize=True,
                             dask='parallelized')
    out.attrs.update(sim.attrs)
    out.attrs["long_name"] = f"Mean absolute error of the {sim.attrs['standard_name']}"
    return out


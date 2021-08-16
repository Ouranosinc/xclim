# noqa: D205,D400
"""
Data flags
===========

Pseudo-indicators designed to analyse supplied variables for suspicious/erroneous indicator values.
"""
from inspect import signature
from typing import Optional

import xarray as xr

from ..indices.run_length import suspicious_run
from .calendar import climatological_mean_doy, within_bnds_doy
from .units import convert_units_to, declare_units
from .utils import VARIABLES, InputKind, MissingVariableError, infer_kind_from_parameter

_REGISTRY = dict()

__all__ = [
    "data_flags",
    "many_1mm_repetitions",
    "many_5mm_repetitions",
    "negative_precipitation_values",
    "outside_n_standard_deviations_of_climatology",
    "tas_below_tasmin",
    "tas_exceeds_tasmax",
    "tasmax_below_tas",
    "tasmax_below_tasmin",
    "tasmin_exceeds_tas",
    "tasmin_exceeds_tasmax",
    "temperature_extremely_high",
    "temperature_extremely_low",
    "values_repeating_for_5_or_more_days",
    "very_large_precipitation_events",
]


def _register_methods(func):
    _REGISTRY[func.__name__] = func
    return func


@_register_methods
@declare_units(tasmax="[temperature]", tas="[temperature]", check_output=False)
def tasmax_below_tas(tasmax: xr.DataArray, tas: xr.DataArray) -> bool:
    """Check if tasmax values are below tas values for any given day.

    Parameters
    ----------
    tasmax : xr.DataArray
    tas : xr.DataArray

    Returns
    -------
    bool
    """
    return (tasmax < tas).any()


@_register_methods
@declare_units(tasmax="[temperature]", tasmin="[temperature]", check_output=False)
def tasmax_below_tasmin(tasmax: xr.DataArray, tasmin: xr.DataArray) -> bool:
    """Check if tasmax values are below tasmin values for any given day.

    Parameters
    ----------
    tasmax : xr.DataArray
    tasmin : xr.DataArray

    Returns
    -------
    bool
    """
    return (tasmax < tasmin).any()


@_register_methods
@declare_units(tas="[temperature]", tasmax="[temperature]", check_output=False)
def tas_exceeds_tasmax(tas: xr.DataArray, tasmax: xr.DataArray) -> bool:
    """Check if tas values tasmax values for any given day.

    Parameters
    ----------
    tas : xr.DataArray
    tasmax : xr.DataArray

    Returns
    -------
    bool
    """
    return (tas > tasmax).any()


@_register_methods
@declare_units(tas="[temperature]", tasmin="[temperature]", check_output=False)
def tas_below_tasmin(tas: xr.DataArray, tasmin: xr.DataArray) -> bool:
    """Check if tas values are below tasmin values for any given day.

    Parameters
    ----------
    tas : xr.DataArray
    tasmin : xr.DataArray

    Returns
    -------
    bool
    """
    return (tas < tasmin).any()


@_register_methods
@declare_units(tasmin="[temperature]", tasmax="[temperature]", check_output=False)
def tasmin_exceeds_tasmax(tasmin: xr.DataArray, tasmax: xr.DataArray) -> bool:
    """Check if tasmin values tasmax values for any given day.

    Parameters
    ----------
    tasmin : xr.DataArray
    tasmax : xr.DataArray

    Returns
    -------
    bool
    """
    return (tasmin > tasmax).any()


@_register_methods
@declare_units(tasmin="[temperature]", tas="[temperature]", check_output=False)
def tasmin_exceeds_tas(tasmin: xr.DataArray, tas: xr.DataArray) -> bool:
    """Check if tasmin values tas values for any given day.

    Parameters
    ----------
    tasmin : xr.DataArray
    tas : xr.DataArray

    Returns
    -------
    bool
    """
    return (tasmin > tas).any()


@_register_methods
@declare_units(da="[temperature]", check_output=False)
def temperature_extremely_low(da: xr.DataArray, thresh: str = "-90 degC") -> bool:
    """Check if temperatures values are below -90 degrees Celsius for any given day.

    Parameters
    ----------
    da : xr.DataArray
    thresh : str

    Returns
    -------
    bool
    """
    thresh = convert_units_to(thresh, da)
    return (da < thresh).any()


@_register_methods
@declare_units(da="[temperature]", check_output=False)
def temperature_extremely_high(da: xr.DataArray, thresh: str = "60 degC") -> bool:
    """Check if temperatures values exceed 60 degrees Celsius for any given day.

    Parameters
    ----------
    da : xr.DataArray
    thresh : str

    Returns
    -------
    bool
    """
    thresh = convert_units_to(thresh, da)
    return (da > thresh).any()


@_register_methods
@declare_units(pr="[precipitation]", check_output=False)
def negative_precipitation_values(pr: xr.DataArray) -> bool:
    """Check if precipitation values are ever negative for any given day.

    Parameters
    ----------
    pr : xr. DataArray

    Returns
    -------
    bool
    """
    return (pr < 0).any()


@_register_methods
@declare_units(pr="[precipitation]", check_output=False)
def very_large_precipitation_events(pr: xr.DataArray, thresh="300 mm d-1") -> bool:
    """Check if precipitation values exceed 300 mm/day for any given day.

    Parameters
    ----------
    pr : xr.DataArray
    thresh : str

    Returns
    -------
    bool
    """
    thresh = convert_units_to(thresh, pr)
    return (pr > thresh).any()


@_register_methods
@declare_units(pr="[precipitation]", check_output=False)
def many_1mm_repetitions(pr: xr.DataArray) -> bool:
    """Check if precipitation values repeat at 5 mm/day for 10 or more days.

    Parameters
    ----------
    pr : xr.DataArray

    Returns
    -------
    bool
    """
    thresh = convert_units_to("1 mm d-1", pr)
    return suspicious_run(pr, window=10, op="==", thresh=thresh).any()


@_register_methods
@declare_units(pr="[precipitation]", check_output=False)
def many_5mm_repetitions(pr: xr.DataArray) -> bool:
    """Check if precipitation values repeat at 5 mm/day for 5 or more days.

    Parameters
    ----------
    pr : xr.DataArray

    Returns
    -------
    bool
    """
    thresh = convert_units_to("5 mm d-1", pr)
    return suspicious_run(pr, window=5, op="==", thresh=thresh).any()


# TODO: 'Many excessive dry days' = the amount of dry days lies outside a 14·bivariate standard deviation


@_register_methods
def outside_n_standard_deviations_of_climatology(
    da: xr.DataArray, window: int = 5, n: int = 5
) -> bool:
    """Check if any daily value is outside `n` standard deviations from the day of year mean.

    Parameters
    ----------
    da : xr.DataArray
    window : int
    n : int

    Returns
    -------
    bool
    """
    mu, sig = climatological_mean_doy(da, window=window)
    return ~within_bnds_doy(da, mu + n * sig, mu - n * sig).all()


@_register_methods
def values_repeating_for_5_or_more_days(da: xr.DataArray) -> bool:
    """Check if exact values are found to be repeating for at least 5 or more days.

    Parameters
    ----------
    da : xr.DataArray

    Returns
    -------
    bool
    """
    return suspicious_run(da, window=5).any()


def data_flags(da: xr.DataArray, ds: xr.Dataset) -> xr.Dataset:
    """Automatically evaluates the supplied DataArray for a set of tests depending on variable name and availability of extra variables within Dataset for comparison.

    Parameters
    ----------
    da : xr.DataArray
    ds : xr.Dataset

    Returns
    -------
    xr.Dataset
    """

    def _missing_vars(function, dataset: xr.Dataset):
        sig = signature(function)
        sig = sig.parameters
        extra_vars = dict()
        for i, (arg, value) in enumerate(sig.items()):
            if i == 0:
                continue
            kind = infer_kind_from_parameter(value)
            if kind == InputKind.VARIABLE:
                if arg in dataset:
                    extra_vars[arg] = dataset[arg]
                else:
                    raise MissingVariableError()
        return extra_vars

    var = str(da.name)
    flag_func = VARIABLES.get(var)["data_flags"]

    flags = dict()
    for name, kwargs in flag_func.items():
        func = _REGISTRY[name]

        try:
            extras = _missing_vars(func, ds)
        except MissingVariableError:
            flags[name] = None
        else:
            flags[name] = func(da, **extras, **(kwargs or dict()))

    dsflags = xr.Dataset(data_vars=flags)
    return dsflags

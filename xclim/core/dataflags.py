# noqa: D205,D400
"""
Data flags
===========

Pseudo-indicators designed to analyse supplied variables for suspicious/erroneous indicator values.
"""
from inspect import signature

import numpy as np
import xarray as xr

from ..indices.run_length import suspicious_run
from .calendar import climatological_mean_doy, within_bnds_doy
from .units import convert_units_to, declare_units
from .utils import VARIABLES, InputKind, MissingVariableError, infer_kind_from_parameter

_REGISTRY = dict()


class DataQualityException(Exception):
    """Raised when any data evaluation checks are flagged as True.

    Attributes:
        data_flags -- Xarray.Dataset of Data Flags

    """

    def __init__(
        self,
        flag_array: xr.Dataset,
        message="Data quality flags indicate suspicious values. Flags raised are:\n\t",
    ):
        self.message = message
        self.flags = list()
        for flag, value in flag_array.data_vars.items():
            if value.any():
                self.flags.append(value.attrs["comment"])
        super().__init__(self.message)

    def __str__(self):
        nl = "\n\t"
        return f"{self.message} {nl.join(self.flags)}."


__all__ = [
    "data_flags",
    "many_1mm_repetitions",
    "many_5mm_repetitions",
    "negative_precipitation_values",
    "outside_n_standard_deviations_of_climatology",
    "tas_below_tasmin",
    "tas_exceeds_tasmax",
    "tasmax_below_tasmin",
    "temperature_extremely_high",
    "temperature_extremely_low",
    "values_repeating_for_5_or_more_days",
    "very_large_precipitation_events",
]


def _register_methods(func):
    _REGISTRY[func.__name__] = func
    return func


@_register_methods
@declare_units(tasmax="[temperature]", tasmin="[temperature]", check_output=False)
def tasmax_below_tasmin(tasmax: xr.DataArray, tasmin: xr.DataArray) -> xr.DataArray:
    """Check if tasmax values are below tasmin values for any given day.

    Parameters
    ----------
    tasmax : xr.DataArray
    tasmin : xr.DataArray

    Returns
    -------
    xr.DataArray, [bool]

    Notes
    -----
    To gain access to the flag_array:

    .. python::
        >>> ds.tasmax < ds.tasmin
    """
    tasmax_lt_tasmin = tasmax < tasmin
    tasmax_lt_tasmin.attrs[
        "comment"
    ] = "Maximum temperature values found below minimum temperatures"
    return tasmax_lt_tasmin.any()


@_register_methods
@declare_units(tas="[temperature]", tasmax="[temperature]", check_output=False)
def tas_exceeds_tasmax(tas: xr.DataArray, tasmax: xr.DataArray) -> xr.DataArray:
    """Check if tas values tasmax values for any given day.

    Parameters
    ----------
    tas : xr.DataArray
    tasmax : xr.DataArray

    Returns
    -------
    xr.DataArray, [bool]

        Notes
    -----
    To gain access to the flag_array:

    .. python::
        >>> flagged = ds.tas > ds.tasmax
    """
    tas_gt_tasmax = tas > tasmax
    tas_gt_tasmax.attrs[
        "comment"
    ] = "Mean temperature values found above maximum temperatures"
    return tas_gt_tasmax.any()


@_register_methods
@declare_units(tas="[temperature]", tasmin="[temperature]", check_output=False)
def tas_below_tasmin(tas: xr.DataArray, tasmin: xr.DataArray) -> xr.DataArray:
    """Check if tas values are below tasmin values for any given day.

    Parameters
    ----------
    tas : xr.DataArray
    tasmin : xr.DataArray

    Returns
    -------
    xr.DataArray, [bool]

    Notes
    -----
    To gain access to the flag_array:

    .. python::
        >>> flagged = ds.tasmax < ds.tasmin
    """
    tas_lt_tasmin = tas < tasmin
    tas_lt_tasmin.attrs[
        "comment"
    ] = "Mean temperature values found below minimum temperatures"
    return tas_lt_tasmin.any()


@_register_methods
@declare_units(da="[temperature]", check_output=False)
def temperature_extremely_low(
    da: xr.DataArray, thresh: str = "-90 degC"
) -> xr.DataArray:
    """Check if temperatures values are below -90 degrees Celsius for any given day.

    Parameters
    ----------
    da : xr.DataArray
    thresh : str

    Returns
    -------
    xr.DataArray, [bool]

    Notes
    -----
    To gain access to the flag_array:

    .. python::
        >>> from xclim.core.units import convert_units_to
        >>> thresh = convert_units_to("-90 degC", ds.tas)
        >>> flagged = ds.tas < thresh
    """
    thresh = convert_units_to(thresh, da)
    extreme_low = da < thresh
    extreme_low.attrs["comment"] = f"Temperatures found below {thresh} K"
    return extreme_low.any()


@_register_methods
@declare_units(da="[temperature]", check_output=False)
def temperature_extremely_high(
    da: xr.DataArray, thresh: str = "60 degC"
) -> xr.DataArray:
    """Check if temperatures values exceed 60 degrees Celsius for any given day.

    Parameters
    ----------
    da : xr.DataArray
    thresh : str

    Returns
    -------
    xr.DataArray, [bool]

    Notes
    -----
    To gain access to the flag_array:

    .. python::
        >>> from xclim.core.units import convert_units_to
        >>> thresh = convert_units_to("60 degC", ds.tas)
        >>> flagged = ds.tas > thresh
    """
    thresh = convert_units_to(thresh, da)
    extreme_high = da > thresh
    extreme_high.attrs["comment"] = f"Temperatures found in excess of {thresh} K"
    return extreme_high.any()


@_register_methods
@declare_units(pr="[precipitation]", check_output=False)
def negative_precipitation_values(pr: xr.DataArray) -> xr.DataArray:
    """Check if precipitation values are ever negative for any given day.

    Parameters
    ----------
    pr : xr. DataArray

    Returns
    -------
    xr.DataArray, [bool]

    Notes
    -----
    To gain access to the flag_array:

    .. python::
        >>> flagged = (ds.pr < 0)
    """
    negative_precip = pr < 0
    negative_precip.attrs["comment"] = "Negative values found for precipitation"
    return negative_precip.any()


@_register_methods
@declare_units(pr="[precipitation]", check_output=False)
def very_large_precipitation_events(
    pr: xr.DataArray, thresh="300 mm d-1"
) -> xr.DataArray:
    """Check if precipitation values exceed 300 mm/day for any given day.

    Parameters
    ----------
    pr : xr.DataArray
    thresh : str

    Returns
    -------
    xr.DataArray, [bool]

    Notes
    -----
    To gain access to the flag_array:

    .. python::
        >>> from xclim.core.units import convert_units_to
        >>> thresh = convert_units_to("300 mm d-1", ds.pr)
        >>> flagged = (ds.pr > thresh)
    """
    thresh = convert_units_to(thresh, pr)
    very_large_events = (pr > thresh).any()
    very_large_events.attrs["comment"] = f"Precipitation events in excess of {thresh}"
    return very_large_events.any()


@_register_methods
@declare_units(pr="[precipitation]", check_output=False)
def many_1mm_repetitions(pr: xr.DataArray) -> xr.DataArray:
    """Check if precipitation values repeat at 5 mm/day for 10 or more days.

    Parameters
    ----------
    pr : xr.DataArray

    Returns
    -------
    xr.DataArray, [bool]

    Notes
    -----
    To gain access to the flag_array:

    .. python::
        >>> from xclim.core.units import convert_units_to
        >>> from xclim.indices.run_length import suspicious_run
        >>> thresh = convert_units_to("1 mm d-1", ds.pr)
        >>> flagged = suspicious_run(ds.pr, window=10, op="==", thresh=thresh)
    """
    thresh = convert_units_to("1 mm d-1", pr)
    repetitions = suspicious_run(pr, window=10, op="==", thresh=thresh)
    repetitions.attrs["comment"] = "Repetitive precipitation values at 1mm"
    return repetitions.any()


@_register_methods
@declare_units(pr="[precipitation]", check_output=False)
def many_5mm_repetitions(pr: xr.DataArray) -> xr.DataArray:
    """Check if precipitation values repeat at 5 mm/day for 5 or more days.

    Parameters
    ----------
    pr : xr.DataArray

    Returns
    -------
    xr.DataArray, [bool]

    Notes
    -----
    To gain access to the flag_array:

    .. python::
        >>> from xclim.core.units import convert_units_to
        >>> from xclim.indices.run_length import suspicious_run
        >>> thresh = convert_units_to("5 mm d-1", ds.pr)
        >>> flagged = suspicious_run(ds.pr, window=5, op="==", thresh=thresh)
    """
    thresh = convert_units_to("5 mm d-1", pr)
    repetitions = suspicious_run(pr, window=5, op="==", thresh=thresh)
    repetitions.attrs["comment"] = "Repetitive precipitation values at 5mm"
    return repetitions.any()


# TODO: 'Many excessive dry days' = the amount of dry days lies outside a 14·bivariate standard deviation


@_register_methods
def outside_n_standard_deviations_of_climatology(
    da: xr.DataArray, window: int = 5, n: int = 5
) -> xr.DataArray:
    """Check if any daily value is outside `n` standard deviations from the day of year mean.

    Parameters
    ----------
    da : xr.DataArray
    window : int
    n : int

    Returns
    -------
    xr.DataArray, [bool]

    Notes
    -----
    To gain access to the flag_array:

    .. python::
        >>> from xclim.core.calendar import climatological_mean_doy, within_bnds_doy
        >>> mu, sig = climatological_mean_doy(ds.variable, window=5)
        >>> std_devs = 5
        >>> flagged = ~within_bnds_doy(ds.variable, mu + std_devs * sig, mu - std_devs * sig)
    """

    mu, sig = climatological_mean_doy(da, window=window)
    within_bounds = within_bnds_doy(da, mu + n * sig, mu - n * sig)
    within_bounds.attrs[
        "comment"
    ] = f"Outside of {n} standard deviations from climatology"
    if within_bounds.all():
        return ~within_bounds.all()
    return ~within_bounds.any()


@_register_methods
def values_repeating_for_5_or_more_days(da: xr.DataArray) -> xr.DataArray:
    """Check if exact values are found to be repeating for at least 5 or more days.

    Parameters
    ----------
    da : xr.DataArray

    Returns
    -------
    xr.DataArray, [bool]

    Notes
    -----
    To gain access to the flag_array:

    .. python::
        >>> from xclim.indices.run_length import suspicious_run
        >>> flagged = suspicious_run(ds.variable, window=5)
    """
    repetition = suspicious_run(da, window=5)
    repetition.attrs["comment"] = "Runs of repetitive values for 5 or more days"
    return repetition.any()


def data_flags(
    da: xr.DataArray, ds: xr.Dataset, raise_flags: bool = False
) -> xr.Dataset:
    """Automatically evaluates the supplied DataArray for a set of data flag tests.

    Test triggers depend on variable name and availability of extra variables within Dataset for comparison.
    If called with `raise_flags=True`, will raise an Exception with comments for each quality control check raised.

    Parameters
    ----------
    da : xr.DataArray
    ds : xr.Dataset
    raise_flags : bool
      Raise exception if any of the quality assessment flags are raised. Default: False.

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
            with xr.set_options(keep_attrs=True):
                flags[name] = func(da, **extras, **(kwargs or dict()))

    dsflags = xr.Dataset(data_vars=flags)

    if raise_flags:
        if np.any(dsflags.data_vars.values()):
            raise DataQualityException(dsflags)

    return dsflags

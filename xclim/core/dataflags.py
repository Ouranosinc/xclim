# noqa: D205,D400
"""
Data flags
===========

Pseudo-indicators designed to analyse supplied variables for suspicious/erroneous indicator values.
"""
import warnings
from inspect import signature

import numpy as np
import xarray

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
        flag_array: xarray.Dataset,
        message="Data quality flags indicate suspicious values. Flags raised are:\n  - ",
    ):
        self.message = message
        self.flags = list()
        for flag, value in flag_array.data_vars.items():
            if value.any():
                self.flags.append(value.attrs["comment"])
        super().__init__(self.message)

    def __str__(self):
        nl = "\n  - "
        return f"{self.message}{nl.join(self.flags)}"


__all__ = [
    "data_flags",
    "many_1mm_repetitions",
    "many_5mm_repetitions",
    "negative_precipitation_values",
    "negative_snow_water_equivalent_values",
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
def tasmax_below_tasmin(
    tasmax: xarray.DataArray, tasmin: xarray.DataArray, dims: str = "all"
) -> xarray.DataArray:
    """Check if tasmax values are below tasmin values for any given day.

    Parameters
    ----------
    tasmax : xarray.DataArray
    tasmin : xarray.DataArray
    dims: str
      Dimenions upon which aggregation should be performed. Default: "all".

    Returns
    -------
    xarray.DataArray, [bool]

    Examples
    --------
    To gain access to the flag_array:

    >>> ds = xr.open_dataset(path_to_tas_file)
    >>> flagged = ds.tasmax < ds.tasmin
    """
    tasmax_lt_tasmin = tasmax < tasmin
    tasmax_lt_tasmin.attrs[
        "comment"
    ] = "Maximum temperature values found below minimum temperatures."
    if dims == "all":
        return tasmax_lt_tasmin.any()
    raise NotImplementedError()


@_register_methods
@declare_units(tas="[temperature]", tasmax="[temperature]", check_output=False)
def tas_exceeds_tasmax(
    tas: xarray.DataArray, tasmax: xarray.DataArray, dims: str = "all"
) -> xarray.DataArray:
    """Check if tas values tasmax values for any given day.

    Parameters
    ----------
    tas : xarray.DataArray
    tasmax : xarray.DataArray
    dims: str
      Dimenions upon which aggregation should be performed. Default: "all".

    Returns
    -------
    xarray.DataArray, [bool]

    Examples
    --------
    To gain access to the flag_array:

    >>> ds = xr.open_dataset(path_to_tas_file)
    >>> flagged = ds.tas > ds.tasmax
    """
    tas_gt_tasmax = tas > tasmax
    tas_gt_tasmax.attrs[
        "comment"
    ] = "Mean temperature values found above maximum temperatures."
    if dims == "all":
        return tas_gt_tasmax.any()
    raise NotImplementedError()


@_register_methods
@declare_units(tas="[temperature]", tasmin="[temperature]", check_output=False)
def tas_below_tasmin(
    tas: xarray.DataArray, tasmin: xarray.DataArray, dims: str = "all"
) -> xarray.DataArray:
    """Check if tas values are below tasmin values for any given day.

    Parameters
    ----------
    tas : xarray.DataArray
    tasmin : xarray.DataArray
    dims: str
      Dimenions upon which aggregation should be performed. Default: "all".

    Returns
    -------
    xarray.DataArray, [bool]

    Examples
    --------
    To gain access to the flag_array:

    >>> ds = xr.open_dataset(path_to_tas_file)
    >>> flagged = ds.tasmax < ds.tasmin
    """
    tas_lt_tasmin = tas < tasmin
    tas_lt_tasmin.attrs[
        "comment"
    ] = "Mean temperature values found below minimum temperatures."
    if dims == "all":
        return tas_lt_tasmin.any()
    raise NotImplementedError()


@_register_methods
@declare_units(da="[temperature]", check_output=False)
def temperature_extremely_low(
    da: xarray.DataArray, thresh: str = "-90 degC", dims: str = "all"
) -> xarray.DataArray:
    """Check if temperatures values are below -90 degrees Celsius for any given day.

    Parameters
    ----------
    da : xarray.DataArray
    thresh : str
    dims: str
      Dimenions upon which aggregation should be performed. Default: "all".

    Returns
    -------
    xarray.DataArray, [bool]

    Examples
    --------
    To gain access to the flag_array:

    >>> from xclim.core.units import convert_units_to
    >>> ds = xr.open_dataset(path_to_tas_file)
    >>> threshold = convert_units_to("-90 degC", ds.tas)
    >>> flagged = ds.tas < threshold
    """
    thresh_converted = convert_units_to(thresh, da)
    extreme_low = da < thresh_converted
    extreme_low.attrs["comment"] = f"Temperatures found below {thresh}."
    if dims == "all":
        return extreme_low.any()
    raise NotImplementedError()


@_register_methods
@declare_units(da="[temperature]", check_output=False)
def temperature_extremely_high(
    da: xarray.DataArray, thresh: str = "60 degC", dims: str = "all"
) -> xarray.DataArray:
    """Check if temperatures values exceed 60 degrees Celsius for any given day.

    Parameters
    ----------
    da : xarray.DataArray
    thresh : str
    dims: str
      Dimenions upon which aggregation should be performed. Default: "all".

    Returns
    -------
    xarray.DataArray, [bool]

    Examples
    --------
    To gain access to the flag_array:

    >>> from xclim.core.units import convert_units_to
    >>> ds = xr.open_dataset(path_to_tas_file)
    >>> threshold = convert_units_to("60 degC", ds.tas)
    >>> flagged = ds.tas > threshold
    """
    thresh_converted = convert_units_to(thresh, da)
    extreme_high = da > thresh_converted
    extreme_high.attrs["comment"] = f"Temperatures found in excess of {thresh}."
    if dims == "all":
        return extreme_high.any()
    raise NotImplementedError()


@_register_methods
@declare_units(pr="[precipitation]", check_output=False)
def negative_precipitation_values(
    pr: xarray.DataArray, dims: str = "all"
) -> xarray.DataArray:
    """Check if precipitation values are ever negative for any given day.

    Parameters
    ----------
    pr : xarray.DataArray
    dims: str
      Dimenions upon which aggregation should be performed. Default: "all".

    Returns
    -------
    xarray.DataArray, [bool]

    Examples
    --------
    To gain access to the flag_array:

    >>> ds = xr.open_dataset(path_to_pr_file)
    >>> flagged = (ds.pr < 0)
    """
    negative_precip = pr < 0
    negative_precip.attrs["comment"] = "Negative values found for precipitation."
    return negative_precip.any()


@_register_methods
@declare_units(pr="[precipitation]", check_output=False)
def very_large_precipitation_events(
    pr: xarray.DataArray, thresh="300 mm d-1", dims: str = "all"
) -> xarray.DataArray:
    """Check if precipitation values exceed 300 mm/day for any given day.

    Parameters
    ----------
    pr : xarray.DataArray
    thresh : str
    dims: str
      Dimenions upon which aggregation should be performed. Default: "all".

    Returns
    -------
    xarray.DataArray, [bool]

    Examples
    --------
    To gain access to the flag_array:

    >>> from xclim.core.units import convert_units_to
    >>> ds = xr.open_dataset(path_to_pr_file)
    >>> threshold = convert_units_to("300 mm d-1", ds.pr)
    >>> flagged = (ds.pr > threshold)
    """
    thresh_converted = convert_units_to(thresh, pr)
    very_large_events = (pr > thresh_converted).any()
    very_large_events.attrs["comment"] = f"Precipitation events in excess of {thresh}."
    if dims == "all":
        return very_large_events.any()
    raise NotImplementedError()


@_register_methods
@declare_units(pr="[precipitation]", check_output=False)
def many_1mm_repetitions(pr: xarray.DataArray, dims: str = "all") -> xarray.DataArray:
    """Check if precipitation values repeat at 1 mm/day for 10 or more days.

    Parameters
    ----------
    pr : xarray.DataArray
    dims: str
      Dimenions upon which aggregation should be performed. Default: "all".

    Returns
    -------
    xarray.DataArray, [bool]

    Examples
    --------
    To gain access to the flag_array:

    >>> from xclim.core.units import convert_units_to
    >>> from xclim.indices.run_length import suspicious_run
    >>> ds = xr.open_dataset(path_to_pr_file)
    >>> threshold = convert_units_to("1 mm d-1", ds.pr)
    >>> flagged = suspicious_run(ds.pr, window=10, op="==", thresh=threshold)
    """
    thresh = convert_units_to("1 mm d-1", pr)
    repetitions = suspicious_run(pr, window=10, op="==", thresh=thresh)
    repetitions.attrs[
        "comment"
    ] = "Repetitive precipitation values at 1mm d-1 for at least 10 days."
    if dims == "all":
        return repetitions.any()


@_register_methods
@declare_units(pr="[precipitation]", check_output=False)
def many_5mm_repetitions(pr: xarray.DataArray, dims: str = "all") -> xarray.DataArray:
    """Check if precipitation values repeat at 5 mm/day for 5 or more days.

    Parameters
    ----------
    pr : xarray.DataArray
    dims: str
      Dimenions upon which aggregation should be performed. Default: "all".

    Returns
    -------
    xarray.DataArray, [bool]

    Examples
    --------
    To gain access to the flag_array:

    >>> from xclim.core.units import convert_units_to
    >>> from xclim.indices.run_length import suspicious_run
    >>> ds = xr.open_dataset(path_to_pr_file)
    >>> threshold = convert_units_to("5 mm d-1", ds.pr)
    >>> flagged = suspicious_run(ds.pr, window=5, op="==", thresh=threshold)
    """
    thresh = convert_units_to("5 mm d-1", pr)
    repetitions = suspicious_run(pr, window=5, op="==", thresh=thresh)
    repetitions.attrs[
        "comment"
    ] = "Repetitive precipitation values at 5mm d-1 for at least 5 days."
    if dims == "all":
        return repetitions.any()
    raise NotImplementedError()


# TODO: 'Many excessive dry days' = the amount of dry days lies outside a 14·bivariate standard deviation


@_register_methods
@declare_units(swe="[length]", check_output=False)
def negative_snow_water_equivalent_values(
    swe: xarray.DataArray, dims: str = "all"
) -> xarray.DataArray:
    """Check if snow depth values are ever negative for any given day.

    Parameters
    ----------
    swe : xarray.DataArray
    dims: str
      Dimenions upon which aggregation should be performed. Default: "all".

    Returns
    -------
    xarray.DataArray, [bool]

    Examples
    --------
    To gain access to the flag_array:

    >>> ds = xr.open_dataset(path_to_tas_file)
    >>> flagged = (ds.swe < 0)
    """
    negative_swe = swe < 0
    negative_swe.attrs["comment"] = "Negative values found for snow water equivalent."
    if dims == "all":
        return negative_swe.any()
    raise NotImplementedError()


@_register_methods
def outside_n_standard_deviations_of_climatology(
    da: xarray.DataArray, window: int = 5, n: int = 5, dims: str = "all"
) -> xarray.DataArray:
    """Check if any daily value is outside `n` standard deviations from the day of year mean.

    Parameters
    ----------
    da : xarray.DataArray
    window : int
    n : int
    dims: str
      Dimenions upon which aggregation should be performed. Default: "all".

    Returns
    -------
    xarray.DataArray, [bool]

    Examples
    --------
    To gain access to the flag_array:

    >>> from xclim.core.calendar import climatological_mean_doy, within_bnds_doy
    >>> ds = xr.open_dataset(path_to_tas_file)
    >>> mu, sig = climatological_mean_doy(ds.tas, window=5)
    >>> std_devs = 5
    >>> flagged = ~within_bnds_doy(ds.tas, mu + std_devs * sig, mu - std_devs * sig)
    """

    mu, sig = climatological_mean_doy(da, window=window)
    within_bounds = within_bnds_doy(da, mu + n * sig, mu - n * sig)
    within_bounds.attrs[
        "comment"
    ] = f"Values found that are outside of {n} standard deviations from climatology."
    if dims == "all":
        if within_bounds.all():
            return ~within_bounds.all()
        return ~within_bounds.any()
    raise NotImplementedError()


@_register_methods
def values_repeating_for_5_or_more_days(
    da: xarray.DataArray, dims: str = "all"
) -> xarray.DataArray:
    """Check if exact values are found to be repeating for at least 5 or more days.

    Parameters
    ----------
    da : xarray.DataArray
    dims: str
      Dimenions upon which aggregation should be performed. Default: "all".

    Returns
    -------
    xarray.DataArray, [bool]

    Examples
    --------
    To gain access to the flag_array:

    >>> from xclim.indices.run_length import suspicious_run
    >>> ds = xr.open_dataset(path_to_pr_file)
    >>> flagged = suspicious_run(ds.pr, window=5)
    """
    repetition = suspicious_run(da, window=5)
    repetition.attrs["comment"] = "Runs of repetitive values for 5 or more days."
    if dims == "all":
        return repetition.any()
    raise NotImplementedError()


def data_flags(
    da: xarray.DataArray, ds: xarray.Dataset, raise_flags: bool = False
) -> xarray.Dataset:
    """Automatically evaluates the supplied DataArray for a set of data flag tests.

    Test triggers depend on variable name and availability of extra variables within Dataset for comparison.
    If called with `raise_flags=True`, will raise an Exception with comments for each quality control check raised.

    Parameters
    ----------
    da : xarray.DataArray
    ds : xarray.Dataset
    raise_flags : bool
      Raise exception if any of the quality assessment flags are raised. Default: False.

    Returns
    -------
    xarray.Dataset

    Examples
    --------
    To evaluate all applicable data flags for a given variable:

    >>> from xclim.core.dataflags import data_flags
    >>> ds = xr.open_dataset(path_to_pr_file)
    >>> flagged = data_flags(ds.pr, ds)
    """

    def _missing_vars(function, dataset: xarray.Dataset):
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
    try:
        flag_func = VARIABLES.get(var)["data_flags"]
    except TypeError:
        if raise_flags:
            raise NotImplementedError(
                f"Data quality checks do not exist for '{var}' variable."
            )
        warnings.warn(f"Data quality checks do not exist for '{var}' variable.")
        return xarray.Dataset()

    flags = dict()
    for name, kwargs in flag_func.items():
        func = _REGISTRY[name]

        try:
            extras = _missing_vars(func, ds)
        except MissingVariableError:
            flags[name] = None
        else:
            with xarray.set_options(keep_attrs=True):
                flags[name] = func(da, **extras, **(kwargs or dict()))

    dsflags = xarray.Dataset(data_vars=flags)

    if raise_flags:
        if np.any(dsflags.data_vars.values()):
            raise DataQualityException(dsflags)

    return dsflags

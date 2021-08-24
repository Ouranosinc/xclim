# noqa: D205,D400
"""
Data flags
===========

Pseudo-indicators designed to analyse supplied variables for suspicious/erroneous indicator values.
"""
import logging
from inspect import signature
from typing import Optional, Sequence, Union

import numpy as np
import xarray

from ..indices.run_length import suspicious_run
from .calendar import climatological_mean_doy, within_bnds_doy
from .units import convert_units_to, declare_units
from .utils import VARIABLES, InputKind, MissingVariableError, infer_kind_from_parameter

_REGISTRY = dict()
logging.basicConfig(format="UserWarning: %(message)s")
logger = logging.getLogger("xclim")


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
                for attribute in value.attrs.keys():
                    if str(attribute).endswith("flag"):
                        self.flags.append(value.attrs[attribute])
        super().__init__(self.message)

    def __str__(self):
        nl = "\n  - "
        return f"{self.message}{nl.join(self.flags)}"


__all__ = [
    "data_flags",
    "ecad_compliant",
    "many_1mm_repetitions",
    "many_5mm_repetitions",
    "negative_accumulation_values",
    "outside_n_standard_deviations_of_climatology",
    "percentage_values_outside_of_bounds",
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


def _sanitize_attrs(da: xarray.DataArray) -> xarray.DataArray:
    to_remove = list()
    for attr in da.attrs.keys():
        if not str(attr).endswith("flag"):
            to_remove.append(attr)
    for attr in to_remove:
        del da.attrs[attr]
    return da


@_register_methods
@declare_units(tasmax="[temperature]", tasmin="[temperature]", check_output=False)
def tasmax_below_tasmin(
    tasmax: xarray.DataArray,
    tasmin: xarray.DataArray,
) -> xarray.DataArray:
    """Check if tasmax values are below tasmin values for any given day.

    Parameters
    ----------
    tasmax : xarray.DataArray
    tasmin : xarray.DataArray

    Returns
    -------
    xarray.DataArray, [bool]

    Examples
    --------
    To gain access to the flag_array:

    >>> ds = xr.open_dataset(path_to_tas_file)
    >>> flagged = ds.tasmax < ds.tasmin
    """
    tasmax_lt_tasmin = _sanitize_attrs(tasmax < tasmin)
    tasmax_lt_tasmin.attrs[
        "tasmax_tasmin_flag"
    ] = "Maximum temperature values found below minimum temperatures."
    return tasmax_lt_tasmin


@_register_methods
@declare_units(tas="[temperature]", tasmax="[temperature]", check_output=False)
def tas_exceeds_tasmax(
    tas: xarray.DataArray,
    tasmax: xarray.DataArray,
) -> xarray.DataArray:
    """Check if tas values tasmax values for any given day.

    Parameters
    ----------
    tas : xarray.DataArray
    tasmax : xarray.DataArray

    Returns
    -------
    xarray.DataArray, [bool]

    Examples
    --------
    To gain access to the flag_array:

    >>> ds = xr.open_dataset(path_to_tas_file)
    >>> flagged = ds.tas > ds.tasmax
    """
    tas_gt_tasmax = _sanitize_attrs(tas > tasmax)
    tas_gt_tasmax.attrs[
        "tas_tasmax_flag"
    ] = "Mean temperature values found above maximum temperatures."
    return tas_gt_tasmax


@_register_methods
@declare_units(tas="[temperature]", tasmin="[temperature]", check_output=False)
def tas_below_tasmin(
    tas: xarray.DataArray, tasmin: xarray.DataArray
) -> xarray.DataArray:
    """Check if tas values are below tasmin values for any given day.

    Parameters
    ----------
    tas : xarray.DataArray
    tasmin : xarray.DataArray

    Returns
    -------
    xarray.DataArray, [bool]

    Examples
    --------
    To gain access to the flag_array:

    >>> ds = xr.open_dataset(path_to_tas_file)
    >>> flagged = ds.tasmax < ds.tasmin
    """
    tas_lt_tasmin = _sanitize_attrs(tas < tasmin)
    tas_lt_tasmin.attrs[
        "tas_tasmin_flag"
    ] = "Mean temperature values found below minimum temperatures."
    return tas_lt_tasmin


@_register_methods
@declare_units(da="[temperature]", check_output=False)
def temperature_extremely_low(
    da: xarray.DataArray, thresh: str = "-90 degC"
) -> xarray.DataArray:
    """Check if temperatures values are below -90 degrees Celsius for any given day.

    Parameters
    ----------
    da : xarray.DataArray
    thresh : str

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
    extreme_low = _sanitize_attrs(da < thresh_converted)
    extreme_low.attrs[
        f"{da.name}_flag"
    ] = f"Temperatures found below {thresh} in {da.name}."
    return extreme_low


@_register_methods
@declare_units(da="[temperature]", check_output=False)
def temperature_extremely_high(
    da: xarray.DataArray, thresh: str = "60 degC"
) -> xarray.DataArray:
    """Check if temperatures values exceed 60 degrees Celsius for any given day.

    Parameters
    ----------
    da : xarray.DataArray
    thresh : str

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
    extreme_high = _sanitize_attrs(da > thresh_converted)
    extreme_high.attrs[
        f"{da.name}_flag"
    ] = f"Temperatures found in excess of {thresh} in {da.name}."
    return extreme_high


@_register_methods
def negative_accumulation_values(
    da: xarray.DataArray,
) -> xarray.DataArray:
    """Check if variable values are negative for any given day.

    Parameters
    ----------
    da : xarray.DataArray

    Returns
    -------
    xarray.DataArray, [bool]

    Examples
    --------
    To gain access to the flag_array:

    >>> ds = xr.open_dataset(path_to_pr_file)
    >>> flagged = (ds.pr < 0)
    """
    negative_accumulations = _sanitize_attrs(da < 0)
    negative_accumulations.attrs[
        f"{da.name}_flag"
    ] = f"Negative values found for {da.name}."
    return negative_accumulations


@_register_methods
@declare_units(da="[precipitation]", check_output=False)
def very_large_precipitation_events(
    da: xarray.DataArray, thresh="300 mm d-1"
) -> xarray.DataArray:
    """Check if precipitation values exceed 300 mm/day for any given day.

    Parameters
    ----------
    da : xarray.DataArray
    thresh : str

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
    thresh_converted = convert_units_to(thresh, da)
    very_large_events = _sanitize_attrs(da > thresh_converted)
    very_large_events.attrs[
        f"{da.name}_flag"
    ] = f"Precipitation events in excess of {thresh} for {da.name}."
    return very_large_events


@_register_methods
@declare_units(da="[precipitation]", check_output=False)
def many_1mm_repetitions(da: xarray.DataArray) -> xarray.DataArray:
    """Check if precipitation values repeat at 1 mm/day for 10 or more days.

    Parameters
    ----------
    da : xarray.DataArray

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
    thresh = convert_units_to("1 mm d-1", da)
    repetitions = _sanitize_attrs(suspicious_run(da, window=10, op="==", thresh=thresh))
    repetitions.attrs[
        f"{da.name}_flag"
    ] = f"Repetitive precipitation values at 1mm d-1 for at least 10 days found for {da.name}."
    return repetitions


@_register_methods
@declare_units(da="[precipitation]", check_output=False)
def many_5mm_repetitions(da: xarray.DataArray, dims: str = "all") -> xarray.DataArray:
    """Check if precipitation values repeat at 5 mm/day for 5 or more days.

    Parameters
    ----------
    da : xarray.DataArray

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
    thresh = convert_units_to("5 mm d-1", da)
    repetitions = _sanitize_attrs(suspicious_run(da, window=5, op="==", thresh=thresh))
    repetitions.attrs[
        f"{da.name}_flag"
    ] = f"Repetitive precipitation values at 5mm d-1 for at least 5 days found for {da.name}."
    return repetitions


# TODO: 'Many excessive dry days' = the amount of dry days lies outside a 14·bivariate standard deviation


@_register_methods
def outside_n_standard_deviations_of_climatology(
    da: xarray.DataArray,
    window: int = 5,
    n: int = 5,
) -> xarray.DataArray:
    """Check if any daily value is outside `n` standard deviations from the day of year mean.

    Parameters
    ----------
    da : xarray.DataArray
    window : int
    n : int

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
    within_bounds = _sanitize_attrs(within_bnds_doy(da, mu + n * sig, mu - n * sig))
    within_bounds.attrs[
        f"{da.name}_flag"
    ] = f"Values outside of {n} standard deviations from climatology found for {da.name}."
    return ~within_bounds


@_register_methods
def values_repeating_for_5_or_more_days(da: xarray.DataArray) -> xarray.DataArray:
    """Check if exact values are found to be repeating for at least 5 or more days.

    Parameters
    ----------
    da : xarray.DataArray

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
    repetition = _sanitize_attrs(suspicious_run(da, window=5))
    repetition.attrs[
        f"{da.name}_flag"
    ] = f"Runs of repetitive values for 5 or more days found for {da.name}."
    return repetition


@_register_methods
def percentage_values_outside_of_bounds(da: xarray.DataArray) -> xarray.DataArray:
    """Check if variable values are negative for any given day.

    Parameters
    ----------
    da : xarray.DataArray

    Returns
    -------
    xarray.DataArray, [bool]

    Examples
    --------
    To gain access to the flag_array:

    >>> ds = xr.open_dataset(path_to_huss_file)  # doctest: +SKIP
    >>> flagged = (ds.huss < 0) | (ds.huss > 100)  # doctest: +SKIP
    """
    unbounded_percentages = _sanitize_attrs((da < 0) | (da > 100))
    unbounded_percentages.attrs[
        f"{da.name}_flag"
    ] = f"Percentage values found beyond bounds found for {da.name}."
    return unbounded_percentages


def data_flags(
    da: xarray.DataArray,
    ds: Optional[xarray.Dataset] = None,
    flags: Optional[dict] = None,
    dims: Union[None, str, Sequence[str]] = "all",
    freq: Optional[str] = None,
    raise_flags: bool = False,
) -> xarray.Dataset:
    """Automatically evaluates the supplied DataArray for a set of data flag tests.

    Test triggers depend on variable name and availability of extra variables within Dataset for comparison.
    If called with `raise_flags=True`, will raise an Exception with comments for each quality control check raised.

    Parameters
    ----------
    da : xarray.DataArray
      The variable to check. Must have a name that is a valid CMIP6 variable name and appears in :py:obj:`xclim.core.utils.VARIABLES`.
    ds : xarray.Dataset, optional
      An optional dataset with extra variables needed by some flags.
    flags : dict, optional
      A dictionary where the keys are the name of the flags to check and the values are parameter dictionaries. The value can be None if there are no parameters to pass (i.e. default will be used).
      The default, None, means that the data flags list will be taken from :py:obj:`xclim.core.utils.VARIABLES`.
    dims : {"all", None} or str or a sequence of strings
      Dimenions upon which aggregation should be performed. Default: "all".
    freq : str, optional
      Resampling frequency to have data_flags aggregated over periods.
      Defaults to None, which means the "time" axis is treated as any other dimension (see `dims`).
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

    The next example evaluates only one data flag, passing specific parameters. It also aggregates the flags
    yearly over the "time" dimension only, such that a True means there is a bad data point for that year at that location.

    >>> flagged = data_flags(
    ...     ds.pr,
    ...     ds,
    ...     flags={'very_large_precipitation_events': {'thresh': '250 mm d-1'}},
    ...     dims=None,
    ...     freq='YS'
    ... )
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
    if dims == "all":
        dims = da.dims
    elif isinstance(dims, str):
        # thus a single dimension name, we allow this option to mirror xarray.
        dims = {dims}
    if freq is not None and dims is not None:
        dims = (
            set(dims) - {"time"}
        ) or None  # Will return None if the only dimension was "time".

    if flags is None:
        try:
            flag_func = VARIABLES.get(var)["data_flags"]
        except (KeyError, TypeError):
            if raise_flags:
                raise NotImplementedError(
                    f"Data quality checks do not exist for '{var}' variable."
                )
            logger.warning(
                f"Data quality checks do not exist for '{var}' variable.",
                exc_info=False,
            )
            return xarray.Dataset()
    else:
        flag_func = flags

    ds = ds or xarray.Dataset()

    flags = dict()
    for name, kwargs in flag_func.items():
        func = _REGISTRY[name]
        try:
            extras = _missing_vars(func, ds)
        except MissingVariableError:
            flags[name] = None
        else:
            with xarray.set_options(keep_attrs=True):
                out = func(da, **extras, **(kwargs or dict()))

                # Aggregation
                if freq is not None:
                    out = out.resample(time=freq).any()
                if dims is not None:
                    out = out.any(dims)

            flags[name] = out

    dsflags = xarray.Dataset(data_vars=flags)

    if raise_flags:
        if np.any(dsflags.data_vars.values()):
            raise DataQualityException(dsflags)

    return dsflags


def ecad_compliant(
    ds: xarray.Dataset, raise_flags: bool = False, append: bool = True
) -> Union[xarray.DataArray, xarray.Dataset]:
    """

    Parameters
    ----------
    ds : xarray.Dataset
      Dataset containing variables to be examined.
    raise_flags : bool
      Raise exception if any of the quality assessment flags are raised. Default: False.
    append : bool
      If `True`, returns the Dataset with the `ecad_qc_flag` array appended to data_vars.
      If `False`, return the DataArray of the `ecad_qc_flag` variable.

    Returns
    -------
    Union[xarray.DataArray, xarray.Dataset]
    """
    flagged_array = xarray.Dataset()
    for var in ds.data_vars:
        df = data_flags(ds[var], ds)
        for flag in df.data_vars:
            try:
                flagged_array = xarray.merge([flagged_array, df[flag]])
            except xarray.MergeError:
                # Collect all true values for commonly-named data flags
                combined = flagged_array[flag] or df[flag]
                flagged_array[flag] = combined

    if raise_flags:
        raise DataQualityException(flagged_array)

    ecad_flag = xarray.DataArray(
        name="ecad_qc_flag",
        attrs=dict(comment="Adheres to ECAD quality control checks"),
    )
    if flagged_array.any():
        # Has suspicious values/trends -> fails
        ecad_flag.values = False
    else:
        # No flags raised -> passes
        ecad_flag.values = True

    if append:
        return xarray.merge([ds, ecad_flag])
    return ecad_flag

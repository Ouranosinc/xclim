# noqa: D205,D400
"""
Data flags
===========

Pseudo-indicators designed to analyse supplied variables for suspicious/erroneous indicator values.
"""
import logging
from decimal import Decimal
from functools import reduce
from inspect import signature
from typing import Optional, Sequence, Union

import numpy as np
import pint
import xarray

from ..indices.run_length import suspicious_run
from .calendar import climatological_mean_doy, within_bnds_doy
from .formatting import update_xclim_history
from .units import convert_units_to, declare_units, str2pint
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
                    if str(attribute) == "description":
                        self.flags.append(value.attrs[attribute])
        super().__init__(self.message)

    def __str__(self):
        nl = "\n  - "
        return f"{self.message}{nl.join(self.flags)}"


__all__ = [
    "data_flags",
    "ecad_compliant",
    "negative_accumulation_values",
    "outside_n_standard_deviations_of_climatology",
    "percentage_values_outside_of_bounds",
    "register_methods",
    "tas_below_tasmin",
    "tas_exceeds_tasmax",
    "tasmax_below_tasmin",
    "temperature_extremely_high",
    "temperature_extremely_low",
    "values_op_thresh_repeating_for_n_or_more_days",
    "values_repeating_for_n_or_more_days",
    "very_large_precipitation_events",
    "wind_values_outside_of_bounds",
]


def register_methods(func):
    _REGISTRY[func.__name__] = func
    return func


def _sanitize_attrs(da: xarray.DataArray) -> xarray.DataArray:
    to_remove = list()
    for attr in da.attrs.keys():
        if not str(attr) == "history":
            to_remove.append(attr)
    for attr in to_remove:
        del da.attrs[attr]
    return da


@register_methods
@update_xclim_history
@declare_units(tasmax="[temperature]", tasmin="[temperature]")
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

    >>> from xclim.core.dataflags import tasmax_below_tasmin
    >>> ds = xr.open_dataset(path_to_tas_file)
    >>> flagged = tasmax_below_tasmin(ds.tasmax, ds.tasmin)
    """
    tasmax_lt_tasmin = _sanitize_attrs(tasmax < tasmin)
    description = "Maximum temperature values found below minimum temperatures."
    tasmax_lt_tasmin.attrs["description"] = description
    tasmax_lt_tasmin.attrs["units"] = ""
    return tasmax_lt_tasmin


@register_methods
@update_xclim_history
@declare_units(tas="[temperature]", tasmax="[temperature]")
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

    >>> from xclim.core.dataflags import tas_exceeds_tasmax
    >>> ds = xr.open_dataset(path_to_tas_file)
    >>> flagged = tas_exceeds_tasmax(ds.tas, ds.tasmax)
    """
    tas_gt_tasmax = _sanitize_attrs(tas > tasmax)
    description = "Mean temperature values found above maximum temperatures."
    tas_gt_tasmax.attrs["description"] = description
    tas_gt_tasmax.attrs["units"] = ""
    return tas_gt_tasmax


@register_methods
@update_xclim_history
@declare_units(tas="[temperature]", tasmin="[temperature]")
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

    >>> from xclim.core.dataflags import tas_below_tasmin
    >>> ds = xr.open_dataset(path_to_tas_file)
    >>> flagged = tas_below_tasmin(ds.tas, ds.tasmin)
    """
    tas_lt_tasmin = _sanitize_attrs(tas < tasmin)
    description = "Mean temperature values found below minimum temperatures."
    tas_lt_tasmin.attrs["description"] = description
    tas_lt_tasmin.attrs["units"] = ""
    return tas_lt_tasmin


@register_methods
@update_xclim_history
@declare_units(da="[temperature]")
def temperature_extremely_low(
    da: xarray.DataArray, *, thresh: str = "-90 degC"
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

    >>> from xclim.core.dataflags import temperature_extremely_low
    >>> ds = xr.open_dataset(path_to_tas_file)
    >>> temperature = "-90 degC"
    >>> flagged = temperature_extremely_low(ds.tas, thresh=temperature)
    """
    thresh_converted = convert_units_to(thresh, da)
    extreme_low = _sanitize_attrs(da < thresh_converted)
    description = f"Temperatures found below {thresh} in {da.name}."
    extreme_low.attrs["description"] = description
    extreme_low.attrs["units"] = ""
    return extreme_low


@register_methods
@update_xclim_history
@declare_units(da="[temperature]")
def temperature_extremely_high(
    da: xarray.DataArray, *, thresh: str = "60 degC"
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

    >>> from xclim.core.dataflags import temperature_extremely_high
    >>> ds = xr.open_dataset(path_to_tas_file)
    >>> temperature = "60 degC"
    >>> flagged = temperature_extremely_high(ds.tas, thresh=temperature)
    """
    thresh_converted = convert_units_to(thresh, da)
    extreme_high = _sanitize_attrs(da > thresh_converted)
    description = f"Temperatures found in excess of {thresh} in {da.name}."
    extreme_high.attrs["description"] = description
    extreme_high.attrs["units"] = ""
    return extreme_high


@register_methods
@update_xclim_history
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

    >>> from xclim.core.dataflags import negative_accumulation_values
    >>> ds = xr.open_dataset(path_to_pr_file)
    >>> flagged = negative_accumulation_values(ds.pr)
    """
    negative_accumulations = _sanitize_attrs(da < 0)
    description = f"Negative values found for {da.name}."
    negative_accumulations.attrs["description"] = description
    negative_accumulations.attrs["units"] = ""
    return negative_accumulations


@register_methods
@update_xclim_history
@declare_units(da="[precipitation]")
def very_large_precipitation_events(
    da: xarray.DataArray, *, thresh="300 mm d-1"
) -> xarray.DataArray:
    """Check if precipitation values exceed 300 mm/day for any given day.

    Parameters
    ----------
    da : xarray.DataArray
    thresh : str
      Threshold to search array for that will trigger flag if any day exceeds value.

    Returns
    -------
    xarray.DataArray, [bool]

    Examples
    --------
    To gain access to the flag_array:

    >>> from xclim.core.dataflags import very_large_precipitation_events
    >>> ds = xr.open_dataset(path_to_pr_file)
    >>> rate = "300 mm d-1"
    >>> flagged = very_large_precipitation_events(ds.pr, thresh=rate)
    """
    thresh_converted = convert_units_to(thresh, da)
    very_large_events = _sanitize_attrs(da > thresh_converted)
    description = f"Precipitation events in excess of {thresh} for {da.name}."
    very_large_events.attrs["description"] = description
    very_large_events.attrs["units"] = ""
    return very_large_events


@register_methods
@update_xclim_history
def values_op_thresh_repeating_for_n_or_more_days(
    da: xarray.DataArray, *, n: int, thresh: str, op: str = "eq"
) -> xarray.DataArray:
    """Check if array values repeat at a given threshold for 'n' or more days.

    Parameters
    ----------
    da : xarray.DataArray
    n : int
      Number of days needed to trigger flag.
    thresh : str
      Repeating values to search for that will trigger flag.
    op : {"eq", "gt", "lt", "gteq", "lteq"}
      Operator used for comparison with thresh.

    Returns
    -------
    xarray.DataArray, [bool]

    Examples
    --------
    To gain access to the flag_array:

    >>> from xclim.core.dataflags import values_op_thresh_repeating_for_n_or_more_days
    >>> ds = xr.open_dataset(path_to_pr_file)
    >>> units = "5 mm d-1"
    >>> days = 5
    >>> comparison = "eq"
    >>> flagged = values_op_thresh_repeating_for_n_or_more_days(ds.pr, n=days, thresh=units, op=comparison)
    """
    thresh = convert_units_to(thresh, da)
    if op not in {"eq", "gt", "lt", "gteq", "lteq"}:
        raise ValueError("Operator must not be symbolic.")
    repetitions = _sanitize_attrs(suspicious_run(da, window=n, op=op, thresh=thresh))
    description = (
        f"Repetitive values at {thresh} for at least {n} days found for {da.name}."
    )
    repetitions.attrs["description"] = description
    repetitions.attrs["units"] = ""
    return repetitions


@register_methods
@update_xclim_history
@declare_units(da="[speed]")
def wind_values_outside_of_bounds(
    da: xarray.DataArray, *, lower: str = "0 m s-1", upper: str = "46 m s-1"
) -> xarray.DataArray:
    """Check if variable values fall below 0% or rise above 100% for any given day.

    Parameters
    ----------
    da : xarray.DataArray
    lower : str
      Lower limit for wind speed.
    upper : str
      Upper limit for wind speed.

    Returns
    -------
    xarray.DataArray, [bool]

    Examples
    --------
    To gain access to the flag_array:
    >>> from xclim.core.dataflags import wind_values_outside_of_bounds
    >>> ds = xr.open_dataset(path_to_tas_file)
    >>> ceiling, floor = "46 m s-1", "0 m s-1"
    >>> flagged = wind_values_outside_of_bounds(ds.wsgsmax, upper=ceiling, lower=floor)
    """
    lower, upper = convert_units_to(lower, da), convert_units_to(upper, da)
    unbounded_percentages = _sanitize_attrs((da < lower) | (da > upper))
    description = f"Percentage values exceeding bounds of {lower} and {upper} found for {da.name}."
    unbounded_percentages.attrs["description"] = description
    unbounded_percentages.attrs["units"] = ""
    return unbounded_percentages


# TODO: 'Many excessive dry days' = the amount of dry days lies outside a 14·bivariate standard deviation


@register_methods
@update_xclim_history
def outside_n_standard_deviations_of_climatology(
    da: xarray.DataArray,
    *,
    n: int,
    window: int = 5,
) -> xarray.DataArray:
    """Check if any daily value is outside `n` standard deviations from the day of year mean.

    Parameters
    ----------
    da : xarray.DataArray
    n : int
      Number of standard deviations.
    window : int
      Moving window used to determining climatological mean. Default: 5.

    Returns
    -------
    xarray.DataArray, [bool]

    Notes
    -----
    A moving window of 5 days is suggested for tas data flag calculations according to ICCLIM data quality standards.

    Examples
    --------
    To gain access to the flag_array:

    >>> from xclim.core.dataflags import outside_n_standard_deviations_of_climatology
    >>> ds = xr.open_dataset(path_to_tas_file)
    >>> std_devs = 5
    >>> average_over = 5
    >>> flagged = outside_n_standard_deviations_of_climatology(ds.tas, n=std_devs, window=average_over)
    """

    mu, sig = climatological_mean_doy(da, window=window)
    within_bounds = _sanitize_attrs(
        within_bnds_doy(da, high=(mu + n * sig), low=(mu - n * sig))
    )
    description = (
        f"Values outside of {n} standard deviations from climatology found for {da.name} "
        f"with moving window of {window} days."
    )
    within_bounds.attrs["description"] = description
    within_bounds.attrs["units"] = ""
    return ~within_bounds


@register_methods
@update_xclim_history
def values_repeating_for_n_or_more_days(
    da: xarray.DataArray, *, n: int
) -> xarray.DataArray:
    """Check if exact values are found to be repeating for at least 5 or more days.

    Parameters
    ----------
    da : xarray.DataArray
    n : int
      Number of days to trigger flag.

    Returns
    -------
    xarray.DataArray, [bool]

    Examples
    --------
    To gain access to the flag_array:

    >>> from xclim.core.dataflags import values_repeating_for_n_or_more_days
    >>> ds = xr.open_dataset(path_to_pr_file)
    >>> flagged = values_repeating_for_n_or_more_days(ds.pr, n=5)
    """
    repetition = _sanitize_attrs(suspicious_run(da, window=n))
    description = f"Runs of repetitive values for {n} or more days found for {da.name}."
    repetition.attrs["description"] = description
    repetition.attrs["units"] = ""
    return repetition


@register_methods
@update_xclim_history
def percentage_values_outside_of_bounds(da: xarray.DataArray) -> xarray.DataArray:
    """Check if variable values fall below 0% or rise above 100% for any given day.

    Parameters
    ----------
    da : xarray.DataArray

    Returns
    -------
    xarray.DataArray, [bool]

    Examples
    --------
    To gain access to the flag_array:
    >>> from xclim.core.dataflags import percentage_values_outside_of_bounds
    >>> ds = xr.open_dataset(path_to_huss_file)  # doctest: +SKIP
    >>> flagged = percentage_values_outside_of_bounds(ds.huss)  # doctest: +SKIP
    """
    unbounded_percentages = _sanitize_attrs((da < 0) | (da > 100))
    description = f"Percentage values beyond bounds found for {da.name}."
    unbounded_percentages.attrs["description"] = description
    return unbounded_percentages


def data_flags(
    da: xarray.DataArray,
    ds: Optional[xarray.Dataset] = None,
    flags: Optional[dict] = None,
    dims: Union[None, str, Sequence[str]] = "all",
    freq: Optional[str] = None,
    raise_flags: bool = False,
) -> xarray.Dataset:
    """Evaluate the supplied DataArray for a set of data flag checks.

    Test triggers depend on variable name and availability of extra variables within Dataset for comparison.
    If called with `raise_flags=True`, will raise a DataQualityException with comments for each failed quality check.

    Parameters
    ----------
    da : xarray.DataArray
      The variable to check.
      Must have a name that is a valid CMIP6 variable name and appears in :py:obj:`xclim.core.utils.VARIABLES`.
    ds : xarray.Dataset, optional
      An optional dataset with extra variables needed by some checks.
    flags : dict, optional
      A dictionary where the keys are the name of the flags to check and the values are parameter dictionaries.
      The value can be None if there are no parameters to pass (i.e. default will be used).
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
    yearly over the "time" dimension only, such that a True means there is a bad data point for that year
    at that location.

    >>> flagged = data_flags(
    ...     ds.pr,
    ...     ds,
    ...     flags={'very_large_precipitation_events': {'thresh': '250 mm d-1'}},
    ...     dims=None,
    ...     freq='YS'
    ... )
    """

    def _convert_value_to_str(var_name, val) -> str:
        """Convert variable units to an xarray data variable-like string."""
        if isinstance(val, str):
            try:
                # Use pint to
                val = str2pint(val).magnitude
                if isinstance(val, float):
                    if Decimal(val) % 1 == 0:
                        val = str(int(val))
                    else:
                        val = "point".join(str(val).split("."))
            except pint.UndefinedUnitError:
                pass

        if isinstance(val, (int, str)):
            # Replace spaces between units with underlines
            var_name = var_name.replace(f"_{param}_", f"_{str(val).replace(' ', '_')}_")
            # Change hyphens in units into the word "_minus_"
            if "-" in var_name:
                var_name = var_name.replace("-", "_minus_")

        return var_name

    def _missing_vars(function, dataset: xarray.Dataset):
        """Handle missing variables in passed datasets."""
        sig = signature(function)
        sig = sig.parameters
        extra_vars = dict()
        for i, (arg, val) in enumerate(sig.items()):
            if i == 0:
                continue
            kind = infer_kind_from_parameter(val)
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
            flag_funcs = VARIABLES.get(var)["data_flags"]
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
        flag_funcs = [flags]

    ds = ds or xarray.Dataset()

    flags = dict()
    for flag_func in flag_funcs:
        for name, kwargs in flag_func.items():
            func = _REGISTRY[name]
            variable_name = str(name)

            if kwargs:
                for param, value in kwargs.items():
                    variable_name = _convert_value_to_str(variable_name, value)
            try:
                extras = _missing_vars(func, ds)
            except MissingVariableError:
                flags[variable_name] = None
            else:
                with xarray.set_options(keep_attrs=True):
                    out = func(da, **extras, **(kwargs or dict()))

                    # Aggregation
                    if freq is not None:
                        out = out.resample(time=freq).any()
                    if dims is not None:
                        out = out.any(dims)

                flags[variable_name] = out

    dsflags = xarray.Dataset(data_vars=flags)

    if raise_flags:
        if np.any(dsflags.data_vars.values()):
            raise DataQualityException(dsflags)

    return dsflags


def ecad_compliant(
    ds: xarray.Dataset,
    dims: Union[None, str, Sequence[str]] = "all",
    raise_flags: bool = False,
    append: bool = True,
) -> Union[xarray.DataArray, xarray.Dataset, None]:
    """Run ECAD compliance tests.

    Assert file adheres to ECAD-based quality assurance checks.

    Parameters
    ----------
    ds : xarray.Dataset
      Dataset containing variables to be examined.
    dims : {"all", None} or str or a sequence of strings
      Dimenions upon which aggregation should be performed. Default: "all".
    raise_flags : bool
      Raise exception if any of the quality assessment flags are raised, otherwise returns None. Default: False.
    append : bool
      If `True`, returns the Dataset with the `ecad_qc_flag` array appended to data_vars.
      If `False`, returns the DataArray of the `ecad_qc_flag` variable.

    Returns
    -------
    Union[xarray.DataArray, xarray.Dataset]
    """
    flags = xarray.Dataset()
    history = list()
    for var in ds.data_vars:
        df = data_flags(ds[var], ds, dims=dims)
        for flag_name, flag_data in df.data_vars.items():
            flags = flags.assign({f"{var}_{flag_name}": flag_data})

            if (
                "history" in flag_data.attrs.keys()
                and np.all(flag_data.values) is not None
            ):
                # The extra `split("\n") should be removed when merge_attributes(missing_str=None)
                history_elems = flag_data.attrs["history"].split("\n")[-1].split(" ")
                if not history:
                    history.append(
                        " ".join(
                            [
                                " ".join(history_elems[0:2]),
                                " ".join(history_elems[-4:]),
                                "- Performed the following checks:",
                            ]
                        )
                    )
                history.append(" ".join(history_elems[3:-4]))

    if raise_flags:
        if np.any([flags[dv] for dv in flags.data_vars]):
            raise DataQualityException(flags)
        else:
            return

    ecad_flag = xarray.DataArray(
        ~reduce(np.logical_or, flags.data_vars.values()),  # noqa
        name="ecad_qc_flag",
        attrs=dict(
            comment="Adheres to ECAD quality control checks.",
            history="\n".join(history),
        ),
    )

    if append:
        return xarray.merge([ds, ecad_flag])
    return ecad_flag

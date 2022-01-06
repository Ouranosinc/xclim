# -*- coding: utf-8 -*-
# noqa: D205,D400
"""
Miscellaneous indices utilities
===============================

Helper functions for the indices computation, indicator construction and other things.
"""
import datetime as pydt
import logging
import os
import warnings
from collections import defaultdict
from enum import IntEnum
from functools import partial
from importlib import import_module
from importlib.resources import open_text
from inspect import Parameter
from pathlib import Path
from types import FunctionType
from typing import Any, Callable, NewType, Optional, Sequence, Tuple, Union

import cftime
import numpy as np
import pandas as pd
import xarray as xr
from boltons.funcutils import update_wrapper
from dask import array as dsk
from xarray import CFTimeIndex, DataArray, Dataset
from xarray.coding.cftime_offsets import to_cftime_datetime
from yaml import safe_dump, safe_load

logger = logging.getLogger("xclim")


# Names of calendars that have the same number of days for all years
uniform_calendars = ("noleap", "all_leap", "365_day", "366_day", "360_day")

# cftime and datetime classes to use for each calendar name
datetime_classes = {"default": pydt.datetime, **cftime._cftime.DATE_TYPES}  # noqa

#: Type annotation for strings representing full dates (YYYY-MM-DD), may include time.
DateStr = NewType("DateStr", str)

#: Type annotation for strings representing dates without a year (MM-DD).
DayOfYearStr = NewType("DayOfYearStr", str)

# Official variables definitions
VARIABLES = safe_load(open_text("xclim.data", "variables.yml"))["variables"]


def wrapped_partial(
    func: FunctionType, suggested: Optional[dict] = None, **fixed
) -> Callable:
    """Wrap a function, updating its signature but keeping its docstring.

    Parameters
    ----------
    func : FunctionType
        The function to be wrapped
    suggested : dict
        Keyword arguments that should have new default values
        but still appear in the signature.
    fixed : kwargs
        Keyword arguments that should be fixed by the wrapped
        and removed from the signature.

    Examples
    --------
    >>> from inspect import signature
    >>> def func(a, b=1, c=1):
    ...     print(a, b, c)
    >>> newf = wrapped_partial(func, b=2)
    >>> signature(newf)
    <Signature (a, *, c=1)>
    >>> newf(1)
    1 2 1
    >>> newf = wrapped_partial(func, suggested=dict(c=2), b=2)
    >>> signature(newf)
    <Signature (a, *, c=2)>
    >>> newf(1)
    1 2 2
    """
    suggested = suggested or {}
    partial_func = partial(func, **suggested, **fixed)

    fully_wrapped = update_wrapper(
        partial_func, func, injected=list(fixed.keys()), hide_wrapped=True
    )

    # Store all injected params,
    injected = getattr(func, "_injected", {}).copy()
    injected.update(fixed)
    fully_wrapped._injected = injected
    return fully_wrapped


# TODO Reconsider the utility of this
def walk_map(d: dict, func: FunctionType):
    """Apply a function recursively to values of dictionary.

    Parameters
    ----------
    d : dict
      Input dictionary, possibly nested.
    func : FunctionType
      Function to apply to dictionary values.

    Returns
    -------
    dict
      Dictionary whose values are the output of the given function.
    """
    out = {}
    for k, v in d.items():
        if isinstance(v, (dict, defaultdict)):
            out[k] = walk_map(v, func)
        else:
            out[k] = func(v)
    return out


def load_module(path: os.PathLike):
    """Load a python module from a single .py file.

    Examples
    --------
    Given a path to a module file (.py)

    >>> from pathlib import Path
    >>> path = Path(path_to_example_py)

    The two following imports are equivalent, the second uses this method.

    >>> # xdoctest: +SKIP
    >>> os.chdir(path.parent)
    >>> import example as mod1
    >>> os.chdir(previous_working_dir)
    >>> mod2 = load_module(path)
    >>> mod1 == mod2
    """
    path = Path(path)
    pwd = Path(os.getcwd())
    os.chdir(path.parent)
    try:
        mod = import_module(path.stem)
    except ModuleNotFoundError as err:
        raise err
    finally:
        os.chdir(pwd)
    return mod


class ValidationError(ValueError):
    """Error raised when input data to an indicator fails the validation tests."""

    @property
    def msg(self):  # noqa
        return self.args[0]


class MissingVariableError(ValueError):
    """Error raised when a dataset is passed to an indicator but one of the needed variable is missing."""


def ensure_chunk_size(da: xr.DataArray, **minchunks: int) -> xr.DataArray:
    """Ensure that the input dataarray has chunks of at least the given size.

    If only one chunk is too small, it is merged with an adjacent chunk.
    If many chunks are too small, they are grouped together by merging adjacent chunks.

    Parameters
    ----------
    da : xr.DataArray
      The input dataarray, with or without the dask backend. Does nothing when passed a non-dask array.
    **minchunks : Mapping[str, int]
      A kwarg mapping from dimension name to minimum chunk size.
      Pass -1 to force a single chunk along that dimension.
    """
    if not uses_dask(da):
        return da

    all_chunks = dict(zip(da.dims, da.chunks))
    chunking = dict()
    for dim, minchunk in minchunks.items():
        chunks = all_chunks[dim]
        if minchunk == -1 and len(chunks) > 1:
            # Rechunk to single chunk only if it's not already one
            chunking[dim] = -1

        toosmall = np.array(chunks) < minchunk  # Chunks that are too small
        if toosmall.sum() > 1:
            # Many chunks are too small, merge them by groups
            fac = np.ceil(minchunk / min(chunks)).astype(int)
            chunking[dim] = tuple(
                sum(chunks[i : i + fac]) for i in range(0, len(chunks), fac)
            )
            # Reset counter is case the last chunks are still too small
            chunks = chunking[dim]
            toosmall = np.array(chunks) < minchunk
        if toosmall.sum() == 1:
            # Only one, merge it with adjacent chunk
            ind = np.where(toosmall)[0][0]
            new_chunks = list(chunks)
            sml = new_chunks.pop(ind)
            new_chunks[max(ind - 1, 0)] += sml
            chunking[dim] = tuple(new_chunks)

    if chunking:
        return da.chunk(chunks=chunking)
    return da


def uses_dask(da):
    if isinstance(da, xr.DataArray) and isinstance(da.data, dsk.Array):
        return True
    if isinstance(da, xr.Dataset) and any(
        isinstance(var.data, dsk.Array) for var in da.variables.values()
    ):
        return True
    return False


def calc_perc(
    arr: np.ndarray,
    percentiles: Sequence[float] = [50.0],
    alpha: float = 1.0,
    beta: float = 1.0,
) -> np.ndarray:
    """
    Compute percentiles using nan_calc_percentiles and move the percentiles axis to the end.
    """
    return np.moveaxis(
        nan_calc_percentiles(
            arr=arr, percentiles=percentiles, axis=-1, alpha=alpha, beta=beta
        ),
        source=0,
        destination=-1,
    )


def nan_calc_percentiles(
    arr: np.ndarray,
    percentiles: Sequence[float] = [50.0],
    axis=-1,
    alpha=1.0,
    beta=1.0,
) -> np.ndarray:
    """
    Convert the percentiles to quantiles and compute them using _nan_quantile.
    """
    arr_copy = arr.copy()
    quantiles = np.array([per / 100.0 for per in percentiles])
    return _nan_quantile(arr_copy, quantiles, axis, alpha, beta)


def _compute_virtual_index(
    n: np.ndarray, quantiles: np.ndarray, alpha: float, beta: float
):
    """
    Compute the floating point indexes of an array for the linear
    interpolation of quantiles.
    n : array_like
        The sample sizes.
    quantiles : array_like
        The quantiles values.
    alpha : float
        A constant used to correct the index computed.
    beta : float
        A constant used to correct the index computed.

    alpha and beta values depend on the chosen method
    (see quantile documentation)

    Reference:
    Hyndman&Fan paper "Sample Quantiles in Statistical Packages",
    DOI: 10.1080/00031305.1996.10473566
    """
    return n * quantiles + (alpha + quantiles * (1 - alpha - beta)) - 1


def _get_gamma(virtual_indexes: np.ndarray, previous_indexes: np.ndarray):
    """
    Compute gamma (a.k.a 'm' or 'weight') for the linear interpolation
    of quantiles.

    virtual_indexes : array_like
        The indexes where the percentile is supposed to be found in the sorted
        sample.
    previous_indexes : array_like
        The floor values of virtual_indexes.

    gamma is usually the fractional part of virtual_indexes but can be modified
    by the interpolation method.
    """
    gamma = np.asanyarray(virtual_indexes - previous_indexes)
    return np.asanyarray(gamma)


def _get_indexes(
    arr: np.ndarray, virtual_indexes: np.ndarray, valid_values_count: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get the valid indexes of arr neighbouring virtual_indexes.

    Notes:
    This is a companion function to linear interpolation of quantiles

    Returns
    -------
    (previous_indexes, next_indexes): Tuple
        A Tuple of virtual_indexes neighbouring indexes
    """
    previous_indexes = np.asanyarray(np.floor(virtual_indexes))
    next_indexes = np.asanyarray(previous_indexes + 1)
    indexes_above_bounds = virtual_indexes >= valid_values_count - 1
    # When indexes is above max index, take the max value of the array
    if indexes_above_bounds.any():
        previous_indexes[indexes_above_bounds] = -1
        next_indexes[indexes_above_bounds] = -1
    # When indexes is below min index, take the min value of the array
    indexes_below_bounds = virtual_indexes < 0
    if indexes_below_bounds.any():
        previous_indexes[indexes_below_bounds] = 0
        next_indexes[indexes_below_bounds] = 0
    if np.issubdtype(arr.dtype, np.inexact):
        # After the sort, slices having NaNs will have for last element a NaN
        virtual_indexes_nans = np.isnan(virtual_indexes)
        if virtual_indexes_nans.any():
            previous_indexes[virtual_indexes_nans] = -1
            next_indexes[virtual_indexes_nans] = -1
    previous_indexes = previous_indexes.astype(np.intp)
    next_indexes = next_indexes.astype(np.intp)
    return previous_indexes, next_indexes


def _linear_interpolation(
    left: np.ndarray,
    right: np.ndarray,
    gamma: np.ndarray,
) -> np.ndarray:
    """
    Compute the linear interpolation weighted by gamma on each point of
    two same shape arrays.

    left : array_like
        Left bound.
    right : array_like
        Right bound.
    gamma : array_like
        The interpolation weight.
    """
    diff_b_a = np.subtract(right, left)
    lerp_interpolation = np.asanyarray(np.add(left, diff_b_a * gamma))
    np.subtract(
        right, diff_b_a * (1 - gamma), out=lerp_interpolation, where=gamma >= 0.5
    )
    if lerp_interpolation.ndim == 0:
        lerp_interpolation = lerp_interpolation[()]  # unpack 0d arrays
    return lerp_interpolation


def _nan_quantile(
    arr: np.ndarray,
    quantiles: np.ndarray,
    axis: int = 0,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> Union[float, np.ndarray]:
    """
    Get the quantiles of the array for the given axis.
    A linear interpolation is performed using alpha and beta.

    By default alpha == beta == 1 which performs the 7th method of Hyndman&Fan.
    with alpha == beta == 1/3 we get the 8th method.
    """
    # --- Setup
    data_axis_length = arr.shape[axis]
    if data_axis_length == 0:
        return np.NAN
    if data_axis_length == 1:
        result = np.take(arr, 0, axis=axis)
        return np.broadcast_to(result, (quantiles.size,) + result.shape)
    # The dimensions of `q` are prepended to the output shape, so we need the
    # axis being sampled from `arr` to be last.
    DATA_AXIS = 0
    if axis != DATA_AXIS:  # But moveaxis is slow, so only call it if axis!=0.
        arr = np.moveaxis(arr, axis, destination=DATA_AXIS)
    # nan_count is not a scalar
    nan_count = np.isnan(arr).sum(axis=DATA_AXIS).astype(float)
    valid_values_count = data_axis_length - nan_count
    # We need at least two values to do an interpolation
    too_few_values = valid_values_count < 2
    if too_few_values.any():
        # This will result in getting the only available value if it exist
        valid_values_count[too_few_values] = np.NaN
    # --- Computation of indexes
    # Add axis for quantiles
    valid_values_count = valid_values_count[..., np.newaxis]
    virtual_indexes = _compute_virtual_index(valid_values_count, quantiles, alpha, beta)
    virtual_indexes = np.asanyarray(virtual_indexes)
    previous_indexes, next_indexes = _get_indexes(
        arr, virtual_indexes, valid_values_count
    )
    # --- Sorting
    arr.sort(axis=DATA_AXIS)
    # --- Get values from indexes
    arr = arr[..., np.newaxis]
    previous = np.squeeze(
        np.take_along_axis(arr, previous_indexes.astype(int)[np.newaxis, ...], axis=0),
        axis=0,
    )
    next_elements = np.squeeze(
        np.take_along_axis(arr, next_indexes.astype(int)[np.newaxis, ...], axis=0),
        axis=0,
    )
    # --- Linear interpolation
    gamma = _get_gamma(virtual_indexes, previous_indexes)
    interpolation = _linear_interpolation(previous, next_elements, gamma)
    # When an interpolation is in Nan range, (near the end of the sorted array) it means
    # we can clip to the array max value.
    result = np.where(np.isnan(interpolation), np.nanmax(arr, axis=0), interpolation)
    # Move quantile axis in front
    result = np.moveaxis(result, axis, 0)
    return result


def raise_warn_or_log(
    err: Exception,
    mode: str,
    msg: Optional[str] = None,
    err_type=ValueError,
    stacklevel: int = 1,
):
    """Raise, warn or log an error according.

    Parameters
    ----------
    err : Exception
      An error.
    mode : {'ignore', 'log', 'warn', 'raise'}
      What to do with the error.
    msg : str, optional
      The string used when logging or warning.
      Defaults to the `msg` attr of the error (if present) or to "Failed with <err>".
    stacklevel : int
      Stacklevel when warning. Relative to the call of this function (1 is added).
    """
    msg = msg or getattr(err, "msg", f"Failed with {err!r}.")
    if mode == "ignore":
        pass
    elif mode == "log":
        logger.info(msg)
    elif mode == "warn":
        warnings.warn(msg, stacklevel=stacklevel + 1)
    else:  # mode == "raise"
        raise err from err_type(msg)


class InputKind(IntEnum):
    """Constants for input parameter kinds.

    For use by external parses to determine what kind of data the indicator expects.
    On the creation of an indicator, the appropriate constant is stored in
    :py:attr:`xclim.core.indicator.Indicator.parameters`. The integer value is what gets stored in the output
    of :py:meth:`xclim.core.indicator.Indicator.json`.

    For developers : for each constant, the docstring specifies the annotation a parameter of an indice function
    should use in order to be picked up by the indicator constructor.
    """

    VARIABLE = 0
    """A data variable (DataArray or variable name).

       Annotation : ``xr.DataArray``.
    """
    OPTIONAL_VARIABLE = 1
    """An optional data variable (DataArray or variable name).

       Annotation : ``xr.DataArray`` or ``Optional[xr.DataArray]``.
    """
    QUANTITY_STR = 2
    """A string representing a quantity with units.

       Annotation : ``str`` +  an entry in the :py:func:`xclim.core.units.declare_units` decorator.
    """
    FREQ_STR = 3
    """A string representing an "offset alias", as defined by pandas.

       See https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases .
       Annotation : ``str`` + ``freq`` as the parameter name.
    """
    NUMBER = 4
    """A number.

       Annotation : ``int``, ``float`` and Unions and Optionals thereof.
    """
    STRING = 5
    """A simple string.

       Annotation : ``str`` or ``Optional[str]``. In most cases, this kind of parameter makes sense with choices indicated
       in the docstring's version of the annotation with curly braces. See :ref:`Defining new indices`.
    """
    DAY_OF_YEAR = 6
    """A date, but without a year, in the MM-DD format.

       Annotation : :py:obj:`xclim.core.utils.DayOfYearStr` (may be optional).
    """
    DATE = 7
    """A date in the YYYY-MM-DD format, may include a time.

       Annotation : :py:obj:`xclim.core.utils.DateStr` (may be optional).
    """
    NUMBER_SEQUENCE = 8
    """A sequence of numbers

       Annotation : ``Sequence[int]``, ``Sequence[float]`` and ``Union`` thereof, may include single ``int`` and ``float``.
    """
    BOOL = 9
    """A boolean flag.

       Annotation : ``bool``, or optional thereof.
    """
    KWARGS = 50
    """A mapping from argument name to value.

       Developers : maps the ``**kwargs``. Please use as little as possible.
    """
    DATASET = 70
    """An xarray dataset.

       Developers : as indices only accept DataArrays, this should only be added on the indicator's constructor.
    """
    OTHER_PARAMETER = 99
    """An object that fits None of the previous kinds.

       Developers : This is the fallback kind, it will raise an error in xclim's unit tests if used.
    """


def _typehint_is_in(hint, hints):
    """Returns whether the first argument is in the other arguments.

    If the first arg is an Union of several typehints, this returns True only
    if all the members of that Union are in the given list.
    """
    # This code makes use of the "set-like" property of Unions and Optionals:
    # Optional[X, Y] == Union[X, Y, None] == Union[X, Union[X, Y], None] etc.
    return Union[(hint,) + tuple(hints)] == Union[tuple(hints)]


def infer_kind_from_parameter(param: Parameter, has_units: bool = False) -> InputKind:
    """Returns the appropriate InputKind constant from an ``inspect.Parameter`` object.

    The correspondance between parameters and kinds is documented in :py:class:`xclim.core.utils.InputKind`.
    The only information not inferable through the inspect object is whether the parameter
    has been assigned units through the :py:func:`xclim.core.units.declare_units` decorator.
    That can be given with the ``has_units`` flag.
    """
    if (
        param.annotation in [DataArray, Union[DataArray, str]]
        and param.default is not None
    ):
        return InputKind.VARIABLE

    if Optional[param.annotation] in [
        Optional[DataArray],
        Optional[Union[DataArray, str]],
    ]:
        return InputKind.OPTIONAL_VARIABLE

    if _typehint_is_in(param.annotation, (str, None)) and has_units:
        return InputKind.QUANTITY_STR

    if param.name == "freq":
        return InputKind.FREQ_STR

    if _typehint_is_in(param.annotation, (None, int, float)):
        return InputKind.NUMBER

    if _typehint_is_in(
        param.annotation, (None, int, float, Sequence[int], Sequence[float])
    ):
        return InputKind.NUMBER_SEQUENCE

    if _typehint_is_in(param.annotation, (None, str)):
        return InputKind.STRING

    if _typehint_is_in(param.annotation, (None, DayOfYearStr)):
        return InputKind.DAY_OF_YEAR

    if _typehint_is_in(param.annotation, (None, DateStr)):
        return InputKind.DATE

    if _typehint_is_in(param.annotation, (None, bool)):
        return InputKind.BOOL

    if _typehint_is_in(param.annotation, (None, Dataset)):
        return InputKind.DATASET

    if param.kind == param.VAR_KEYWORD:
        return InputKind.KWARGS

    return InputKind.OTHER_PARAMETER


def adapt_clix_meta_yaml(raw: os.PathLike, adapted: os.PathLike):
    """Reads in a clix-meta yaml and refactors it to fit xclim's yaml specifications."""
    from xclim.indices import generic

    freq_names = {"annual": "A", "seasonal": "Q", "monthly": "M", "weekly": "W"}
    freq_defs = {"annual": "YS", "seasonal": "QS-DEC", "monthly": "MS", "weekly": "W"}

    with open(raw) as f:
        yml = safe_load(f)

    yml["realm"] = "atmos"
    yml[
        "doc"
    ] = """  ===================
  CF Standard indices
  ===================

  Indicator found here are defined by the team at `clix-meta`_.
  Adapted documentation from that repository follows:

  The repository aims to provide a platform for thinking about, and developing,
  a unified view of metadata elements required to describe climate indices (aka climate indicators).

  To facilitate data exchange and dissemination the metadata should, as far as possible,
  follow the Climate and Forecasting (CF) Conventions. Considering the very rich and diverse flora of
  climate indices this is however not always possible. By collecting a wide range of different indices
  it is easier to discover any common patterns and features that are currently not well covered by the
  CF Conventions. Currently identified issues frequently relate to standard_name or/and cell_methods
  which both are controlled vocabularies of the CF Conventions.

  .. _clix-meta: https://github.com/clix-meta/clix-meta
"""
    yml["references"] = "clix-meta https://github.com/clix-meta/clix-meta"

    remove_ids = []
    rename_ids = {}
    for cmid, data in yml["indices"].items():
        if "reference" in data:
            data["references"] = data.pop("reference")

        index_function = data.pop("index_function")

        data["compute"] = index_function["name"]
        if getattr(generic, data["compute"], None) is None:
            remove_ids.append(cmid)
            print(
                f"Indicator {cmid} uses non-implemented function {data['compute']}, removing."
            )
            continue

        if (data["output"].get("standard_name") or "").startswith(
            "number_of_days"
        ) or cmid == "nzero":
            remove_ids.append(cmid)
            print(
                f"Indicator {cmid} has a 'number_of_days' standard name and xclim disagrees with the CF conventions on the correct output units, removing."
            )
            continue

        if (data["output"].get("standard_name") or "").endswith("precipitation_amount"):
            remove_ids.append(cmid)
            print(
                f"Indicator {cmid} has a 'precipitation_amount' standard name and clix-meta has incoherent output units, removing."
            )
            continue

        rename_params = {}
        if index_function["parameters"]:
            data["parameters"] = index_function["parameters"]
            for name, param in data["parameters"].copy().items():
                if param["kind"] in ["operator", "reducer"]:
                    data["parameters"][name] = param[param["kind"]]
                else:  # kind = quantity
                    if param.get("proposed_standard_name") == "temporal_window_size":
                        # Window, nothing to do.
                        del data["parameters"][name]
                    elif isinstance(param["data"], dict):
                        # No value
                        data["parameters"][name] = {
                            "description": param.get(
                                "long_name",
                                param.get(
                                    "proposed_standard_name", param.get("standard_name")
                                ).replace("_", " "),
                            ),
                            "units": param["units"],
                        }
                        rename_params[
                            f"{{{name}}}"
                        ] = f"{{{list(param['data'].keys())[0]}}}"
                    else:
                        # Value
                        data["parameters"][name] = f"{param['data']} {param['units']}"

        period = data.pop("period")
        data["allowed_periods"] = [freq_names[per] for per in period["allowed"].keys()]
        data.setdefault("parameters", {})["freq"] = {
            "default": freq_defs[period["default"]]
        }

        attrs = {}
        for attr, val in data.pop("output").items():
            if val is None:
                continue
            if attr == "cell_methods":
                methods = []
                for cell_method in val:
                    methods.append(
                        "".join([f"{dim}: {meth}" for dim, meth in cell_method.items()])
                    )
                val = " ".join(methods)
            elif attr in ["var_name", "long_name"]:
                for new, old in rename_params.items():
                    val = val.replace(old, new)
            attrs[attr] = val
        data["cf_attrs"] = [attrs]

        del data["ET"]

        if "{" in cmid:
            rename_ids[cmid] = cmid.replace("{", "").replace("}", "")

    for old, new in rename_ids.items():
        yml["indices"][new] = yml["indices"].pop(old)

    for cmid in remove_ids:
        del yml["indices"][cmid]

    with open(adapted, "w") as f:
        safe_dump(yml, f)


# Datetime-relevant utilities


def days_in_year(year: int, calendar: str = "default") -> int:
    """Return the number of days in the input year according to the input calendar."""
    return (
        (datetime_classes[calendar](year + 1, 1, 1) - pydt.timedelta(days=1))
        .timetuple()
        .tm_yday
    )


def date_range(
    *args, calendar: str = "default", **kwargs
) -> Union[pd.DatetimeIndex, CFTimeIndex]:
    """Wrap pd.date_range (if calendar == 'default') or xr.cftime_range (otherwise)."""
    if calendar == "default":
        return pd.date_range(*args, **kwargs)
    return xr.cftime_range(*args, calendar=calendar, **kwargs)


def select_time(
    da: Union[xr.DataArray, xr.Dataset],
    drop: bool = False,
    season: Union[str, Sequence[str]] = None,
    month: Union[int, Sequence[int]] = None,
    doy_bounds: Tuple[int, int] = None,
    date_bounds: Tuple[str, str] = None,
):
    """Select entries according to a time period.

    This conveniently improves xarray's :py:meth:`xarray.DataArray.where` and
    :py:meth:`xarray.DataArray.sel` with fancier ways of indexing over time elements.
    In addition to the data `da` and argument `drop`, only one of `season`, `month`,
    `doy_bounds` or `date_bounds` may be passed.

    Parameters
    ----------
    da : xr.DataArray or xr.Dataset
      Input data.
    drop: boolean
      Whether to drop elements outside the period of interest or
      to simply mask them (default).
    season: string or sequence of strings
      One or more of 'DJF', 'MAM', 'JJA' and 'SON'.
    month: integer or sequence of integers
      Sequence of month numbers (January = 1 ... December = 12)
    doy_bounds: 2-tuple of integers
      The bounds as (start, end) of the period of interest expressed in day-of-year,
      integers going from 1 (January 1st) to 365 or 366 (December 31st). If calendar
      awareness is needed, consider using ``date_bounds`` instead.
      Bounds are inclusive.
    date_bounds: 2-tuple of strings
      The bounds as (start, end) of the period of interest expressed as dates in the
      month-day (%m-%d) format.
      Bounds are inclusive.

    Returns
    -------
    xr.DataArray or xr.Dataset
      Selected input values. If ``drop=False``, this has the same length as ``da``
      (along dimension 'time'), but with masked (NaN) values outside the period of
      interest.

    Examples
    --------
    Keep only the values of fall and spring.

    >>> ds = open_dataset("ERA5/daily_surface_cancities_1990-1993.nc")
    >>> ds.time.size
    1461
    >>> out = select_time(ds, drop=True, season=['MAM', 'SON'])
    >>> out.time.size
    732

    Or all values between two dates (included).

    >>> out = select_time(ds, drop=True, date_bounds=('02-29', '03-02'))
    >>> out.time.values
    array(['1990-03-01T00:00:00.000000000', '1990-03-02T00:00:00.000000000',
           '1991-03-01T00:00:00.000000000', '1991-03-02T00:00:00.000000000',
           '1992-02-29T00:00:00.000000000', '1992-03-01T00:00:00.000000000',
           '1992-03-02T00:00:00.000000000', '1993-03-01T00:00:00.000000000',
           '1993-03-02T00:00:00.000000000'], dtype='datetime64[ns]')
    """
    N = sum(arg is not None for arg in [season, month, doy_bounds, date_bounds])
    if N > 1:
        raise ValueError(f"Only one method of indexing may be given, got {N}.")

    if N == 0:
        return da

    def get_doys(start, end):
        if start <= end:
            return np.arange(start, end + 1)
        return np.concatenate((np.arange(start, 367), np.arange(0, end + 1)))

    if season is not None:
        if isinstance(season, str):
            season = [season]
        mask = da.time.dt.season.isin(season)

    elif month is not None:
        if isinstance(month, int):
            month = [month]
        mask = da.time.dt.month.isin(month)

    elif doy_bounds is not None:
        mask = da.time.dt.dayofyear.isin(get_doys(*doy_bounds))

    elif date_bounds is not None:
        # This one is a bit trickier.
        start, end = date_bounds
        time = da.time
        calendar = get_calendar(time)
        if calendar not in uniform_calendars:
            # For non-uniform calendars, we can't simply convert dates to doys
            # conversion to all_leap is safe for all non-uniform calendar as it doesn't remove any date.
            time = convert_calendar(time, "all_leap")
            # values of time are the _old_ calendar
            # and the new calendar is in the coordinate
            calendar = "all_leap"

        # Get doy of date, this is now safe because the calendar is uniform.
        doys = get_doys(
            to_cftime_datetime("2000-" + start, calendar).dayofyr,
            to_cftime_datetime("2000-" + end, calendar).dayofyr,
        )
        mask = time.time.dt.dayofyear.isin(doys)
        # Needed if we converted calendar, this puts back the correct coord
        mask["time"] = da.time

    return da.where(mask, drop=drop)


def get_calendar(obj: Any, dim: str = "time") -> str:
    """Return the calendar of an object.

    Parameters
    ----------
    obj : Any
      An object defining some date.
      If `obj` is an array/dataset with a datetime coordinate, use `dim` to specify its name.
      Values must have either a datetime64 dtype or a cftime dtype.
      `obj` can also be a python datetime.datetime, a cftime object or a pandas Timestamp
      or an iterable of those, in which case the calendar is inferred from the first value.
    dim : str
      Name of the coordinate to check (if `obj` is a DataArray or Dataset).

    Raises
    ------
    ValueError
      If no calendar could be inferred.

    Returns
    -------
    str
      The cftime calendar name or "default" when the data is using numpy's or python's datetime types.
      Will always return "standard" instead of "gregorian", following CF conventions 1.9.
    """
    if isinstance(obj, (xr.DataArray, xr.Dataset)):
        if obj[dim].dtype == "O":
            obj = obj[dim].where(obj[dim].notnull(), drop=True)[0].item()
        elif "datetime64" in obj[dim].dtype.name:
            return "default"
    elif isinstance(obj, xr.CFTimeIndex):
        obj = obj.values[0]
    else:
        obj = np.take(obj, 0)
        # Take zeroth element, overcome cases when arrays or lists are passed.
    if isinstance(obj, pydt.datetime):  # Also covers pandas Timestamp
        return "default"
    if isinstance(obj, cftime.datetime):
        if obj.calendar == "gregorian":
            return "standard"
        return obj.calendar

    raise ValueError(f"Calendar could not be inferred from object of type {type(obj)}.")


def convert_calendar(
    source: Union[xr.DataArray, xr.Dataset],
    target: Union[xr.DataArray, str],
    align_on: Optional[str] = None,
    missing: Optional[Any] = None,
    dim: str = "time",
) -> Union[xr.DataArray, xr.Dataset]:
    """Convert a DataArray/Dataset to another calendar using the specified method.

    Only converts the individual timestamps, does not modify any data except in dropping invalid/surplus dates or inserting missing dates.

    If the source and target calendars are either no_leap, all_leap or a standard type, only the type of the time array is modified.
    When converting to a leap year from a non-leap year, the 29th of February is removed from the array.
    In the other direction and if `target` is a string, the 29th of February will be missing in the output,
    unless `missing` is specified, in which case that value is inserted.

    For conversions involving `360_day` calendars, see Notes.

    This method is safe to use with sub-daily data as it doesn't touch the time part of the timestamps.

    Parameters
    ----------
    source : xr.DataArray
      Input array/dataset with a time coordinate of a valid dtype (datetime64 or a cftime.datetime).
    target : Union[xr.DataArray, str]
      Either a calendar name or the 1D time coordinate to convert to.
      If an array is provided, the output will be reindexed using it and in that case, days in `target`
      that are missing in the converted `source` are filled by `missing` (which defaults to NaN).
    align_on : {None, 'date', 'year', 'random'}
      Must be specified when either source or target is a `360_day` calendar, ignored otherwise. See Notes.
    missing : Optional[any]
      A value to use for filling in dates in the target that were missing in the source.
      If `target` is a string, default (None) is not to fill values. If it is an array, default is to fill with NaN.
    dim : str
      Name of the time coordinate.

    Returns
    -------
    Union[xr.DataArray, xr.Dataset]
      Copy of source with the time coordinate converted to the target calendar.
      If `target` is given as an array, the output is reindexed to it, with fill value `missing`.
      If `target` was a string and `missing` was None (default), invalid dates in the new calendar are dropped, but missing dates are not inserted.
      If `target` was a string and `missing` was given, then start, end and frequency of the new time axis are inferred and
      the output is reindexed to that a new array.

    Notes
    -----
    If one of the source or target calendars is `360_day`, `align_on` must be specified and two options are offered.

    "year"
      The dates are translated according to their rank in the year (dayofyear), ignoring their original month and day information,
      meaning that the missing/surplus days are added/removed at regular intervals.

      From a `360_day` to a standard calendar, the output will be missing the following dates (day of year in parenthesis):
        To a leap year:
          January 31st (31), March 31st (91), June 1st (153), July 31st (213), September 31st (275) and November 30th (335).
        To a non-leap year:
          February 6th (36), April 19th (109), July 2nd (183), September 12th (255), November 25th (329).

      From standard calendar to a '360_day', the following dates in the source array will be dropped:
        From a leap year:
          January 31st (31), April 1st (92), June 1st (153), August 1st (214), September 31st (275), December 1st (336)
        From a non-leap year:
          February 6th (37), April 20th (110), July 2nd (183), September 13th (256), November 25th (329)

      This option is best used on daily and subdaily data.

    "date"
      The month/day information is conserved and invalid dates are dropped from the output. This means that when converting from
      a `360_day` to a standard calendar, all 31st (Jan, March, May, July, August, October and December) will be missing as there is no equivalent
      dates in the `360_day` and the 29th (on non-leap years) and 30th of February will be dropped as there are no equivalent dates in
      a standard calendar.

      This option is best used with data on a frequency coarser than daily.

    "random"
      Similar to "year", each day of year of the source is mapped to another day of year
      of the target. However, instead of having always the same missing days according
      the source and target years, here 5 days are chosen randomly, one for each fifth
      of the year. However, February 29th is always missing when converting to a leap year,
      or its value is dropped when converting from a leap year. This is similar to method
      used in the [LOCA]_ dataset.

      This option best used on daily data.

    References
    ----------
    .. [LOCA] Pierce, D. W., D. R. Cayan, and B. L. Thrasher, 2014: Statistical downscaling using Localized Constructed Analogs (LOCA). Journal of Hydrometeorology, volume 15, page 2558-2585

    Examples
    --------
    This method does not try to fill the missing dates other than with a constant value,
    passed with `missing`. In order to fill the missing dates with interpolation, one
    can simply use xarray's method:

    >>> tas_nl = convert_calendar(tas, 'noleap')  # For the example
    >>> with_missing = convert_calendar(tas_nl, 'standard', missing=np.NaN)
    >>> out = with_missing.interpolate_na('time', method='linear')

    Here, if Nans existed in the source data, they will be interpolated too. If that is,
    for some reason, not wanted, the workaround is to do:

    >>> mask = convert_calendar(tas_nl, 'standard').notnull()
    >>> out2 = out.where(mask)
    """
    cal_src = get_calendar(source, dim=dim)

    if isinstance(target, str):
        cal_tgt = target
    else:
        cal_tgt = get_calendar(target, dim=dim)

    if cal_src == cal_tgt:
        return source

    if (cal_src == "360_day" or cal_tgt == "360_day") and align_on not in [
        "year",
        "date",
        "random",
    ]:
        raise ValueError(
            "Argument `align_on` must be specified with either 'date', 'year' or "
            "'random' when converting to or from a '360_day' calendar."
        )
    if cal_src != "360_day" and cal_tgt != "360_day":
        align_on = None

    out = source.copy()
    # TODO Maybe the 5-6 days to remove could be given by the user?
    if align_on in ["year", "random"]:
        if align_on == "year":

            def _yearly_interp_doy(time):
                # Returns the nearest day in the target calendar of the corresponding "decimal year" in the source calendar
                yr = int(time.dt.year[0])
                return np.round(
                    days_in_year(yr, cal_tgt)
                    * time.dt.dayofyear
                    / days_in_year(yr, cal_src)
                ).astype(int)

            new_doy = source.time.groupby(f"{dim}.year").map(_yearly_interp_doy)
        elif align_on == "random":

            def _yearly_random_doy(time, rng):
                # Return a doy in the new calendar, removing the Feb 29th and 5 other
                # days chosen randomly within 5 sections of 72 days.
                yr = int(time.dt.year[0])
                new_doy = np.arange(360) + 1
                rm_idx = rng.integers(0, 72, 5) + (np.arange(5) * 72)
                if cal_src == "360_day":
                    for idx in rm_idx:
                        new_doy[idx + 1 :] = new_doy[idx + 1 :] + 1
                    if days_in_year(yr, cal_tgt) == 366:
                        new_doy[new_doy >= 60] = new_doy[new_doy >= 60] + 1
                elif cal_tgt == "360_day":
                    new_doy = np.insert(new_doy, rm_idx - np.arange(5), -1)
                    if days_in_year(yr, cal_src) == 366:
                        new_doy = np.insert(new_doy, 60, -1)
                return new_doy[time.dt.dayofyear - 1]

            new_doy = source.time.groupby(f"{dim}.year").map(
                _yearly_random_doy, rng=np.random.default_rng()
            )

        # Convert the source datetimes, but override the doy with our new doys
        out[dim] = xr.DataArray(
            [
                _convert_datetime(datetime, new_doy=doy, calendar=cal_tgt)
                for datetime, doy in zip(source[dim].indexes[dim], new_doy)
            ],
            dims=(dim,),
            name=dim,
        )
        # Remove NaN that where put on invalid dates in target calendar
        out = out.where(out[dim].notnull(), drop=True)
        # Remove duplicate timestamps, happens when reducing the number of days
        out = out.isel({dim: np.unique(out[dim], return_index=True)[1]})
    else:
        time_idx = source[dim].indexes[dim]
        out[dim] = xr.DataArray(
            [_convert_datetime(time, calendar=cal_tgt) for time in time_idx],
            dims=(dim,),
            name=dim,
        )
        # Remove NaN that where put on invalid dates in target calendar
        out = out.where(out[dim].notnull(), drop=True)

    if isinstance(target, str) and missing is not None:
        target = date_range_like(source[dim], cal_tgt)

    if isinstance(target, xr.DataArray):
        out = out.reindex({dim: target}, fill_value=missing or np.nan)

    # Copy attrs but change remove `calendar` is still present.
    out[dim].attrs.update(source[dim].attrs)
    out[dim].attrs.pop("calendar", None)
    return out


def date_range_like(source: xr.DataArray, calendar: str) -> xr.DataArray:
    """Generate a datetime array with the same frequency, start and end as another one, but in a different calendar.

    Parameters
    ----------
    source : xr.DataArray
      1D datetime coordinate DataArray
    calendar : str
      New calendar name.

    Raises
    ------
    ValueError
      If the source's frequency was not found.

    Returns
    -------
    xr.DataArray
      1D datetime coordinate with the same start, end and frequency as the source, but in the new calendar.
        The start date is assumed to exist in the target calendar.
        If the end date doesn't exist, the code tries 1 and 2 calendar days before.
        Exception when the source is in 360_day and the end of the range is the 30th of a 31-days month,
        then the 31st is appended to the range.
    """
    freq = xr.infer_freq(source)
    if freq is None:
        raise ValueError(
            "`date_range_like` was unable to generate a range as the source frequency was not inferrable."
        )

    src_cal = get_calendar(source)
    if src_cal == calendar:
        return source

    index = source.indexes[source.dims[0]]
    end_src = index[-1]
    end = _convert_datetime(end_src, calendar=calendar)
    if end is np.nan:  # Day is invalid, happens at the end of months.
        end = _convert_datetime(end_src.replace(day=end_src.day - 1), calendar=calendar)
        if end is np.nan:  # Still invalid : 360_day to non-leap february.
            end = _convert_datetime(
                end_src.replace(day=end_src.day - 2), calendar=calendar
            )
    if src_cal == "360_day" and end_src.day == 30 and end.daysinmonth == 31:
        # For the specific case of daily data from 360_day source, the last day is expected to be "missing"
        end = end.replace(day=31)

    return xr.DataArray(
        date_range(
            _convert_datetime(index[0], calendar=calendar),
            end,
            freq=freq,
            calendar=calendar,
        ),
        dims=source.dims,
        name=source.dims[0],
    )


def _convert_datetime(
    datetime: Union[pydt.datetime, cftime.datetime],
    new_doy: Optional[Union[float, int]] = None,
    calendar: str = "default",
) -> Union[cftime.datetime, pydt.datetime, float]:
    """Convert a datetime object to another calendar.

    Nanosecond information are lost as cftime.datetime doesn't support them.

    Parameters
    ----------
    datetime: Union[datetime.datetime, cftime.datetime]
      A datetime object to convert.
    new_doy:  Optional[Union[float, int]]
      Allows for redefining the day of year (thus ignoring month and day information from the source datetime).
      -1 is understood as a nan.
    calendar: str
      The target calendar

    Returns
    -------
    Union[cftime.datetime, datetime.datetime, np.nan]
      A datetime object of the target calendar with the same year, month, day and time
      as the source (month and day according to `new_doy` if given).
      If the month and day doesn't exist in the target calendar, returns np.nan. (Ex. 02-29 in "noleap")
    """
    if new_doy in [np.nan, -1]:
        return np.nan
    if new_doy is not None:
        new_date = cftime.num2date(
            new_doy - 1,
            f"days since {datetime.year}-01-01",
            calendar=calendar if calendar != "default" else "standard",
        )
    else:
        new_date = datetime
    try:
        return datetime_classes[calendar](
            datetime.year,
            new_date.month,
            new_date.day,
            datetime.hour,
            datetime.minute,
            datetime.second,
            datetime.microsecond,
        )
    except ValueError:
        return np.nan

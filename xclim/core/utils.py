# -*- coding: utf-8 -*-
# noqa: D205,D400
"""
Miscellaneous indices utilities
===============================

Helper functions for the indices computation, indicator construction and other things.
"""
import logging
import os
import warnings
from collections import defaultdict
from enum import IntEnum
from functools import partial
from importlib import import_module
from importlib.resources import open_text
from inspect import Parameter
from math import floor
from pathlib import Path
from sys import float_info
from types import FunctionType
from typing import Callable, NewType, Optional, Sequence, Union

import numpy as np
import xarray as xr
from boltons.funcutils import update_wrapper
from dask import array as dsk
from xarray import DataArray, Dataset
from yaml import safe_load

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


def calc_percentiles(
    arr: np.array,
    percentiles: Sequence[float] = [50.0],
    axis=-1,
    alpha=1.0,
    beta=1.0,
):
    arr_copy = arr.copy()
    result = np.array(
        [_quantile(arr_copy, per / 100.0, axis, alpha, beta) for per in percentiles]
    )
    return result


# temp
def nan_calc_percentiles(
    arr: np.array,
    percentiles: Sequence[float] = [50.0],
    axis=-1,
    alpha=1.0,
    beta=1.0,
):
    arr_copy = arr.copy()
    result = np.array(
        [_quantile(arr_copy, per / 100.0, axis, alpha, beta) for per in percentiles]
    )
    return result


# Typical values of (alphap,betap) are:
#         - (0,1)    : ``p(k) = k/n`` : linear interpolation of cdf
#           (**R** type 4)
#         - (.5,.5)  : ``p(k) = (k - 1/2.)/n`` : piecewise linear function
#           (**R** type 5)
#         - (0,0)    : ``p(k) = k/(n+1)`` :
#           (**R** type 6)
#         - (1,1)    : ``p(k) = (k-1)/(n-1)``: p(k) = mode[F(x[k])].
#           (**R** type 7, **R** default)
#         - (1/3,1/3): ``p(k) = (k-1/3)/(n+1/3)``: Then p(k) ~ median[F(x[k])].
#           The resulting quantile estimates are approximately median-unbiased
#           regardless of the distribution of x.
#           (**R** type 8)
#         - (3/8,3/8): ``p(k) = (k-3/8)/(n+1/4)``: Blom.
#           The resulting quantile estimates are approximately unbiased
#           if x is normally distributed
#           (**R** type 9)
#         - (.4,.4)  : approximately quantile unbiased (Cunnane)
#         - (.35,.35): APL, used with PWM
def _nan_quantile(arr: np.array, quantile: float, axis=0, alpha=1.0, beta=1.0):
    values_count = arr.shape[axis]
    if values_count == 0:
        return np.NAN
    if values_count == 1:
        return arr[()]
    nan_count = np.isnan(arr).sum(axis).astype(float)
    valid_values_count = values_count - nan_count
    # need at least two values to interpolate
    valid_values_count[valid_values_count < 2] = np.NaN
    # Index where to find the value in the sorted array.
    # Virtual because it is a floating point value not an actual index, the nearest neighbour of it are used for interpolation
    virtual_index = np.where(
        valid_values_count == 0,
        0.0,
        valid_values_count * quantile + (alpha + quantile * (1.0 - alpha - beta)) - 1,
    )
    out_of_bounds_indexes = virtual_index >= valid_values_count - 1
    if out_of_bounds_indexes.sum() > 0:
        virtual_index[out_of_bounds_indexes] = valid_values_count - 1
    previous_index = np.floor(virtual_index)
    gamma = virtual_index - previous_index
    next_index = previous_index + 1
    arr.sort(axis=axis)
    # TODO find out why epsilon * 4 instead of just epsilon (it is done like this in R and in climdex c++ impl)
    fuzz = float_info.epsilon * 4
    previous_index[np.isnan(previous_index)] = -1
    next_index[np.isnan(next_index)] = -1
    previous_element = np.squeeze(
        np.take_along_axis(arr, previous_index.astype(int)[..., np.newaxis], axis=axis)
    )
    next_element = np.squeeze(
        np.take_along_axis(arr, next_index.astype(int)[..., np.newaxis], axis=axis)
    )
    gamma[gamma < fuzz] = 0
    interpolation = gamma * next_element + (1 - gamma) * previous_element
    result = np.where(
        np.isnan(interpolation),
        np.nanmax(arr, axis=axis),
        interpolation,
    )
    return result


def _quantile(arr: np.array, quantile: float, axis=-1, alpha=1.0, beta=1.0):
    #  TODO generalization
    #   - Add doc
    values_count = arr.shape[axis]
    if values_count == 0:
        return np.NAN
    if values_count == 1:
        return arr[()]
    # Virtual because it is a floating point value not an actual index, the nearest neighbour of it are used for interpolation
    virtual_index = (
        values_count * quantile + (alpha + quantile * (1.0 - alpha - beta)) - 1
    )
    if virtual_index >= values_count - 1:
        # When virtual_index is out of bounds we are looking for the array maximum (e.g when quantile == 1)
        # This avoid sorting the array for nothing
        return np.max(arr, axis=axis)
    previous_index = int(floor(virtual_index))
    gamma = virtual_index - previous_index
    next_index = previous_index + 1
    arr.partition([previous_index, next_index, -1], axis=axis)
    slices_having_nans = np.isnan(np.take(arr, -1, axis=axis))
    # TODO find out why epsilon * 4 instead of epsilon (it is done like this in R and in climdex c++ impl)
    fuzz = float_info.epsilon * 4
    previous_element = np.take(arr, previous_index, axis=axis)
    next_element = np.take(arr, next_index, axis=axis)
    if gamma < fuzz:
        result = previous_element
    else:
        result = gamma * next_element + (1 - gamma) * previous_element
    if np.any(slices_having_nans):
        # Mask the slices where a NaN was detected
        result[..., slices_having_nans] = np.nan
    return result


def REF_quantile(arr: np.array, quantile: float, axis=-1, alpha=1.0, beta=1.0):
    #  TODO generalization
    #   - Add doc
    FIRST_AXIS = 0
    values_count = arr.shape[axis]
    if values_count == 0:
        return np.NAN
    if values_count == 1:
        return arr[()]
    nan_count = np.isnan(arr).sum()  # FIXME bottleneck ?
    has_nans = nan_count > 0
    arr = np.moveaxis(arr, axis, FIRST_AXIS)
    valid_values_count = values_count - nan_count
    if valid_values_count == 0:
        return np.NAN
    # Index where to find the value in the sorted array.
    # Virtual because it is a floating point value not an actual index, the nearest neighbour of it are used for interpolation
    virtual_index = (
        valid_values_count * quantile + (alpha + quantile * (1.0 - alpha - beta)) - 1
    )
    if virtual_index >= valid_values_count - 1:
        # When virtual_index is out of bounds we are looking for the array maximum (e.g when quantile == 1)
        # This avoid sorting the array for nothing
        if has_nans:
            return np.nanmax(arr, axis=axis)
        else:
            return np.max(arr, axis=axis)
    previous_index = int(floor(virtual_index))
    gamma = virtual_index - previous_index
    next_index = previous_index + 1
    # MAX_VALUE_COUNT_FOR_FAST_SORT = 100
    # if values_count <= MAX_VALUE_COUNT_FOR_FAST_SORT or has_nans:
    #     # if the array has NaNs, ::sort will push them at the end of the sorted array which is very convenient
    #     arr.sort(axis=FIRST_AXIS)
    # else:
    #     # Partition is faster only when len(arr) > ~100
    arr.partition([previous_index, next_index], axis=FIRST_AXIS)
    # TODO find out why epsilon * 4 instead of epsilon (it is done like this in R and in climdex c++ impl)
    fuzz = float_info.epsilon * 4
    previous_element = arr[previous_index]
    next_element = arr[next_index]
    if gamma < fuzz:
        if has_nans:
            return np.where(
                np.isnan(previous_element),
                np.nanmax(arr, axis=FIRST_AXIS),
                previous_element,
            )
        else:
            return previous_element
    else:
        interpolation = gamma * next_element + (1 - gamma) * previous_element
        if has_nans:
            return np.where(
                np.isnan(interpolation),
                np.nanmax(arr, axis=FIRST_AXIS),
                interpolation,
            )
        else:
            return interpolation


def raise_warn_or_log(
    err: Exception, mode: str, msg: Optional[str] = None, stacklevel: int = 1
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
        logging.info(msg)
    elif mode == "warn":
        warnings.warn(msg, stacklevel=stacklevel + 1)
    else:  # mode == "raise"
        raise err


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

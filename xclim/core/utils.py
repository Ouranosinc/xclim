"""
Miscellaneous Indices Utilities
===============================

Helper functions for the indices computations, indicator construction and other things.
"""
from __future__ import annotations

import importlib.util
import logging
import os
import warnings
from collections import defaultdict
from enum import IntEnum
from functools import partial
from importlib.resources import open_text
from inspect import Parameter, _empty  # noqa
from io import StringIO
from pathlib import Path
from typing import Callable, Mapping, NewType, Sequence, TypeVar

import numpy as np
import xarray as xr
from boltons.funcutils import update_wrapper
from dask import array as dsk
from pint import Quantity
from yaml import safe_dump, safe_load

logger = logging.getLogger("xclim")

#: Type annotation for strings representing full dates (YYYY-MM-DD), may include time.
DateStr = NewType("DateStr", str)

#: Type annotation for strings representing dates without a year (MM-DD).
DayOfYearStr = NewType("DayOfYearStr", str)

#: Type annotation for thresholds and other not-exactly-a-variable quantities
Quantified = TypeVar("Quantified", xr.DataArray, str, Quantity)

# Official variables definitions
VARIABLES = safe_load(open_text("xclim.data", "variables.yml"))["variables"]

# Input cell methods
ICM = {
    "tasmin": "time: minimum within days",
    "tasmax": "time: maximum within days",
    "tas": "time: mean within days",
    "pr": "time: sum within days",
}


def wrapped_partial(func: Callable, suggested: dict | None = None, **fixed) -> Callable:
    r"""Wrap a function, updating its signature but keeping its docstring.

    Parameters
    ----------
    func : Callable
        The function to be wrapped
    suggested : dict, optional
        Keyword arguments that should have new default values but still appear in the signature.
    \*\*fixed
        Keyword arguments that should be fixed by the wrapped and removed from the signature.

    Returns
    -------
    Callable

    Examples
    --------
    >>> from inspect import signature
    >>> def func(a, b=1, c=1):
    ...     print(a, b, c)
    ...
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
        partial_func, func, injected=list(fixed.keys()), hide_wrapped=True  # noqa
    )

    # Store all injected params,
    injected = getattr(func, "_injected", {}).copy()
    injected.update(fixed)
    fully_wrapped._injected = injected
    return fully_wrapped


# TODO Reconsider the utility of this
def walk_map(d: dict, func: Callable) -> dict:
    """Apply a function recursively to values of dictionary.

    Parameters
    ----------
    d : dict
        Input dictionary, possibly nested.
    func : Callable
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


def load_module(path: os.PathLike, name: str | None = None):
    """Load a python module from a python file, optionally changing its name.

    Examples
    --------
    Given a path to a module file (.py):

    .. code-block:: python

        from pathlib import Path
        import os

        path = Path("path/to/example.py")

    The two following imports are equivalent, the second uses this method.

    .. code-block:: python

        os.chdir(path.parent)
        import example as mod1  # noqa

        os.chdir(previous_working_dir)
        mod2 = load_module(path)
        mod1 == mod2
    """
    path = Path(path)
    spec = importlib.util.spec_from_file_location(name or path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # This executes code, effectively loading the module
    return mod


class ValidationError(ValueError):
    """Error raised when input data to an indicator fails the validation tests."""

    @property
    def msg(self):  # noqa
        return self.args[0]


class MissingVariableError(ValueError):
    """Error raised when a dataset is passed to an indicator but one of the needed variable is missing."""


def ensure_chunk_size(da: xr.DataArray, **minchunks: dict[str, int]) -> xr.DataArray:
    r"""Ensure that the input DataArray has chunks of at least the given size.

    If only one chunk is too small, it is merged with an adjacent chunk.
    If many chunks are too small, they are grouped together by merging adjacent chunks.

    Parameters
    ----------
    da : xr.DataArray
        The input DataArray, with or without the dask backend. Does nothing when passed a non-dask array.
    \*\*minchunks : dict[str, int]
        A kwarg mapping from dimension name to minimum chunk size.
        Pass -1 to force a single chunk along that dimension.

    Returns
    -------
    xr.DataArray
    """
    if not uses_dask(da):
        return da

    all_chunks = dict(zip(da.dims, da.chunks))
    chunking = {}
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


def uses_dask(da: xr.DataArray) -> bool:
    """Evaluate whether dask is installed and array is loaded as a dask array.

    Parameters
    ----------
    da: xr.DataArray

    Returns
    -------
    bool
    """
    if isinstance(da, xr.DataArray) and isinstance(da.data, dsk.Array):
        return True
    if isinstance(da, xr.Dataset) and any(
        isinstance(var.data, dsk.Array) for var in da.variables.values()
    ):
        return True
    return False


def calc_perc(
    arr: np.ndarray,
    percentiles: Sequence[float] = None,
    alpha: float = 1.0,
    beta: float = 1.0,
    copy: bool = True,
) -> np.ndarray:
    """Compute percentiles using nan_calc_percentiles and move the percentiles' axis to the end."""
    if percentiles is None:
        percentiles = [50.0]

    return np.moveaxis(
        nan_calc_percentiles(
            arr=arr, percentiles=percentiles, axis=-1, alpha=alpha, beta=beta, copy=copy
        ),
        source=0,
        destination=-1,
    )


def nan_calc_percentiles(
    arr: np.ndarray,
    percentiles: Sequence[float] = None,
    axis=-1,
    alpha=1.0,
    beta=1.0,
    copy=True,
) -> np.ndarray:
    """Convert the percentiles to quantiles and compute them using _nan_quantile."""
    if percentiles is None:
        percentiles = [50.0]

    if copy:
        # bootstrapping already works on a data's copy
        # doing it again is extremely costly, especially with dask.
        arr = arr.copy()
    quantiles = np.array([per / 100.0 for per in percentiles])
    return _nan_quantile(arr, quantiles, axis, alpha, beta)


def _compute_virtual_index(
    n: np.ndarray, quantiles: np.ndarray, alpha: float, beta: float
):
    """Compute the floating point indexes of an array for the linear interpolation of quantiles.

    Based on the approach used by :cite:t:`hyndman_sample_1996`.

    Parameters
    ----------
    n : array_like
        The sample sizes.
    quantiles : array_like
        The quantiles values.
    alpha : float
        A constant used to correct the index computed.
    beta : float
        A constant used to correct the index computed.

    Notes
    -----
    `alpha` and `beta` values depend on the chosen method (see quantile documentation).

    References
    ----------
    :cite:cts:`hyndman_sample_1996`
    """
    return n * quantiles + (alpha + quantiles * (1 - alpha - beta)) - 1


def _get_gamma(virtual_indexes: np.ndarray, previous_indexes: np.ndarray):
    """Compute gamma (AKA 'm' or 'weight') for the linear interpolation of quantiles.

    Parameters
    ----------
    virtual_indexes: array_like
      The indexes where the percentile is supposed to be found in the sorted sample.
    previous_indexes: array_like
      The floor values of virtual_indexes.

    Notes
    -----
    `gamma` is usually the fractional part of virtual_indexes but can be modified by the interpolation method.
    """
    gamma = np.asanyarray(virtual_indexes - previous_indexes)
    return np.asanyarray(gamma)


def _get_indexes(
    arr: np.ndarray, virtual_indexes: np.ndarray, valid_values_count: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Get the valid indexes of arr neighbouring virtual_indexes.

    Notes
    -----
    This is a companion function to linear interpolation of quantiles.

    Parameters
    ----------
    arr : array-like
    virtual_indexes : array-like
    valid_values_count : array-like

    Returns
    -------
    array-like, array-like
        A tuple of virtual_indexes neighbouring indexes (previous and next)
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
    """Compute the linear interpolation weighted by gamma on each point of two same shape arrays.

    Parameters
    ----------
    left : array_like
        Left bound.
    right : array_like
        Right bound.
    gamma : array_like
        The interpolation weight.

    Returns
    -------
    array_like
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
) -> float | np.ndarray:
    """Get the quantiles of the array for the given axis.

    A linear interpolation is performed using alpha and beta.

    Notes
    -----
    By default, alpha == beta == 1 which performs the 7th method of :cite:t:`hyndman_sample_1996`.
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
        # This will result in getting the only available value if it exists
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
    msg: str | None = None,
    err_type: type = ValueError,
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
    err_type : type
        The type of error/exception to raise.
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
    should use in order to be picked up by the indicator constructor. Notice that we are using the annotation format
    as described in PEP604/py3.10, i.e. with | indicating an union and without import objects from `typing`.
    """

    VARIABLE = 0
    """A data variable (DataArray or variable name).

       Annotation : ``xr.DataArray``.
    """
    OPTIONAL_VARIABLE = 1
    """An optional data variable (DataArray or variable name).

       Annotation : ``xr.DataArray | None``. The default should be None.
    """
    QUANTIFIED = 2
    """A quantity with units, either as a string (scalar), a pint.Quantity (scalar) or a DataArray (with units set).

       Annotation : ``xclim.core.utils.Quantified`` and an entry in the :py:func:`xclim.core.units.declare_units` decorator.
       "Quantified" translates to ``str | xr.DataArray | pint.util.Quantity``.
    """
    FREQ_STR = 3
    """A string representing an "offset alias", as defined by pandas.

       See https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases .
       Annotation : ``str`` + ``freq`` as the parameter name.
    """
    NUMBER = 4
    """A number.

       Annotation : ``int``, ``float`` and unions thereof, potentially optional.
    """
    STRING = 5
    """A simple string.

       Annotation : ``str`` or ``str | None``. In most cases, this kind of parameter makes sense with choices indicated
       in the docstring's version of the annotation with curly braces. See :ref:`notebooks/extendxclim:Defining new indices`.
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

       Annotation : ``Sequence[int]``, ``Sequence[float]`` and unions thereof,
       may include single ``int`` and ``float``, may be optional.
    """
    BOOL = 9
    """A boolean flag.

       Annotation : ``bool``, may be optional.
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


def infer_kind_from_parameter(param, has_units: bool = False) -> InputKind:
    """Return the appropriate InputKind constant from an ``inspect.Parameter`` object.

    Parameters
    ----------
    param : Parameter
    has_units : bool

    Notes
    -----
    The correspondence between parameters and kinds is documented in :py:class:`xclim.core.utils.InputKind`.
    The only information not inferable through the `inspect` object is whether the parameter
    has been assigned units through the :py:func:`xclim.core.units.declare_units` decorator.
    That can be given with the ``has_units`` flag.
    """
    if param.annotation is not _empty:
        annot = set(
            param.annotation.replace("xarray.", "").replace("xr.", "").split(" | ")
        )
    else:
        annot = {"no_annotation"}

    if "DataArray" in annot and "None" not in annot and param.default is not None:
        return InputKind.VARIABLE

    annot = annot - {"None"}

    if annot.issubset({"DataArray", "str"}) and has_units:
        return InputKind.OPTIONAL_VARIABLE

    if param.name == "freq":
        return InputKind.FREQ_STR

    if annot == {"Quantified"} and has_units:
        return InputKind.QUANTIFIED

    if annot.issubset({"int", "float"}):
        return InputKind.NUMBER

    if annot.issubset({"int", "float", "Sequence[int]", "Sequence[float]"}):
        return InputKind.NUMBER_SEQUENCE

    if annot == {"str"}:
        return InputKind.STRING

    if annot == {"DayOfYearStr"}:
        return InputKind.DAY_OF_YEAR

    if annot == {"DateStr"}:
        return InputKind.DATE

    if annot == {"bool"}:
        return InputKind.BOOL

    if annot == {"Dataset"}:
        return InputKind.DATASET

    if param.kind == param.VAR_KEYWORD:
        return InputKind.KWARGS

    return InputKind.OTHER_PARAMETER


def adapt_clix_meta_yaml(raw: os.PathLike | StringIO | str, adapted: os.PathLike):
    """Read in a clix-meta yaml representation and refactor it to fit xclim's yaml specifications."""
    from ..indices import generic  # pylint: disable=import-outside-toplevel

    # freq_names = {"annual": "A", "seasonal": "Q", "monthly": "M", "weekly": "W"}
    freq_defs = {"annual": "YS", "seasonal": "QS-DEC", "monthly": "MS", "weekly": "W"}

    if isinstance(raw, os.PathLike):
        with open(raw) as f:
            yml = safe_load(f)
    else:
        yml = safe_load(raw)

    yml["realm"] = "atmos"
    yml[
        "doc"
    ] = """  ===================
  CF Standard indices
  ===================

  Indicators found here are defined by the `clix-meta project`_. Adapted documentation from that repository follows:

  The repository aims to provide a platform for thinking about, and developing,
  a unified view of metadata elements required to describe climate indices (aka climate indicators).

  To facilitate data exchange and dissemination the metadata should, as far as possible,
  follow the Climate and Forecasting (CF) Conventions. Considering the very rich and diverse flora of
  climate indices this is however not always possible. By collecting a wide range of different indices
  it is easier to discover any common patterns and features that are currently not well covered by the
  CF Conventions. Currently identified issues frequently relate to standard_name or/and cell_methods
  which both are controlled vocabularies of the CF Conventions.

  .. _clix-meta project: https://github.com/clix-meta/clix-meta
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
                f"Indicator {cmid} has a 'number_of_days' standard name"
                " and xclim disagrees with the CF conventions on the correct output units, removing."
            )
            continue

        if (data["output"].get("standard_name") or "").endswith("precipitation_amount"):
            remove_ids.append(cmid)
            print(
                f"Indicator {cmid} has a 'precipitation_amount' standard name"
                " and clix-meta has incoherent output units, removing."
            )
            continue

        rename_params = {}
        if index_function["parameters"]:
            data["parameters"] = index_function["parameters"]
            for name, param in data["parameters"].copy().items():
                if param["kind"] in ["operator", "reducer"]:
                    # Compatibility with xclim `op` notation for comparison symbols
                    if name == "condition":
                        data["parameters"]["op"] = param[param["kind"]]
                        del data["parameters"][name]
                    else:
                        data["parameters"][name] = param[param["kind"]]
                else:  # kind = quantified
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

        period = data.pop("default_period")
        # data["allowed_periods"] = [freq_names[per] for per in period["allowed"].keys()]
        data.setdefault("parameters", {})["freq"] = {"default": freq_defs[period]}

        attrs = {}
        output = data.pop("output")
        for attr, val in output.items():
            if val is None:
                continue
            if attr == "cell_methods":
                methods = []
                for i, cell_method in enumerate(val):
                    # Construct cell_method string
                    cm = "".join(
                        [f"{dim}: {meth}" for dim, meth in cell_method.items()]
                    )

                    # If cell_method seems to be describing input data, and not the operation, skip.
                    if i == 0:
                        if cm in [ICM.get(v) for v in data["input"].values()]:
                            continue

                    methods.append(cm)

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

    yml["indicators"] = yml.pop("indices")

    with open(adapted, "w") as f:
        safe_dump(yml, f)


def is_percentile_dataarray(source: xr.DataArray) -> bool:
    """Evaluate whether a DataArray is a Percentile.

    A percentile dataarray must have climatology_bounds attributes and either a
    quantile or percentiles coordinate, the window is not mandatory.
    """
    return (
        isinstance(source, xr.DataArray)
        and source.attrs.get("climatology_bounds", None) is not None
        and ("quantile" in source.coords or "percentiles" in source.coords)
    )

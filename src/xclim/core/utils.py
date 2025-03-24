"""
Miscellaneous Indices Utilities
===============================

Helper functions for the indices computations, indicator construction and other things.
"""

from __future__ import annotations

import functools
import importlib.util
import logging
import os
import warnings
from collections.abc import Callable, Sequence
from enum import IntEnum
from inspect import _empty  # noqa
from io import StringIO
from pathlib import Path
from types import ModuleType

import numpy as np
import xarray as xr
from dask import array as dsk
from yaml import safe_dump, safe_load

logger = logging.getLogger("xclim")


# Input cell methods for clix-meta
ICM = {
    "tasmin": "time: minimum within days",
    "tasmax": "time: maximum within days",
    "tas": "time: mean within days",
    "pr": "time: sum within days",
}


def deprecated(from_version: str | None, suggested: str | None = None) -> Callable:
    """
    Mark an index as deprecated and optionally suggest a replacement.

    Parameters
    ----------
    from_version : str, optional
        The version of xclim from which the function is deprecated.
    suggested : str, optional
        The name of the function to use instead.

    Returns
    -------
    Callable
        The decorated function.
    """

    def _decorator(func):
        @functools.wraps(func)
        def _wrapper(*args, **kwargs):
            msg = (
                f"`{func.__name__}` is deprecated"
                f"{f' from version {from_version}' if from_version else ''} "
                "and will be removed in a future version of xclim"
                f"{f'. Use `{suggested}` instead' if suggested else ''}. "
                "Please update your scripts accordingly."
            )
            warnings.warn(
                msg,
                DeprecationWarning,
                stacklevel=3,
            )

            return func(*args, **kwargs)

        return _wrapper

    return _decorator


def load_module(path: os.PathLike, name: str | None = None) -> ModuleType:
    """
    Load a python module from a python file, optionally changing its name.

    Parameters
    ----------
    path : os.PathLike
        The path to the python file.
    name : str, optional
        The name to give to the module.
        If None, the module name will be the stem of the path.

    Returns
    -------
    ModuleType
        The loaded module.

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


def ensure_chunk_size(da: xr.DataArray, **minchunks: int) -> xr.DataArray:
    r"""
    Ensure that the input DataArray has chunks of at least the given size.

    If only one chunk is too small, it is merged with an adjacent chunk.
    If many chunks are too small, they are grouped together by merging adjacent chunks.

    Parameters
    ----------
    da : xr.DataArray
        The input DataArray, with or without the dask backend. Does nothing when passed a non-dask array.
    **minchunks : dict[str, int]
        A kwarg mapping from dimension name to minimum chunk size.
        Pass -1 to force a single chunk along that dimension.

    Returns
    -------
    xr.DataArray
        The input DataArray, possibly rechunked.
    """
    if not uses_dask(da):
        return da

    all_chunks = dict(zip(da.dims, da.chunks, strict=False))
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
            chunking[dim] = tuple(sum(chunks[i : i + fac]) for i in range(0, len(chunks), fac))
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


def uses_dask(*das) -> bool:
    r"""
    Evaluate whether dask is installed and array is loaded as a dask array.

    Parameters
    ----------
    *das : xr.DataArray or xr.Dataset
        DataArrays or Datasets to check.

    Returns
    -------
    bool
        True if any of the passed objects is using dask.
    """

    def _is_dask_array(da):
        if isinstance(da, xr.DataArray):
            return isinstance(da.data, dsk.Array)
        if isinstance(da, xr.Dataset):
            return any(isinstance(var.data, dsk.Array) for var in da.variables.values())
        return False

    return any(_is_dask_array(da) for da in das)


def calc_perc(
    arr: np.ndarray,
    percentiles: Sequence[float] | None = None,
    alpha: float = 1.0,
    beta: float = 1.0,
    copy: bool = True,
) -> np.ndarray:
    """
    Compute percentiles using nan_calc_percentiles and move the percentiles' axis to the end.

    Parameters
    ----------
    arr : array-like
        The input array.
    percentiles : sequence of float, optional
        The percentiles to compute. If None, only the median is computed.
    alpha : float
        A constant used to correct the index computed.
    beta : float
        A constant used to correct the index computed.
    copy : bool
        If True, the input array is copied before computation. Default is True.

    Returns
    -------
    np.ndarray
        The percentiles along the last axis.
    """
    if percentiles is None:
        _percentiles = [50.0]
    else:
        _percentiles = percentiles

    return np.moveaxis(
        nan_calc_percentiles(
            arr=arr,
            percentiles=_percentiles,
            axis=-1,
            alpha=alpha,
            beta=beta,
            copy=copy,
        ),
        source=0,
        destination=-1,
    )


def nan_calc_percentiles(
    arr: np.ndarray,
    percentiles: Sequence[float] | None = None,
    axis: int = -1,
    alpha: float = 1.0,
    beta: float = 1.0,
    copy: bool = True,
) -> np.ndarray:
    """
    Convert the percentiles to quantiles and compute them using _nan_quantile.

    Parameters
    ----------
    arr : array-like
        The input array.
    percentiles : sequence of float, optional
        The percentiles to compute. If None, only the median is computed.
    axis : int
        The axis along which to compute the percentiles.
    alpha : float
        A constant used to correct the index computed.
    beta : float
        A constant used to correct the index computed.
    copy : bool
        If True, the input array is copied before computation. Default is True.

    Returns
    -------
    np.ndarray
        The percentiles along the specified axis.
    """
    if percentiles is None:
        _percentiles = [50.0]
    else:
        _percentiles = percentiles

    if copy:
        # bootstrapping already works on a data's copy
        # doing it again is extremely costly, especially with dask.
        arr = arr.copy()
    quantiles = np.array([per / 100.0 for per in _percentiles])
    return _nan_quantile(arr, quantiles, axis, alpha, beta)


def _compute_virtual_index(n: np.ndarray, quantiles: np.ndarray, alpha: float, beta: float):
    """
    Compute the floating point indexes of an array for the linear interpolation of quantiles.

    Based on the approach used by :cite:t:`hyndman_sample_1996`.

    Parameters
    ----------
    n : array-like
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
    """
    Compute gamma (AKA 'm' or 'weight') for the linear interpolation of quantiles.

    Parameters
    ----------
    virtual_indexes : array-like
        The indexes where the percentile is supposed to be found in the sorted sample.
    previous_indexes : array-like
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
    """
    Get the valid indexes of arr neighbouring virtual_indexes.

    Parameters
    ----------
    arr : array-like
        The input array.
    virtual_indexes : array-like
        The indexes where the percentile is supposed to be found in the sorted sample.
    valid_values_count : array-like
        The number of valid values in the sorted array.

    Returns
    -------
    array-like, array-like
        A tuple of virtual_indexes neighbouring indexes (previous and next).

    Notes
    -----
    This is a companion function to linear interpolation of quantiles.
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
    Compute the linear interpolation weighted by gamma on each point of two same shape arrays.

    Parameters
    ----------
    left : array-like
        Left bound.
    right : array-like
        Right bound.
    gamma : array-like
        The interpolation weight.

    Returns
    -------
    array-like
        The linearly interpolated array.
    """
    diff_b_a = np.subtract(right, left)
    lerp_interpolation = np.asanyarray(np.add(left, diff_b_a * gamma))
    np.subtract(right, diff_b_a * (1 - gamma), out=lerp_interpolation, where=gamma >= 0.5)
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
    """
    Get the quantiles of the array for the given axis.

    A linear interpolation is performed using alpha and beta.

    Notes
    -----
    By default, alpha == beta == 1 which performs the 7th method of :cite:t:`hyndman_sample_1996`.
    With alpha == beta == 1/3 we get the 8th method.
    """
    # --- Setup
    data_axis_length = arr.shape[axis]
    if data_axis_length == 0:
        return np.nan
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
        valid_values_count[too_few_values] = np.nan
    # --- Computation of indexes
    # Add axis for quantiles
    valid_values_count = valid_values_count[..., np.newaxis]
    virtual_indexes = _compute_virtual_index(valid_values_count, quantiles, alpha, beta)
    virtual_indexes = np.asanyarray(virtual_indexes)
    previous_indexes, next_indexes = _get_indexes(arr, virtual_indexes, valid_values_count)
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


class InputKind(IntEnum):
    """
    Constants for input parameter kinds.

    For use by external parses to determine what kind of data the indicator expects.
    On the creation of an indicator, the appropriate constant is stored in
    :py:attr:`xclim.core.indicator.Indicator.parameters`. The integer value is what gets stored in the output
    of :py:meth:`xclim.core.indicator.Indicator.json`.

    For developers : for each constant, the docstring specifies the annotation a parameter of an indice function
    should use in order to be picked up by the indicator constructor. Notice that we are using the annotation format
    as described in `PEP 604 <https://peps.python.org/pep-0604/>`_, i.e. with '|' indicating a union and without import
    objects from `typing`.
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

       Annotation : ``xclim.core.utils.Quantified`` and an entry in the :py:func:`xclim.core.units.declare_units`
       decorator. "Quantified" translates to ``str | xr.DataArray | pint.util.Quantity``.
    """
    FREQ_STR = 3
    """A string representing an "offset alias", as defined by pandas.

       See the Pandas documentation on :ref:`timeseries.offset_aliases` for a list of valid aliases.

       Annotation : ``str`` + ``freq`` as the parameter name.
    """
    NUMBER = 4
    """A number.

       Annotation : ``int``, ``float`` and unions thereof, potentially optional.
    """
    STRING = 5
    """A simple string.

       Annotation : ``str`` or ``str | None``. In most cases, this kind of parameter makes sense
       with choices indicated in the docstring's version of the annotation with curly braces.
       See :ref:`notebooks/extendxclim:Defining new indices`.
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

       Annotation : ``Sequence[int]``, ``Sequence[float]`` and unions thereof, may include single ``int`` and ``float``,
       may be optional.
    """
    BOOL = 9
    """A boolean flag.

       Annotation : ``bool``, may be optional.
    """
    DICT = 10
    """A dictionary.

       Annotation : ``dict`` or ``dict | None``, may be optional.
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


def infer_kind_from_parameter(param) -> InputKind:
    """
    Return the appropriate InputKind constant from an ``inspect.Parameter`` object.

    Parameters
    ----------
    param : Parameter
        An inspect.Parameter instance.

    Returns
    -------
    InputKind
        The appropriate InputKind constant.

    Notes
    -----
    The correspondence between parameters and kinds is documented in :py:class:`xclim.core.utils.InputKind`.
    """
    if param.annotation is not _empty:
        annot = set(param.annotation.replace("xarray.", "").replace("xr.", "").split(" | "))
    else:
        annot = {"no_annotation"}

    if "DataArray" in annot and "None" not in annot and param.default is not None:
        return InputKind.VARIABLE

    annot = annot - {"None"}

    if "DataArray" in annot:
        return InputKind.OPTIONAL_VARIABLE

    if param.name == "freq":
        return InputKind.FREQ_STR

    if param.kind == param.VAR_KEYWORD:
        return InputKind.KWARGS

    if annot == {"Quantified"}:
        return InputKind.QUANTIFIED

    if "DayOfYearStr" in annot:
        return InputKind.DAY_OF_YEAR

    if annot.issubset({"int", "float"}):
        return InputKind.NUMBER

    if annot.issubset({"int", "float", "Sequence[int]", "Sequence[float]"}):
        return InputKind.NUMBER_SEQUENCE

    if annot.issuperset({"str"}):
        return InputKind.STRING

    if annot == {"DateStr"}:
        return InputKind.DATE

    if annot == {"bool"}:
        return InputKind.BOOL

    if annot == {"dict"}:
        return InputKind.DICT

    if annot == {"Dataset"}:
        return InputKind.DATASET

    return InputKind.OTHER_PARAMETER


# FIXME: Should we be using logging instead of print?
def adapt_clix_meta_yaml(  # noqa: C901
    raw: os.PathLike | StringIO | str, adapted: os.PathLike
) -> None:
    """
    Read in a clix-meta yaml representation and refactor it to fit xclim YAML specifications.

    Parameters
    ----------
    raw : os.PathLike or StringIO or str
        The path to the clix-meta yaml file or the string representation of the yaml.
    adapted : os.PathLike
        The path to the adapted yaml file.
    """
    from ..indices import generic  # pylint: disable=import-outside-toplevel

    freq_defs = {"annual": "YS", "seasonal": "QS-DEC", "monthly": "MS", "weekly": "W"}

    if isinstance(raw, os.PathLike):
        with open(raw, encoding="utf-8") as f:
            yml = safe_load(f)
    else:
        yml = safe_load(raw)

    yml["realm"] = "atmos"
    yml["doc"] = """  ===================
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
            print(f"Indicator {cmid} uses non-implemented function {data['compute']}, removing.")
            continue

        if (data["output"].get("standard_name") or "").startswith("number_of_days") or cmid == "nzero":
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
                                param.get("proposed_standard_name", param.get("standard_name")).replace("_", " "),
                            ),
                            "units": param["units"],
                        }
                        rename_params[f"{{{name}}}"] = f"{{{list(param['data'].keys())[0]}}}"
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
                    cm = "".join([f"{dim}: {meth}" for dim, meth in cell_method.items()])

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

    with open(adapted, "w", encoding="utf-8") as f:
        safe_dump(yml, f)


def is_percentile_dataarray(source: xr.DataArray) -> bool:
    """
    Evaluate whether a DataArray is a Percentile.

    A percentile DataArray must have 'climatology_bounds' attributes and either a
    quantile or percentiles coordinate, the window is not mandatory.

    Parameters
    ----------
    source : xr.DataArray
        The DataArray to evaluate.

    Returns
    -------
    bool
        True if the DataArray is a percentile.
    """
    return (
        isinstance(source, xr.DataArray)
        and source.attrs.get("climatology_bounds", None) is not None
        and ("quantile" in source.coords or "percentiles" in source.coords)
    )


def _chunk_like(*inputs, chunks: dict[str, int] | None):  # *inputs : xr.DataArray | xr.Dataset
    """
    Helper function that (re-)chunks inputs according to a single chunking dictionary.

    Will also ensure passed inputs are not IndexVariable types, so that they can be chunked.
    """
    if not chunks:
        return tuple(inputs)

    outputs = []
    for da in inputs:
        if isinstance(da, xr.DataArray) and isinstance(da.variable, xr.IndexVariable):
            da = xr.DataArray(da, dims=da.dims, coords=da.coords, name=da.name)
        if not isinstance(da, xr.DataArray | xr.Dataset):
            outputs.append(da)
        else:
            outputs.append(da.chunk(**{d: c for d, c in chunks.items() if d in da.dims}))
    return tuple(outputs)


def split_auxiliary_coordinates(
    obj: xr.DataArray | xr.Dataset,
) -> tuple[xr.DataArray | xr.Dataset, xr.Dataset]:
    """
    Split auxiliary coords from the dataset.

    An auxiliary coordinate is a coordinate variable that does not define a dimension and thus
    is not necessarily needed for dataset alignment. Any coordinate that has a name different from
    its dimension(s) is flagged as auxiliary. All scalar coordinates are flagged as auxiliary.

    Parameters
    ----------
    obj : xr.DataArray or xr.Dataset
        An xarray object.

    Returns
    -------
    clean_obj : xr.DataArray or xr.Dataset
        Same as `obj` but without any auxiliary coordinate.
    aux_crd_ds : xr.Dataset
        The auxiliary coordinates as a dataset. Might be empty.

    Notes
    -----
    This is useful to circumvent xarray's alignment checks that will sometimes look the auxiliary coordinate's data,
    which can trigger unwanted dask computations.

    The auxiliary coordinates can be merged back with the dataset with
    :py:meth:`xarray.Dataset.assign_coords` or :py:meth:`xarray.DataArray.assign_coords`.

    >>> # xdoctest: +SKIP
    >>> clean, aux = split_auxiliary_coordinates(ds)
    >>> merged = clean.assign_coords(da.coords)
    >>> merged.identical(ds)  # True
    """
    aux_crd_names = [nm for nm, crd in obj.coords.items() if len(crd.dims) != 1 or crd.dims[0] != nm]
    aux_crd_ds = obj.coords.to_dataset()[aux_crd_names]
    clean_obj = obj.drop_vars(aux_crd_names)
    return clean_obj, aux_crd_ds


# Copied from xarray
def get_temp_dimname(dims: Sequence[str], new_dim: str) -> str:
    """
    Get an new dimension name based on new_dim, that is not used in dims.

    Parameters
    ----------
    dims : sequence of str
        The dimension names that already exist.
    new_dim : str
        The new name we want.

    Returns
    -------
    str
        The new dimension name with as many underscores prepended as necessary to make it unique.
    """
    while new_dim in dims:
        new_dim = "_" + str(new_dim)
    return new_dim

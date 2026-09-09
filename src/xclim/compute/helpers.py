"""
Helper Functions Submodule
==========================

Functions that encapsulate logic and can be shared by many compute functions,
but are not particularly index-like themselves (those should go in the :py:mod:`xclim.compute.generic` module).
"""

from __future__ import annotations

import operator
import warnings
from collections.abc import Callable, Sequence
from datetime import timedelta
from inspect import stack
from typing import Literal, TypeVar

import numpy as np
import pandas as pd
import xarray as xr
from packaging.version import Version
from xarray import __version__ as __xr_version__

from xclim.compute import run_length as rl
from xclim.core import Condition, Freq, Reducer
from xclim.core.options import MAP_BLOCKS, OPTIONS
from xclim.core.units import convert_units_to
from xclim.core.utils import sel_with_nans, uses_dask

if Version(__xr_version__) >= Version("24.9.0"):
    XR2409 = True
else:
    XR2409 = False

try:
    from flox.xarray import rechunk_for_blockwise

    flox_err = None
except ImportError:
    rechunk_for_blockwise = None


__all__ = ["accumulate_between_times", "interpolate_to_time", "resample_map", "wind_speed_height_conversion"]


def _get_dt(freq: str):
    """
    Get the time delta, in seconds for a given pandas frequency. Only valid for freq <= 'D'.

    Parameters
    ----------
    freq : str
        Pandas time frequency.

    Returns
    -------
    float
        Total seconds between two timestamps with this frequency.
    """
    return pd.date_range(freq=freq, periods=2, start="2000-01-01").diff()[1].total_seconds()


def accumulate_between_times(
    da: xr.DataArray, prev_time: xr.DataArray, curr_time: xr.DataArray, freq: str | None = None
) -> xr.DataArray:
    """
    Accumulate (sum) between the given time DataArray (usually solar noon yesterday and solar noon today).

    Parameters
    ----------
    da : xr.DataArray
        DataArray with variable `var` to accumulate.
    prev_time : xr.DataArray
        Time occurrence of the previous event (indexed at the current time).
    curr_time : xr.DataArray
        Time occurrence of the current event (indexed at the current time).
    freq : str | None
        Pandas frequency for ds.time. Defaults to xr.infer_freq(da).

    Returns
    -------
    xr.DataArray
        Variable accumulated between prev_time and curr_time.
    """
    if freq is None:
        freq = xr.infer_freq(da)
    dt = _get_dt(freq)
    da_cum = da.cumsum("time")

    curr_fl = curr_time.dt.floor(freq)
    curr_ratio = (curr_time - curr_fl).dt.total_seconds() / dt

    prev_fl = prev_time.dt.floor(freq)
    prev_ratio = (prev_time - prev_fl).dt.total_seconds() / dt

    d_tilcurr = sel_with_nans(da_cum, "time", curr_fl - pd.Timedelta(dt, "s"))
    d_curr = sel_with_nans(da, "time", curr_fl)
    d_tilprev = sel_with_nans(da_cum, "time", prev_fl - pd.Timedelta(dt, "s"))
    d_prev = sel_with_nans(da, "time", prev_fl)
    da_accum = (d_tilcurr + curr_ratio * d_curr) - (d_tilprev + prev_ratio * d_prev)

    return da_accum


def interpolate_to_time(da: xr.DataArray, curr_time: xr.DataArray, freq: str | None = None) -> xr.DataArray:
    """
    Interpolate Dataset to the given time DataArray (such as Solar noon times).

    This is equivalent to ds[var].interp(time=curr_time), but tends to be faster.

    Parameters
    ----------
    da : xr.DataArray
        DataArray to interpolate, with dimension time and variable `var`.
    curr_time : xr.DataArray
        Time array to interpolate.
    freq : str or None
        Pandas frequency for ds.time. Defaults to xr.infer_freq(ds).

    Returns
    -------
    xr.DataArray
        DataArray of interpolated times.
    """
    if freq is None:
        freq = xr.infer_freq(da)
    dt = _get_dt(freq)

    curr_time_fl = curr_time.dt.floor(freq)
    curr_time_cl = curr_time.dt.ceil(freq)

    curr_ratio = (curr_time - curr_time_fl).dt.total_seconds() / dt

    d_curr_fl = sel_with_nans(da, "time", curr_time_fl)
    d_curr_cl = sel_with_nans(da, "time", curr_time_cl)

    da_interp = (1 - curr_ratio) * d_curr_fl + curr_ratio * d_curr_cl
    return da_interp


DataType = TypeVar("DataType", xr.DataArray, xr.Dataset)

BINARY_OPS = {">": "gt", "<": "lt", ">=": "ge", "<=": "le", "==": "eq", "!=": "ne"}
"""Known binary operators and their translation between symbolic and letter forms."""


def get_binary_op(condition: Condition, constrain: Sequence[Condition] | None = None) -> Callable:
    """
    Get the Python comparison function according to its name or representation and validate allowed usage.

    Accepted condition strings are keys and values of :py:data:`BINARY_OPS`.

    Parameters
    ----------
    condition : {">", "gt", "<", "lt", ">=", "ge", "<=", "le", "==", "eq", "!=", "ne"}
        Comparison binary operator name of symbol.
    constrain : sequence of {">", "gt", "<", "lt", ">=", "ge", "<=", "le", "==", "eq", "!=", "ne"}, optional
        A tuple of allowed operators. Or None to allow all known operators.

    Returns
    -------
    Callable
        A binary function to perform the comparison.
    """
    if condition == "gteq":
        warnings.warn(f"`{condition}` is being renamed `ge` for compatibility.")
        condition = "ge"
    if condition == "lteq":
        warnings.warn(f"`{condition}` is being renamed `le` for compatibility.")
        condition = "le"

    if condition in BINARY_OPS:
        binary_op = BINARY_OPS[condition]
    elif condition in BINARY_OPS.values():
        binary_op = condition
    else:
        raise ValueError(f"Operation `{condition}` not recognized.")

    constraints = []
    if isinstance(constrain, list | tuple | set):
        constraints.extend([BINARY_OPS[c] for c in constrain])
        constraints.extend(constrain)
    elif isinstance(constrain, str):
        constraints.extend([BINARY_OPS[constrain], constrain])

    if constrain:
        if condition not in constraints:
            raise ValueError(f"Operation `{condition}` not permitted for this indicator.")

    return getattr(operator, f"__{binary_op}__")


def compare(
    left: xr.DataArray,
    condition: Condition,
    right: float | int | np.ndarray | xr.DataArray,
    constrain: Sequence[Condition] | None = None,
) -> xr.DataArray:
    """
    Compare a DataArray to a threshold using given operator.

    Comparison is done as ``left condition right``.

    Parameters
    ----------
    left : xr.DataArray
        A DataArray being evaluated against `right`.
    condition : {">", "gt", "<", "lt", ">=", "ge", "<=", "le", "==", "eq", "!=", "ne"}
        Logical comparison operator.
    right : float, int, np.ndarray, or xr.DataArray
        A value or array-like being evaluated against left`.
    constrain : sequence of str, optional
        Optionally allowed conditions.

    Returns
    -------
    xr.DataArray
        Boolean mask of the comparison.
    """
    return get_binary_op(condition, constrain)(left, right)


def spell_mask(
    data: xr.DataArray | Sequence[xr.DataArray],
    window: int,
    window_statistic: Reducer,
    condition: Condition,
    thresh: float | Sequence[float] | xr.DataArray | Sequence[xr.DataArray],
    constrain: Sequence[Condition] | None = None,
    min_gap: int = 1,
    weights: Sequence[float] | None = None,
    var_reducer: Literal["any", "all"] = "all",
) -> xr.DataArray:
    """
    Compute the boolean mask of data points that are part of a spell as defined by a rolling statistic.

    A timestep is part of a spell (True in the mask) if it is contained in any period that fulfills the condition.

    Parameters
    ----------
    data : DataArray or sequence of DataArray
        The input data. Can be a list, in which case the condition is checked on all variables.
        See var_reducer for the latter case.
    window : int
        The length of the rolling window in which to compute statistics.
    window_statistic : {'min', 'max', 'sum', 'mean', 'std', 'var'}
        The statistics to compute on the rolling window.
    condition : {">", "gt", "<", "lt", ">=", "ge", "<=", "le", "==", "eq", "!=", "ne"}
        The comparison operator to use when finding spells. Comparison is done as ``rolled_data {condition} thresh``.
    thresh : float or sequence of floats or DataArray or sequence of DataArray
        The threshold(s) to compare the rolling statistics against.
        If data is a list, this must be a list of the same length as ``data``,
        with a threshold for each variable. This function does not handle units and can't accept Quantified objects.
    constrain : sequence of str, optional
        Optionally allowed conditions.
    min_gap : int
        The shortest possible gap between two spells.
        Spells closer than this are merged by assigning the gap steps to the merged spell.
    weights : sequence of floats, optional
        A list of weights of the same length as the window.
        Only supported if ``window_statistic`` is ``"mean"``.
    var_reducer : {'all', 'any'}
        If the data is a list, the condition must either be fulfilled on *all*
        or *any* variables for the period to be considered a spell.

    Returns
    -------
    xr.DataArray
        Same shape as ``data``, but boolean.
        If ``data`` was a list, this is a DataArray of the same shape as the alignment of all variables.
    """
    _singlevar = True
    # Checks
    if not isinstance(data, xr.DataArray):
        # thus a sequence
        if np.isscalar(thresh) or isinstance(thresh, xr.DataArray) or len(data) != len(thresh):
            raise ValueError("When ``data`` is given as a list, ``thresh`` must be a sequence of the same length.")
        data = xr.concat(data, "variable")
        if isinstance(thresh[0], xr.DataArray):
            thresh = xr.concat(thresh, "variable")
        else:
            thresh = xr.DataArray(thresh, dims=("variable",))
        _singlevar = False

    if weights is not None:
        if window_statistic != "mean":
            raise ValueError(
                f"Argument 'weights' is only supported if 'window_statistic' is 'mean'. Got :  {window_statistic}"
            )
        if len(weights) != window:
            raise ValueError(f"Weights have a different length ({len(weights)}) than the window ({window}).")
        weights = xr.DataArray(weights, dims=("window",))

    if window == 1:  # Fast path
        is_in_spell = compare(data, condition, thresh, constrain=constrain)
        if not _singlevar:
            is_in_spell = getattr(is_in_spell, var_reducer)("variable")
    elif (window_statistic == "min" and condition in [">", ">=", "ge", "gt"]) or (
        window_statistic == "max" and condition in ["`<", "<=", "le", "lt"]
    ):
        # Fast path for specific cases, this yields a smaller dask graph (rolling twice is expensive!)
        # For these two cases, a day can't be part of a spell if it doesn't respect the condition itself
        mask = compare(data, condition, thresh, constrain=constrain)
        if not _singlevar:
            mask = getattr(mask, var_reducer)("variable")
        # We need to filter out the spells shorter than "window"
        # find sequences of consecutive respected constraints
        cs_s = rl._cumsum_reset(mask)
        # end of these sequences
        cs_s = cs_s.where(mask.shift({"time": -1}, fill_value=0) == 0)
        # propagate these end of sequences
        # the `.where(mask>0, 0)` acts a stopper
        is_in_spell = cs_s.where(cs_s >= window).where(mask > 0, 0).bfill("time") > 0
    else:
        data_pad = data.pad(time=(0, window))
        # The spell-wise value to test
        # For example, "window_reducer='sum'",
        # we want the sum over the minimum spell length (window) to be above the thresh
        if weights is not None:
            spell_value = data_pad.rolling(time=window).construct("window").dot(weights)
        else:
            spell_value = getattr(data_pad.rolling(time=window), window_statistic)()
        # True at the end of a spell respecting the condition
        mask = compare(spell_value, condition, thresh, constrain=constrain)
        if not _singlevar:
            mask = getattr(mask, var_reducer)("variable")
        # True for all days part of a spell that respected the condition (shift because of the two rollings)
        is_in_spell = (mask.rolling(time=window).sum() >= 1).shift(time=-(window - 1), fill_value=False)
        # Cut back to the original size
        is_in_spell = is_in_spell.isel(time=slice(0, data.time.size))

    if min_gap > 1:
        is_in_spell = rl.runs_with_holes(is_in_spell, 1, ~is_in_spell, min_gap).astype(bool)

    return is_in_spell


def detrend(ds: DataType, dim="time", deg=1) -> DataType:
    """
    Detrend data along a given dimension computing a polynomial trend of a given order.

    Parameters
    ----------
    ds : xr.Dataset or xr.DataArray
      The data to detrend. If a Dataset, detrending is done on all data variables.
    dim : str
      Dimension along which to compute the trend.
    deg : int
      Degree of the polynomial to fit.

    Returns
    -------
    xr.Dataset or xr.DataArray
      Same as `ds`, but with its trend removed (subtracted).
    """
    if isinstance(ds, xr.Dataset):
        return ds.map(detrend, keep_attrs=False, dim=dim, deg=deg)
    # is a DataArray
    # detrend along a single dimension
    coeff = ds.polyfit(dim=dim, deg=deg)
    trend = xr.polyval(ds[dim], coeff.polyfit_coefficients)
    with xr.set_options(keep_attrs=True):
        return ds - trend


def _wrap_radians(da):
    with xr.set_options(keep_attrs=True):
        return ((da + np.pi) % (2 * np.pi)) - np.pi


def wind_speed_height_conversion(
    ua: xr.DataArray,
    h_source: str,
    h_target: str,
    method: Literal["log"] = "log",
) -> xr.DataArray:
    r"""
    Wind speed at two meters.

    Parameters
    ----------
    ua : xarray.DataArray
        Wind speed at height `h`.
    h_source : str
        Height of the input wind speed `ua` (e.g. `h == "10 m"` for a wind speed at `10 meters`).
    h_target : str
        Height of the output wind speed.
    method : {"log"}
        Method used to convert wind speed from one height to another.

    Returns
    -------
    xarray.DataArray
        Wind speed at height `h_target`.

    References
    ----------
    :cite:cts:`allen_crop_1998`
    """
    h_source_m = convert_units_to(h_source, "m")
    h_target_m = convert_units_to(h_target, "m")
    if method == "log":
        if min(h_source_m, h_target_m) < 1 + 5.42 / 67.8:
            raise ValueError(
                f"The height {min(h_source_m, h_target_m)}m is too small for method {method}. "
                f"Heights must be greater than {1 + 5.42 / 67.8}"
            )
        with xr.set_options(keep_attrs=True):
            return ua * np.log(67.8 * h_target_m - 5.42) / np.log(67.8 * h_source_m - 5.42)
    else:
        raise NotImplementedError(f"'{method}' method is not implemented.")


def _gather_lat(da: xr.DataArray) -> xr.DataArray:
    """
    Gather latitude coordinate using cf-xarray.

    Parameters
    ----------
    da : xarray.DataArray
        CF-conformant DataArray with a "latitude" coordinate.

    Returns
    -------
    xarray.DataArray
        Latitude coordinate.
    """
    try:
        lat = da.cf["latitude"]
        return lat
    except KeyError as err:
        n_func = stack()[1].function
        msg = f"{n_func} could not find latitude coordinate in DataArray. Try passing it explicitly (`lat=ds.lat`)."
        raise ValueError(msg) from err


def _gather_lon(da: xr.DataArray) -> xr.DataArray:
    """
    Gather longitude coordinate using cf-xarray.

    Parameters
    ----------
    da : xarray.DataArray
        CF-conformant DataArray with a "longitude" coordinate.

    Returns
    -------
    xarray.DataArray
        Longitude coordinate.
    """
    try:
        lat = da.cf["longitude"]
        return lat
    except KeyError as err:
        n_func = stack()[1].function
        msg = f"{n_func} could not find longitude coordinate in DataArray. Try passing it explicitly (`lon=ds.lon`)."
        raise ValueError(msg) from err


def resample_map(
    obj: DataType,
    dim: str,
    freq: Freq,
    func: Callable,
    map_blocks: bool | Literal["from_context"] = "from_context",
    resample_kwargs: dict | None = None,
    map_kwargs: dict | None = None,
) -> DataType:
    r"""
    Wrap xarray's resample(...).map() with a :py:func:`xarray.map_blocks`.

    Ensures that the chunking is appropriate using `flox`.

    Parameters
    ----------
    obj : DataArray or Dataset
        The xarray object to resample.
    dim : str
        Dimension over which to resample.
    freq : str
        Resampling frequency along `dim`.
    func : callable
        Function to map on each resampled group.
    map_blocks : bool or "from_context"
        If True, the resample().map() call is wrapped inside a `map_blocks`.
        If False, this does not do anything special.
        If "from_context", xclim's "resample_map_blocks" option is used.
        If the object is not using dask, this is set to False.
    resample_kwargs : dict, optional
        Other arguments to pass to `obj.resample()`.
    map_kwargs : dict, optional
        Arguments to pass to `map`.

    Returns
    -------
    xr.DataArray or xr.Dataset
        Resampled object.
    """
    resample_kwargs = resample_kwargs or {}
    map_kwargs = map_kwargs or {}
    if map_blocks == "from_context":
        map_blocks = OPTIONS[MAP_BLOCKS]

    if not uses_dask(obj) or not map_blocks:
        return obj.resample({dim: freq}, **resample_kwargs).map(func, **map_kwargs)

    if rechunk_for_blockwise is None:
        msg = f"Using {MAP_BLOCKS}=True requires flox."
        raise ValueError(msg) from flox_err

    # Make labels, a unique integer for each resample group
    labels = xr.full_like(obj[dim], -1, dtype=np.int32)
    for lbl, group_slice in enumerate(obj[dim].resample({dim: freq}).groups.values()):
        labels[group_slice] = lbl

    obj_rechunked = rechunk_for_blockwise(obj, dim, labels)

    def _resample_map(obj_chnk, dm, frq, rs_kws, fun, mp_kws):
        return obj_chnk.resample({dm: frq}, **rs_kws).map(fun, **mp_kws)

    # Template. We are hoping that this takes a negligible time as it is never loaded.
    template = obj_rechunked.resample(**{dim: freq}, **resample_kwargs).first()

    # New chunks along the time dim : infer the number of elements resulting from the resampling of each chunk
    if isinstance(obj_rechunked, xr.Dataset):
        chunksizes = obj_rechunked.chunks[dim]
    else:
        chunksizes = obj_rechunked.chunks[obj_rechunked.get_axis_num(dim)]
    new_chunks = []
    i = 0
    for chunksize in chunksizes:
        new_chunks.append(len(np.unique(labels[i : i + chunksize])))
        i += chunksize
    template = template.chunk({dim: tuple(new_chunks)})

    return obj_rechunked.map_blocks(_resample_map, (dim, freq, resample_kwargs, func, map_kwargs), template=template)


def _add_one_day(time: xr.DataArray) -> xr.DataArray:
    """
    Add one day to a time coordinate.

    Depending on the calendar/dtype of the time array we need to use numpy's or datetime's (for cftimes) timedelta.

    Parameters
    ----------
    time : xr.DataArray
        Time coordinate.

    Returns
    -------
    xr.DataArray
        Next day.
    """
    if time.dtype == "O":
        return time + timedelta(days=1)
    return time + np.timedelta64(1, "D")

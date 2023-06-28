"""Module comprising the bootstrapping algorithm for indicators."""
from __future__ import annotations

import warnings
from inspect import signature
from typing import Any, Callable

import cftime
import numpy as np
import xarray
from boltons.funcutils import wraps
from xarray.core.dataarray import DataArray

import xclim.core.utils

from .calendar import convert_calendar, parse_offset, percentile_doy

BOOTSTRAP_DIM = "_bootstrap"


def percentile_bootstrap(func):
    """Decorator applying a bootstrap step to the calculation of exceedance over a percentile threshold.

    This feature is experimental.

    Bootstraping avoids discontinuities in the exceedance between the reference period over which percentiles are
    computed, and "out of reference" periods. See `bootstrap_func` for details.

    Declaration example:

    .. code-block:: python

        @declare_units(tas="[temperature]", t90="[temperature]")
        @percentile_bootstrap
        def tg90p(
            tas: xarray.DataArray,
            t90: xarray.DataArray,
            freq: str = "YS",
            bootstrap: bool = False,
        ) -> xarray.DataArray:
            pass

    Examples
    --------
    >>> from xclim.core.calendar import percentile_doy
    >>> from xclim.indices import tg90p
    >>> tas = xr.open_dataset(path_to_tas_file).tas
    >>> # To start bootstrap reference period must not fully overlap the studied period.
    >>> tas_ref = tas.sel(time=slice("1990-01-01", "1992-12-31"))
    >>> t90 = percentile_doy(tas_ref, window=5, per=90)
    >>> tg90p(tas=tas, tas_per=t90.sel(percentiles=90), freq="YS", bootstrap=True)
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        ba = signature(func).bind(*args, **kwargs)
        ba.apply_defaults()
        bootstrap = ba.arguments.get("bootstrap", False)
        if bootstrap is False:
            return func(*args, **kwargs)

        return bootstrap_func(func, **ba.arguments)

    return wrapper


def bootstrap_func(compute_index_func: Callable, **kwargs) -> xarray.DataArray:
    r"""Bootstrap the computation of percentile-based indices.

    Indices measuring exceedance over percentile-based thresholds (such as tx90p) may contain artificial discontinuities
    at the beginning and end of the reference period used to calculate percentiles. The bootstrap procedure can reduce
    those discontinuities by iteratively computing the percentile estimate and the index on altered reference periods.

    Theses altered reference periods are themselves built iteratively: When computing the index for year x, the
    bootstrapping create as many altered reference period as the number of years in the reference period.
    To build one altered reference period, the values of year x are replaced by the values of another year in the
    reference period, then the index is computed on this altered period. This is repeated for each year of the reference
    period, excluding year x, The final result of the index for year x, is then the average of all the index results on
    altered years.

    Parameters
    ----------
    compute_index_func : Callable
        Index function.
    \*\*kwargs
        Arguments to `func`.

    Returns
    -------
    xr.DataArray
        The result of func with bootstrapping.

    References
    ----------
    :cite:cts:`zhang_avoiding_2005`

    Notes
    -----
    This function is meant to be used by the `percentile_bootstrap` decorator.
    The parameters of the percentile calculation (percentile, window, reference_period)
    are stored in the attributes of the percentile DataArray.
    The bootstrap algorithm implemented here does the following::

        For each temporal grouping in the calculation of the index
            If the group `g_t` is in the reference period
                For every other group `g_s` in the reference period
                    Replace group `g_t` by `g_s`
                    Compute percentile on resampled time series
                    Compute index function using percentile
                Average output from index function over all resampled time series
            Else compute index function using original percentile

    """
    # Identify the input and the percentile arrays from the bound arguments
    per_key = None
    for name, val in kwargs.items():
        if isinstance(val, DataArray):
            if "percentile_doy" in val.attrs.get("history", ""):
                per_key = name
            else:
                da_key = name
    # Extract the DataArray inputs from the arguments
    da: DataArray = kwargs.pop(da_key)
    per_da: DataArray | None = kwargs.pop(per_key, None)
    if per_da is None:
        # per may be empty on non doy percentiles
        raise KeyError(
            "`bootstrap` can only be used with percentiles computed using `percentile_doy`"
        )
    # Boundary years of reference period
    clim = per_da.attrs["climatology_bounds"]
    if xclim.core.utils.uses_dask(da) and len(da.chunks[da.get_axis_num("time")]) > 1:
        warnings.warn(
            "The input data is chunked on time dimension and must be fully re-chunked to"
            " run percentile bootstrapping."
            " Beware, this operation can significantly increase the number of tasks dask"
            " has to handle.",
            stacklevel=2,
        )
        chunking = {d: "auto" for d in da.dims}
        chunking["time"] = -1  # no chunking on time to use map_block
        da = da.chunk(chunking)
    # overlap of studied `da` and reference period used to compute percentile
    overlap_da = da.sel(time=slice(*clim))
    if len(overlap_da.time) == len(da.time):
        raise KeyError(
            "`bootstrap` is unnecessary when all years are overlapping between reference "
            "(percentiles period) and studied (index period) periods"
        )
    if len(overlap_da) == 0:
        raise KeyError(
            "`bootstrap` is unnecessary when no year overlap between reference "
            "(percentiles period) and studied (index period) periods."
        )
    pdoy_args = dict(
        window=per_da.attrs["window"],
        alpha=per_da.attrs["alpha"],
        beta=per_da.attrs["beta"],
        per=per_da.percentiles.data[()],
    )
    bfreq = _get_bootstrap_freq(kwargs["freq"])
    # Group input array in years, with an offset matching freq
    overlap_years_groups = overlap_da.resample(time=bfreq).groups
    da_years_groups = da.resample(time=bfreq).groups
    per_template = per_da.copy(deep=True)
    acc = []
    # Compute bootstrapped index on each year of overlapping years
    for year_key, year_slice in da_years_groups.items():
        kw = {da_key: da.isel(time=year_slice), **kwargs}
        if _get_year_label(year_key) in overlap_da.get_index("time").year:
            # If the group year is in both reference and studied periods, run the bootstrap
            bda = build_bootstrap_year_da(overlap_da, overlap_years_groups, year_key)
            if BOOTSTRAP_DIM not in per_template.dims:
                per_template = per_template.expand_dims(
                    {BOOTSTRAP_DIM: np.arange(len(bda._bootstrap))}
                )
                if xclim.core.utils.uses_dask(bda):
                    chunking = {
                        d: bda.chunks[bda.get_axis_num(d)]
                        for d in set(bda.dims).intersection(set(per_template.dims))
                    }
                    per_template = per_template.chunk(chunking)
            per = xarray.map_blocks(
                percentile_doy.__wrapped__,  # strip history update from percentile_doy
                obj=bda,
                kwargs={**pdoy_args, "copy": False},
                template=per_template,
            )
            if "percentiles" not in per_da.dims:
                per = per.squeeze("percentiles")
            kw[per_key] = per
            value = compute_index_func(**kw).mean(dim=BOOTSTRAP_DIM, keep_attrs=True)
        else:
            # Otherwise, run the normal computation using the original percentile
            kw[per_key] = per_da
            value = compute_index_func(**kw)
        acc.append(value)
    result = xarray.concat(acc, dim="time")
    result.attrs["units"] = value.attrs["units"]
    return result


def _get_bootstrap_freq(freq):
    _, base, start_anchor, anchor = parse_offset(freq)  # noqa
    bfreq = "A"
    if start_anchor:
        bfreq += "S"
    if base in ["A", "Q"] and anchor is not None:
        bfreq = f"{bfreq}-{anchor}"
    return bfreq


def _get_year_label(year_dt) -> str:
    if isinstance(year_dt, cftime.datetime):
        year_label = year_dt.year
    else:
        year_label = year_dt.astype("datetime64[Y]").astype(int) + 1970
    return year_label


# TODO: Return a generator instead and assess performance
def build_bootstrap_year_da(
    da: DataArray, groups: dict[Any, slice], label: Any, dim: str = "time"
) -> DataArray:
    """Return an array where a group in the original is replaced by every other groups along a new dimension.

    Parameters
    ----------
    da : DataArray
      Original input array over reference period.
    groups : dict
      Output of grouping functions, such as `DataArrayResample.groups`.
    label : Any
      Key identifying the group item to replace.
    dim : str
      Dimension recognized as time. Default: `time`.

    Returns
    -------
    DataArray:
      Array where one group is replaced by values from every other group along the `bootstrap` dimension.
    """
    gr = groups.copy()
    # Location along dim that must be replaced
    bloc = da[dim][gr.pop(label)]
    # Initialize output array with new bootstrap dimension
    out = da.expand_dims({BOOTSTRAP_DIM: np.arange(len(gr))}).copy(deep=True)
    # With dask, mutating the views of out is not working, thus the accumulator
    out_accumulator = []
    # Replace `bloc` by every other group
    for i, (_, group_slice) in enumerate(gr.items()):
        source = da.isel({dim: group_slice})
        out_view = out.loc[{BOOTSTRAP_DIM: i}]
        if len(source[dim]) < 360 and len(source[dim]) < len(bloc):
            # This happens when the sampling frequency is anchored thus
            # source[dim] would be only a few months on the first and last year
            pass
        elif len(source[dim]) == len(bloc):
            out_view.loc[{dim: bloc}] = source.data
        elif len(bloc) == 365:
            out_view.loc[{dim: bloc}] = convert_calendar(source, "365_day").data
        elif len(bloc) == 366:
            out_view.loc[{dim: bloc}] = convert_calendar(
                source, "366_day", missing=np.NAN
            ).data
        elif len(bloc) < 365:
            # 360 days calendar case or anchored years for both source[dim] and bloc case
            out_view.loc[{dim: bloc}] = source.data[: len(bloc)]
        else:
            raise NotImplementedError
        out_accumulator.append(out_view)
    return xarray.concat(out_accumulator, dim=BOOTSTRAP_DIM)

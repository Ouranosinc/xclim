from inspect import signature
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import cftime
import numpy as np
import pandas as pd
import xarray as xr
from boltons.funcutils import wraps
from xarray.core.dataarray import DataArray

import xclim.core.utils

from .calendar import compare_offsets, convert_calendar, parse_offset, percentile_doy

BOOTSTRAP_DIM = "_bootstrap"


def percentile_bootstrap(func):
    """Decorator applying a bootstrap step to the calculation of exceedance over a percentile threshold.

    This feature is experimental.

    Bootstraping avoids discontinuities in the exceedance between the "in base" period over which percentiles are
    computed, and "out of base" periods. See `bootstrap_func` for details.

    Example of declaration::

    >>> # xdoctest: +SKIP
    >>> @declare_units(tas="[temperature]", t90="[temperature]")
    >>> @percentile_bootstrap
    >>> def tg90p(
    >>>    tas: xarray.DataArray,
    >>>    t90: xarray.DataArray,
    >>>    freq: str = "YS",
    >>>    bootstrap: bool = False
    >>> ) -> xarray.DataArray:

    Examples
    --------
    >>> from xclim.core.calendar import percentile_doy
    >>> from xclim.indices import tg90p
    >>> tas = xr.open_dataset(path_to_tas_file).tas
    >>> t90 = percentile_doy(tas, window=5, per=90)
    >>> tg90p(tas=tas, t90=t90.sel(percentiles=90), freq="YS", bootstrap=True)
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


def bootstrap_func(compute_index_func: Callable, **kwargs) -> xr.DataArray:
    """Bootstrap the computation of percentile-based exceedance indices.

    Indices measuring exceedance over percentile-based threshold may contain artificial discontinuities at the
    beginning and end of the base period used for calculating the percentile. A bootstrap resampling
    procedure can reduce those discontinuities by iteratively replacing each the year the indice is computed on from
    the percentile estimate, and replacing it with another year within the base period.

    Parameters
    ----------
    compute_index_func : Callable
      Indice function.
    kwargs : dict
      Arguments to `func`.

    Returns
    -------
    xr.DataArray
      The result of func with bootstrapping.

    References
    ----------
    Zhang, X., Hegerl, G., Zwiers, F. W., & Kenyon, J. (2005). Avoiding Inhomogeneity in Percentile-Based Indices of
    Temperature Extremes, Journal of Climate, 18(11), 1641-1651, https://doi.org/10.1175/JCLI3366.1

    Notes
    -----
    This function is meant to be used by the `percentile_bootstrap` decorator.
    The parameters of the percentile calculation (percentile, window, base period) are stored in the
    attributes of the percentile DataArray.
    The bootstrap algorithm implemented here does the following::

        For each temporal grouping in the calculation of the indice
            If the group `g_t` is in the base period
                For every other group `g_s` in the base period
                    Replace group `g_t` by `g_s`
                    Compute percentile on resampled time series
                    Compute indice function using percentile
                Average output from indice function over all resampled time series
            Else compute indice function using original percentile

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
    per: Optional[DataArray] = kwargs.pop(per_key, None)
    if per is None:
        # per may be empty on non doy percentiles
        raise KeyError(
            "`bootstrap` can only be used with percentiles computed using `percentile_doy`"
        )

    # List of years in base period
    clim = per.attrs["climatology_bounds"]

    if xclim.core.utils.uses_dask(da):
        chunking = {d: "auto" for d in da.dims}
        chunking["time"] = -1  # no chunking on time to use map_block
        da = da.chunk(chunking)
    # overlap of `da` and base period used to compute percentile
    overlap_da = da.sel(time=slice(*clim))
    # TODO add unit tests for these exceptions
    if len(overlap_da) == len(da):
        raise KeyError(
            "`bootstrap` is unnecessary when all years between in_base (percentiles period) and out_of_base (index period) are overlapping"
        )
    if len(overlap_da) == 0:
        raise KeyError(
            "`bootstrap` is unnecessary when no year overlap between in_base (percentiles period) and out_of_base (index period)."
        )
    # Arguments used to compute percentile
    percentile = per.percentiles.data.tolist()  # Can be a list or scalar
    pdoy_args = dict(
        window=per.attrs["window"],
        alpha=per.attrs["alpha"],
        beta=per.attrs["beta"],
        per=percentile if np.isscalar(percentile) else percentile[0],
    )

    # Group input array in years, with an offset matching freq
    freq = kwargs["freq"]
    _, base, start_anchor, anchor = parse_offset(freq)  # noqa
    bfreq = "A"
    if start_anchor:
        bfreq += "S"
    if base in ["A", "Q"] and anchor is not None:
        bfreq = f"{bfreq}-{anchor}"

    acc = []
    no_bs_dates = da.indexes["time"].difference(overlap_da.indexes["time"])
    kw = {da_key: da.sel(time=no_bs_dates), per_key: per, **kwargs}
    no_bs_result = compute_index_func(**kw)
    no_bs_result.name = "pouet_pouet"

    overlap_years_groups = overlap_da.resample(time=bfreq).groups
    # Copy is not costly per is small
    per_template = per.copy(deep=True).expand_dims(
        _bootstrap=np.arange(len(overlap_years_groups) - 1)
    )
    # Compute bootstrapped index on each year of overlapping years
    bs_acc = []
    for year, time_slice in overlap_years_groups.items():
        kw = {da_key: overlap_da.isel(time=time_slice), **kwargs}
        bda = build_bootstrap_year_da(overlap_da, overlap_years_groups, year)
        if xclim.core.utils.uses_dask(bda):
            chunking = {
                d: bda.chunks[bda.get_axis_num(d)]
                for d in set(bda.dims).intersection(set(per_template.dims))
            }
            per_template = per_template.chunk(chunking)
        kw[per_key] = xr.map_blocks(
            percentile_doy.__wrapped__,  # strip history update from percentile_doy
            obj=bda,
            kwargs={**pdoy_args, "copy": False, "keep_chunk_size": True},
            template=per_template,
        )
        value = compute_index_func(**kw).mean(dim=BOOTSTRAP_DIM, keep_attrs=True)
        value.name = "pouet_pouet"
        bs_acc.append(value)
        acc.append(value)
    bs_result = xr.concat(bs_acc, dim="time")
    # inter_mask = bs_result.sel(time = no_bs_result.get_indexes("time").intesection(bs_result.get_indexes("time")))
    # no_bs_result[inter_mask] = inter_mask
    # todo make sure it's not too slow to use merge
    # no_bs_result --> ~700 tasks
    # bs_result --> ~9400 tasks
    # xr.merge([no_bs_result, bs_result]) --> 781 + 11593 tasks (2 graphs)

    result = xr.merge(
        [no_bs_result, bs_result],
        compat="no_conflicts",  # build pouet_pouet by preserving computed values
        join="outer",  # build dims using indexes' union
        combine_attrs="override",  # Take no_bs_result attributes
    )
    result.pouet_pouet.attrs["units"] = value.attrs["units"]
    return result.pouet_pouet


def get_year(label):
    if isinstance(label, cftime.datetime):
        year = label.year
    else:
        year = label.astype("datetime64[Y]").astype(int) + 1970
    return year


# TODO: Return a generator instead and assess performance
def build_bootstrap_year_da(
    da: DataArray,
    groups: Dict[Any, slice],
    label: Any,
    dim: str = "time",
) -> DataArray:
    """Return an array where a group in the original is replaced by every other groups along a new dimension.

    Parameters
    ----------
    da : DataArray
      Original input array over base period.
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
    for i, (key, group_slice) in enumerate(gr.items()):
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

    return xr.concat(out_accumulator, dim=BOOTSTRAP_DIM)

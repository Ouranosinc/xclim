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


def bootstrap_func(compute_indice_func: Callable, **kwargs) -> xr.DataArray:
    """Bootstrap the computation of percentile-based exceedance indices.

    Indices measuring exceedance over percentile-based threshold may contain artificial discontinuities at the
    beginning and end of the base period used for calculating the percentile. A bootstrap resampling
    procedure can reduce those discontinuities by iteratively replacing each the year the indice is computed on from
    the percentile estimate, and replacing it with another year within the base period.

    Parameters
    ----------
    compute_indice_func : Callable
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

    # `da` over base period used to compute percentile
    overlapping_da = da.sel(time=slice(*clim))

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
    # overlap_da = da.sel(time=slice(str(min(per_clim_years)), str(max(per_clim_years))))
    # for item in da_years.items():
    #     year = get_year(item[0])
    #     if year in per_clim_years:
    #         overlap_years.append(year)
    #     else:
    #         no_overlap_years.append(year)
    # if len(no_overlap_years) == 0:
    #     raise KeyError(
    #         "`bootstrap` is unnecessary when all years between in_base (percentiles period) and out_of_base (index period) are overlapping"
    #     )
    # if len(overlap_years) == 0 :
    #     raise KeyError(
    #         "`bootstrap` is unnecessary when no year overlap between in_base (percentiles period) and out_of_base (index period)."
    #     )
    overlapping_years = overlapping_da.resample(time=bfreq).groups
    out = []

    out_of_base_da = da.sel(
        time=da.indexes["time"].difference(overlapping_da.indexes["time"])
    )
    kw = {da_key: out_of_base_da, **kwargs}
    kw[per_key] = per
    out.append(compute_indice_func(**kw))
    # template = out[0]

    if xclim.core.utils.uses_dask(overlapping_da):
        chunking = {d: "auto" for d in da.dims}
        chunking["time"] = -1  # no chunking on time to use map_block
        overlapping_da = overlapping_da.chunk(chunking)
    #     # TODO, 1. would be better with xr.chunksizes be it needs xarray>=20
    #     # TODO, 2. make sure time is always the first dimension
    #     a = (len(out[0].time),) + overlapping_da.chunks[1:]
    #     template = out[0].chunk(a)

    # def bs_percentile_doys(overlapping_da, overlapping_years, pdoy_args):
    #     out = []
    #     for label, time_slice in overlapping_years.items():
    #         bda = bootstrap_year(overlapping_da, overlapping_years, label)
    #         out.append(percentile_doy(bda, **pdoy_args, copy=False))
    #     return xr.concat(out, dim="time")
    #
    # per_template: DataArray = kwargs[per_key]  # noqa
    # per_template.expand_dims(time=)
    # per_doys = xr.map_blocks(
    #     bs_percentile_doys,
    #     obj=overlapping_da,
    #     kwargs={
    #         "overlapping_years": overlapping_years,
    #         "per_key":       per_key,
    #         "pdoy_args":     pdoy_args,
    #     },
    #     template=per_template,
    # )
    template = per.copy(deep=True)
    # Compute bootstrapped index on each year
    overlapping_da_years = overlapping_da.resample(time=bfreq).groups.items()
    for year_label, _ in overlapping_da_years:
        da_year = da.sel(time=str(get_year(year_label)))
        kw = {da_key: da_year, **kwargs}
        # template.coords["time"] = pd.date_range(
        #     start=da_year.time[0].dt.date.values[()],
        #     periods=len(template["time"]),
        #     freq=freq,
        # )
        bda = build_bootstrap_year_da(
            overlapping_da, overlapping_years, year_label, bs_dim_name="_bootstrap"
        )
        if "_bootstrap" not in template.dims:
            template = template.expand_dims(_bootstrap=len(bda._bootstrap))
            template["_bootstrap"] = bda._bootstrap
            # TODO: fill a bug on gh-xarray to make the indexing automatic ?
            #       (without this reindex, the expect of check_result_variables in parallel.py is not properly computed)
            template = template.reindex(
                {
                    "_bootstrap": pd.RangeIndex(
                        start=0, stop=len(bda._bootstrap), step=1, name="_bootstrap"
                    )
                }
            )
            for d in set(bda.dims).intersection(set(template.dims)):
                template = template.chunk({d: bda.chunks[bda.get_axis_num(d)]})

        # initial_kwargs[per_key] = percentile_doy(bda, **pdoy_args, copy=False)
        kw[per_key] = xr.map_blocks(
            bs_percentile_doy,
            obj=bda,
            kwargs={**pdoy_args, "copy": False},
            template=template,
        )  # noqa
        value = compute_indice_func(**kw).mean(dim="_bootstrap", keep_attrs=True)
        out.append(value)
    out = xr.concat(out, dim="time")
    duplications = out.get_index("time").duplicated()
    if len(duplications) > 0:
        out = out.sel(time=~duplications)
    out.attrs["units"] = value.attrs["units"]
    return out


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
    bs_dim_name: str,
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
    bdim = bs_dim_name
    out = da.expand_dims({bdim: np.arange(len(gr))}).copy(deep=True)
    # With dask, mutating the views of out is not working, thus the accumulator
    out_accumulator = []
    # Replace `bloc` by every other group
    for i, (key, group_slice) in enumerate(gr.items()):
        source = da.isel({dim: group_slice})
        out_view = out.loc[{bdim: i}]
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
            out_view.loc[{dim: bloc}] = source.data[: len(bloc)]
        else:
            raise NotImplementedError
        out_accumulator.append(out_view)

    return xr.concat(out_accumulator, dim=bdim)


# def percentile_doy(
#     arr: xr.DataArray,
#     window: int = 5,
#     per: Union[float, Sequence[float]] = 10.0,
#     alpha: float = 1.0 / 3.0,
#     beta: float = 1.0 / 3.0,
#     copy: bool = True,
# ) -> xr.DataArray:
#     from .utils import calc_perc
#     rr = arr.rolling(min_periods=1, center=True, time=window).construct("window")
#     ind = pd.MultiIndex.from_arrays(
#         (rr.time.dt.year.values, rr.time.dt.dayofyear.values),
#         names=("year", "dayofyear"),
#     )
#     rr = rr.assign_coords(time=ind).unstack("time").stack(stack_dim=("year", "window"))
#     if np.isscalar(per):
#         per = [per]
#     p = xr.apply_ufunc(
#         calc_perc,
#         rr,
#         input_core_dims=[["stack_dim"]],
#         output_core_dims=[["percentiles"]],
#         keep_attrs=True,
#         kwargs=dict(percentiles=per, alpha=alpha, beta=beta, copy=copy),
#         dask="parallelized",
#         output_dtypes=[rr.dtype],
#         dask_gufunc_kwargs=dict(output_sizes={"percentiles": len(per)}),
#     )
#     p = p.assign_coords(percentiles=xr.DataArray(per, dims=("percentiles",)))
#     return p.rename("per")


def bs_percentile_doy(
    arr: xr.DataArray,
    window: int = 5,
    per: Union[float, Sequence[float]] = 10.0,
    alpha: float = 1.0 / 3.0,
    beta: float = 1.0 / 3.0,
    copy: bool = True,
) -> xr.DataArray:
    """Percentile value for each day of the year.

    Return the climatological percentile over a moving window around each day of the year.
    Different quantile estimators can be used by specifying `alpha` and `beta` according to specifications given by [HyndmanFan]_. The default definition corresponds to method 8, which meets multiple desirable statistical properties for sample quantiles. Note that `numpy.percentile` corresponds to method 7, with alpha and beta set to 1.

    Parameters
    ----------
    arr : xr.DataArray
      Input data, a daily frequency (or coarser) is required.
    window : int
      Number of time-steps around each day of the year to include in the calculation.
    per : float or sequence of floats
      Percentile(s) between [0, 100]
    alpha: float
        Plotting position parameter.
    beta: float
        Plotting position parameter.
    copy: bool
        If True (default) the input array will be deep copied. It's a necessary step
        to keep the data integrity but it can be costly.
        If False, no copy is made of the input array. It will be mutated and rendered
        unusable but performances may significantly improve.
        Put this flag to False only if you understand the consequences.

    Returns
    -------
    xr.DataArray
      The percentiles indexed by the day of the year.
      For calendars with 366 days, percentiles of doys 1-365 are interpolated to the 1-366 range.

    References
    ----------
    .. [HyndmanFan] Hyndman, R. J., & Fan, Y. (1996). Sample quantiles in statistical packages. The American Statistician, 50(4), 361-365.
    """
    from .utils import calc_perc

    rr = arr.rolling(min_periods=1, center=True, time=window).construct("window")

    ind = pd.MultiIndex.from_arrays(
        (rr.time.dt.year.values, rr.time.dt.dayofyear.values),
        names=("year", "dayofyear"),
    )
    rrr = rr.assign_coords(time=ind).unstack("time").stack(stack_dim=("year", "window"))

    if rrr.chunks is not None and len(rrr.chunks[rrr.get_axis_num("stack_dim")]) > 1:
        # # Preserve chunk size
        time_chunks_count = len(arr.chunks[arr.get_axis_num("time")])
        doy_chunk_size = np.ceil(len(rrr.dayofyear) / (window * time_chunks_count))
        rrr = rrr.chunk(dict(stack_dim=-1, dayofyear=doy_chunk_size))

    if np.isscalar(per):
        per = [per]

    p = xr.apply_ufunc(
        calc_perc,
        rrr,
        input_core_dims=[["stack_dim"]],
        output_core_dims=[["percentiles"]],
        keep_attrs=True,
        kwargs=dict(percentiles=per, alpha=alpha, beta=beta, copy=copy),
        dask="parallelized",
        output_dtypes=[rrr.dtype],
        dask_gufunc_kwargs=dict(output_sizes={"percentiles": len(per)}),
    )
    p = p.assign_coords(percentiles=xr.DataArray(per, dims=("percentiles",)))

    # The percentile for the 366th day has a sample size of 1/4 of the other days.
    # To have the same sample size, we interpolate the percentile from 1-365 doy range to 1-366
    if p.dayofyear.max() == 366:
        p = xclim.core.calendar.adjust_doy_calendar(
            p.sel(dayofyear=(p.dayofyear < 366)), arr
        )
    return p.rename("per")

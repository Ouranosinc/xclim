from inspect import signature
from typing import Any, Callable, Dict, Optional, Sequence, Union

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
    per_clim_years = xr.cftime_range(*clim, freq="YS").year

    # `da` over base period used to compute percentile
    in_base_da = da.sel(time=slice(*clim))

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
    da_years = da.resample(time=bfreq).groups
    no_overlap_years = list(
        filter(lambda x: get_year(x[0]) not in per_clim_years, da_years.items())
    )
    if len(no_overlap_years) == 0:
        raise KeyError(
            "`bootstrap` is unnecessary when all years between in_base and out_of_base are overlapping"
        )
    in_base_years = in_base_da.resample(time=bfreq).groups
    out = []

    for label, time_slice in no_overlap_years:
        kw = {da_key: da.isel(time=time_slice), **kwargs}
        kw[per_key] = per
        value = compute_indice_func(**kw)
        out.append(value)

    if xclim.core.utils.uses_dask(in_base_da):
        chunking = {d: "auto" for d in da.dims}
        chunking["time"] = -1  # no chunking on time for percentile_doy in map_block
        in_base_da = in_base_da.chunk(chunking)

    # TODO, would be better with xr.chunksizes be it needs xarray>=20
    # TODO, make sure time is always the first dimension
    a = (len(out[0].time),) + in_base_da.chunks[1:]
    template = out[0].chunk(a)
    # Compute bootstrapped index on each year
    for label, time_slice in in_base_years.items():
        da_year = da.isel(time=time_slice)
        kw = {da_key: da_year, **kwargs}
        template.coords["time"] = pd.date_range(
            start=da_year.time[0].dt.date.values[()],
            end=da_year.time[-1].dt.date.values[()],
            freq=freq,
        )
        value = xr.map_blocks(
            yolo,
            obj=in_base_da,
            kwargs={
                "in_base_years": in_base_years,
                "label": label,
                "kw": kw,
                "per_key": per_key,
                "pdoy_args": pdoy_args,
                "compute_indice_func": compute_indice_func,
            },
            template=template,
        )
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


def yolo(in_base_da, in_base_years, label, kw, per_key, pdoy_args, compute_indice_func):
    bda = bootstrap_year(in_base_da, in_base_years, label)
    kw[per_key] = percentile_doy(bda, **pdoy_args, copy=False)
    return compute_indice_func(**kw).mean(dim="_bootstrap", keep_attrs=True)


# TODO: Return a generator instead and assess performance
def bootstrap_year(
    da: DataArray, groups: Dict[Any, slice], label: Any, dim: str = "time"
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
    bdim = "_bootstrap"
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
#     """Percentile value for each day of the year.
#
#     Return the climatological percentile over a moving window around each day of the year.
#     Different quantile estimators can be used by specifying `alpha` and `beta` according to specifications given by [HyndmanFan]_. The default definition corresponds to method 8, which meets multiple desirable statistical properties for sample quantiles. Note that `numpy.percentile` corresponds to method 7, with alpha and beta set to 1.
#
#     Parameters
#     ----------
#     arr : xr.DataArray
#       Input data, a daily frequency (or coarser) is required.
#     window : int
#       Number of time-steps around each day of the year to include in the calculation.
#     per : float or sequence of floats
#       Percentile(s) between [0, 100]
#     alpha: float
#         Plotting position parameter.
#     beta: float
#         Plotting position parameter.
#     copy: bool
#         If True (default) the input array will be deep copied. It's a necessary step
#         to keep the data integrity but it can be costly.
#         If False, no copy is made of the input array. It will be mutated and rendered
#         unusable but performances may significantly improve.
#         Put this flag to False only if you understand the consequences.
#
#     Returns
#     -------
#     xr.DataArray
#       The percentiles indexed by the day of the year.
#       For calendars with 366 days, percentiles of doys 1-365 are interpolated to the 1-366 range.
#
#     References
#     ----------
#     .. [HyndmanFan] Hyndman, R. J., & Fan, Y. (1996). Sample quantiles in statistical packages. The American Statistician, 50(4), 361-365.
#     """
#     from .utils import calc_perc
#
#     # Ensure arr sampling frequency is daily or coarser
#     # but cowardly escape the non-inferrable case.
#     if compare_offsets(xr.infer_freq(arr.time) or "D", "<", "D"):
#         raise ValueError("input data should have daily or coarser frequency")
#
#     rr = arr.rolling(min_periods=1, center=True, time=window).construct("window")
#
#     ind = pd.MultiIndex.from_arrays(
#         (rr.time.dt.year.values, rr.time.dt.dayofyear.values),
#         names=("year", "dayofyear"),
#     )
#     rrr = rr.assign_coords(time=ind).unstack("time").stack(stack_dim=("year", "window"))
#
#     if rrr.chunks is not None and len(rrr.chunks[rrr.get_axis_num("stack_dim")]) > 1:
#         # Preserve chunk size
#         time_chunks_count = len(arr.chunks[arr.get_axis_num("time")])
#         doy_chunk_size = np.ceil(len(rrr.dayofyear) / (window * time_chunks_count))
#         rrr = rrr.chunk(dict(stack_dim=-1, dayofyear=doy_chunk_size))
#
#     if np.isscalar(per):
#         per = [per]
#
#     p = xr.apply_ufunc(
#         calc_perc,
#         rrr,
#         input_core_dims=[["stack_dim"]],
#         output_core_dims=[["percentiles"]],
#         keep_attrs=True,
#         kwargs=dict(percentiles=per, alpha=alpha, beta=beta, copy=copy),
#         dask="parallelized",
#         output_dtypes=[rrr.dtype],
#         dask_gufunc_kwargs=dict(output_sizes={"percentiles": len(per)}),
#     )
#     p = p.assign_coords(percentiles=xr.DataArray(per, dims=("percentiles",)))
#
#     # The percentile for the 366th day has a sample size of 1/4 of the other days.
#     # To have the same sample size, we interpolate the percentile from 1-365 doy range to 1-366
#     if p.dayofyear.max() == 366:
#         p = adjust_doy_calendar(p.sel(dayofyear=(p.dayofyear < 366)), arr)
#
#     p.attrs.update(arr.attrs.copy())
#
#     # Saving percentile attributes
#     n = len(arr.time)
#     p.attrs["climatology_bounds"] = (
#         arr.time[0 :: n - 1].dt.strftime("%Y-%m-%d").values.tolist()
#     )
#     p.attrs["window"] = window
#     p.attrs["alpha"] = alpha
#     p.attrs["beta"] = beta
#     return p.rename("per")

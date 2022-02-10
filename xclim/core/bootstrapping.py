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
    template = out[0]

    if xclim.core.utils.uses_dask(overlapping_da):
        chunking = {d: "auto" for d in da.dims}
        chunking["time"] = -1  # no chunking on time to use map_block
        overlapping_da = overlapping_da.chunk(chunking)
        # TODO, 1. would be better with xr.chunksizes be it needs xarray>=20
        # TODO, 2. make sure time is always the first dimension
        a = (len(out[0].time),) + overlapping_da.chunks[1:]
        template = out[0].chunk(a)

    # def bs_percentile_doys(overlapping_da, overlapping_years, pdoy_args):
    #     out = []
    #     for year, _ in overlapping_years.items():
    #         bda = build_bootstrap_year_da(overlapping_da,
    #                                       overlapping_years,
    #                                       year,
    #                                       bs_dim_name="_bootstrap")
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
    #         "per_key":           per_key,
    #         "pdoy_args":         pdoy_args,
    #     },
    #     template=per_template,
    # )

    # Compute bootstrapped index on each year
    overlapping_da_years = overlapping_da.resample(time=bfreq).groups.items()
    for year_label, _ in overlapping_da_years:
        da_year = da.sel(time=str(get_year(year_label)))
        kw = {da_key: da_year, **kwargs}
        template.coords["time"] = pd.date_range(
            start=da_year.time[0].dt.date.values[()],
            periods=len(template["time"]),
            freq=freq,
        )
        value = xr.map_blocks(
            bootstrap_year,
            obj=overlapping_da,
            kwargs={
                "overlapping_years": overlapping_years,
                "year": year_label,
                "initial_kwargs": kw,
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


def bootstrap_year(
    overlapping_da,
    overlapping_years: Dict[Any, slice],
    year: str,
    initial_kwargs: Dict,
    per_key: str,
    pdoy_args: Dict,
    compute_indice_func: Callable,
):
    # bda = build_bootstrap_year_da(
    #     overlapping_da, overlapping_years, year, bs_dim_name="_bootstrap"
    # )
    gr = overlapping_years.copy()
    da_copy = overlapping_da.copy(deep=True)
    bloc = overlapping_da["time"][gr.pop(year)]
    da_copy.loc[{"time": bloc}] = np.NAN
    out_accumulator = []
    for i, (key, group_slice) in enumerate(gr.items()):
        out_accumulator.append(overlapping_da.isel({"time": group_slice}))
    bs_arr = xr.concat(out_accumulator, dim="bdim")
    bs_arr = bs_arr.rolling(min_periods=1, center=True, time=5).construct("window")
    bs_arr = (
        bs_arr.assign_coords(
            time=pd.MultiIndex.from_arrays(
                (bs_arr.time.dt.year.values, bs_arr.time.dt.dayofyear.values),
                names=("year", "dayofyear"),
            )
        )
        .unstack("time")
        .stack(stack_dim=("year", "window"))
    )
    rr = da_copy.rolling(min_periods=1, center=True, time=5).construct("window")
    ind = pd.MultiIndex.from_arrays(
        (rr.time.dt.year.values, rr.time.dt.dayofyear.values),
        names=("year", "dayofyear"),
    )
    rr = rr.assign_coords(time=ind).unstack("time").stack(stack_dim=("year", "window"))
    per = [90]
    p = xr.apply_ufunc(
        calc_perc,
        rr,
        input_core_dims=[["stack_dim"]],
        output_core_dims=[["percentiles"]],
        keep_attrs=True,
        kwargs=dict(
            percentiles=per, bs_arr=bs_arr.values, alpha=1 / 3, beta=1 / 3, copy=False
        ),
        dask="parallelized",
        output_dtypes=[rr.dtype],
        dask_gufunc_kwargs=dict(output_sizes={"percentiles": len(per)}),
    )
    initial_kwargs[per_key] = p
    return compute_indice_func(**initial_kwargs).mean(dim="_bootstrap", keep_attrs=True)


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
    da_copy = da.copy(deep=True)

    # Location along dim that must be replaced
    bloc = da[dim][gr.pop(label)]

    # Initialize output array with new bootstrap dimension
    bdim = bs_dim_name
    out = da_copy.expand_dims({bdim: np.arange(len(gr))})
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


# def bs_percentile_doy(
#     arr: xr.DataArray,
#     window: int = 5,
#     per: Union[float, Sequence[float]] = 10.0,
#     alpha: float = 1.0 / 3.0,
#     beta: float = 1.0 / 3.0,
#     copy: bool = True,
# ) -> xr.DataArray:
#     rr = arr.rolling(min_periods=1, center=True, time=window).construct("window")
#     ind = pd.MultiIndex.from_arrays(
#         (rr.time.dt.year.values, rr.time.dt.dayofyear.values),
#         names=("year", "dayofyear"),
#     )
#     rr = rr.assign_coords(time=ind).unstack("time").stack(stack_dim=("year", "window"))
#     if np.isscalar(per):
#         per = [per]
#
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


def calc_perc(
    arr: np.ndarray,
    bs_arr,
    percentiles: Sequence[float] = [50.0],
    alpha: float = 1.0,
    beta: float = 1.0,
    copy: bool = True,
) -> np.ndarray:
    """
    Compute percentiles using nan_calc_percentiles and move the percentiles axis to the end.
    """
    return np.moveaxis(
        nan_calc_percentiles(
            arr=arr,
            bs_arr=bs_arr,
            percentiles=percentiles,
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
    bs_arr,
    percentiles: Sequence[float] = [50.0],
    axis=-1,
    alpha=1.0,
    beta=1.0,
    copy=True,
) -> np.ndarray:
    """
    Convert the percentiles to quantiles and compute them using _nan_quantile.
    """
    if copy:
        # bootstrapping already works on a data's copy
        # doing it again is extremely costly, especially with dask.
        arr = arr.copy()
    quantiles = np.array([per / 100.0 for per in percentiles])
    return _nan_quantile(arr, quantiles, bs_arr, axis, alpha, beta)


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
    bs_arr: np.ndarray,
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
    data_axis_length = arr.shape[axis] + bs_arr.shape[axis]
    if data_axis_length == 0:
        return np.NAN
    if data_axis_length == 1:
        result = np.take(arr, 0, axis=axis)
        return np.broadcast_to(result, (quantiles.size,) + result.shape)
    # The dimensions of `q` are prepended to the output shape, so we need the
    # axis being sampled from `arr` to be last.
    DATA_AXIS = -1
    if axis != DATA_AXIS:  # But moveaxis is slow, so only call it if axis!=0.
        arr = np.moveaxis(arr, axis, destination=DATA_AXIS)
    # nan_count is not a scalar
    nan_count = np.isnan(arr).sum(axis=DATA_AXIS).astype(float)
    bs_nan_count = np.isnan(bs_arr).sum(axis=DATA_AXIS).astype(float)
    valid_values_count = data_axis_length - nan_count - bs_nan_count
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
    # Add bootstrap axis
    # Fixme, broadcast here may exhaust memory...
    arr = np.broadcast_to(arr, (bs_arr.shape[0],) + arr.shape)
    data_arr = np.concatenate([arr, bs_arr], axis=DATA_AXIS)
    data_arr.sort(axis=DATA_AXIS)
    # --- Get values from indexes
    data_arr = data_arr[..., np.newaxis]
    previous = np.squeeze(
        np.take_along_axis(
            data_arr, previous_indexes.astype(int)[np.newaxis, ...], axis=0
        ),
        axis=0,
    )
    next_elements = np.squeeze(
        np.take_along_axis(data_arr, next_indexes.astype(int)[np.newaxis, ...], axis=0),
        axis=0,
    )
    # --- Linear interpolation
    gamma = _get_gamma(virtual_indexes, previous_indexes)
    interpolation = _linear_interpolation(previous, next_elements, gamma)
    # When an interpolation is in Nan range, (near the end of the sorted array) it means
    # we can clip to the array max value.
    result = np.where(
        np.isnan(interpolation), np.nanmax(data_arr, axis=0), interpolation
    )
    # Move quantile axis in front
    result = np.moveaxis(result, axis, 0)
    np.mean(data_arr, axis=-1)
    return result

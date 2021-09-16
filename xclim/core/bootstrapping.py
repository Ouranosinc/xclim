from inspect import signature
from typing import Any, Callable, Dict

import numpy as np
import xarray
from boltons.funcutils import wraps
from xarray.core.dataarray import DataArray

from xclim.core.calendar import convert_calendar, parse_offset, percentile_doy


def percentile_bootstrap(func):
    """Decorator applying a bootstrap step to the calculation of exceedance over a percentile threshold.

    This feature is experimental.

    Boostraping avoids discontinuities in the exceedance between the "in base" period over which percentiles are
    computed, and "out of base" periods. See `bootstrap_func` for details.

    Example of declaration:
    @declare_units(tas="[temperature]", t90="[temperature]")
    @percentile_bootstrap
    def tg90p(
        tas: xarray.DataArray,
        t90: xarray.DataArray,
        freq: str = "YS",
        bootstrap: bool = False
    ) -> xarray.DataArray:

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


def bootstrap_func(compute_indice_func: Callable, **kwargs) -> xarray.DataArray:
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
    The bootstrap algorithm implemented here does the following:

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
    for name, val in kwargs.items():
        if isinstance(val, DataArray):
            if "percentile_doy" in val.attrs.get("history", ""):
                per_key = name
            else:
                da_key = name

    # Extract the DataArray inputs from the arguments
    da: DataArray = kwargs.pop(da_key)
    per: DataArray = kwargs.pop(per_key)

    # List of years in base period
    clim = per.attrs["climatology_bounds"]
    per_clim_years = xarray.cftime_range(*clim, freq="YS").year

    # `da` over base period used to compute percentile
    da_base = da.sel(time=slice(*clim))

    # Arguments used to compute percentile
    percentile = per.percentiles.data.tolist()  # Can be a list or scalar
    pdoy_args = dict(
        window=per.attrs["window"],
        per=percentile if np.isscalar(percentile) else percentile[0],
    )

    # Group input array in years, with an offset matching freq
    freq = kwargs["freq"]
    _, base, start_stamp, anchor = parse_offset(freq)
    bfreq = "A"
    if start_stamp is not None:
        bfreq += "S"
    if base in ["A", "Q"] and anchor is not None:
        bfreq = f"{bfreq}-{anchor}"
    da_years = da.resample(time=bfreq).groups
    in_base_years = da_base.resample(time=bfreq).groups

    out = []
    # Compute func on each grouping
    for label, time_slice in da_years.items():
        year = label.astype("datetime64[Y]").astype(int) + 1970
        kw = {da_key: da.isel(time=time_slice), **kwargs}

        # If the group year is in the base period, run the bootstrap
        if year in per_clim_years:
            bda = bootstrap_year(da_base, in_base_years, label)
            kw[per_key] = percentile_doy(bda, **pdoy_args)
            value = compute_indice_func(**kw).mean(dim="_bootstrap", keep_attrs=True)

        # Otherwise run the normal computation using the original percentile
        else:
            kw[per_key] = per
            value = compute_indice_func(**kw)

        out.append(value)
    out = xarray.concat(out, dim="time")
    duplications = out.get_index("time").duplicated()
    if len(duplications) > 0:
        out = out.sel(time=~duplications)
    out.attrs["units"] = value.attrs["units"]
    return out


# TODO: Return a generator instead and assess performance
def bootstrap_year(
    da: DataArray, groups: Dict[Any, slice], label: Any, dim: str = "time"
) -> DataArray:
    """Return an array where a group in the original is replace by every other groups along a new dimension.

    Parameters
    ----------
    da : DataArray
      Original input array over base period.
    groups : dict
      Output of grouping functions, such as `DataArrayResample.groups`.
    label : Any
      Key identifying the group item to replace.

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

    return xarray.concat(out_accumulator, dim=bdim)

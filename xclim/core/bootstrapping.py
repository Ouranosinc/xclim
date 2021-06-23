from dataclasses import dataclass
from functools import wraps
from inspect import signature
from typing import Callable, Optional

import click
import numpy as np
import xarray as xr
from xarray.core.dataarray import DataArray

import xclim.core.calendar as calendar
from xclim.core.calendar import convert_calendar, parse_offset, percentile_doy
from xclim.core.percentile_config import PercentileConfig

# TODO: I'm not sure this matches all signatures (missing optional thresh argument)
ExceedanceFunction = Callable[[DataArray, DataArray, str, Optional[int]], DataArray]


def percentile_bootstrap(func):
    """Decorator applying a bootstrap step to the calculation of exceedance over a percentile threshold.

    Boostraping avoids discontinuities in the exceedance between the "in base" period over which percentiles are
    computed, and "out of base" periods. See `bootstrap_func` for details.

    Example of declaration:
    @declare_units(tas="[temperature]", t90="[temperature]")
    @percentile_bootstrap
    def tg90p(
        tas: xarray.DataArray,
        t90: xarray.DataArray,
        freq: str = "YS",
        bootstrap=False
    ) -> xarray.DataArray:

    Example when called:
    >>> t90 = percentile_doy(ds.tmax, window=5, per=90)
    >>> tg90p(tas=da, t90=t90, freq="YS", bootstrap=True)
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        # TODO: Modify signature and docstring to include bootstrap parameter

        bootstrap = kwargs.pop("bootstrap", False)
        if bootstrap is False:
            return func(*args, **kwargs)

        ba = signature(func).bind(*args, **kwargs)
        ba.apply_defaults()
        return bootstrap_func(func, **ba.arguments)

    return wrapper


def bootstrap_func(func, **kwargs):
    """Bootstrap the computation of percentile-based exceedance indices.

    Indices measuring exceedance over percentile-based threshold may contain artificial discontinuities at the
    beginning and end of the base period used for calculating the percentile. A bootstrap resampling
    procedure can reduce those discontinuities by iteratively replacing each the year the indice is computed on from
    the percentile estimate, and replacing it with another year within the base period.

    Parameters
    ----------
    func : callable
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
            if "percentile_doy" in val.attrs.get("xclim_history", ""):
                per_key = name
            else:
                da_key = name

    # Extract the DataArray inputs from the arguments
    da = kwargs.pop(da_key)
    per = kwargs.pop(per_key)

    # List of years in base period
    clim = per.attrs["climatology_bounds"]
    per_clim_years = xr.cftime_range(*clim, freq="YS").year

    # `da` over base period used to compute percentile
    da_base = da.sel(time=slice(*clim))

    # Arguments used to compute percentile
    pdoy_args = dict(window=per.attrs["window"], per=per.percentiles.data.tolist()[0])

    # Group input array in years, with an offset matching freq
    freq = kwargs["freq"]
    mul, b, anchor = parse_offset(freq)
    bfreq = "YS"
    if anchor is not None:
        bfreq += f"-{anchor}"
    g_full = da.resample(time=bfreq).groups
    g_base = da_base.resample(time=bfreq).groups

    out = []
    # Compute func on each grouping
    for label, sl in g_full.items():
        year = label.astype("datetime64[Y]").astype(int) + 1970
        kw = {da_key: da[sl], **kwargs}

        # If the group year is in the base period, run the bootstrap
        if year in per_clim_years:
            bda = bootstrap_year(da_base, g_base, label)
            kw[per_key] = percentile_doy(bda, **pdoy_args)
            value = func(**kw).mean(dim="_bootstrap")

        # Otherwise run the normal computation using the original percentile
        else:
            kw[per_key] = per
            value = func(**kw)

        out.append(value)

    return xr.concat(out, dim="time")


# TODO: Return a generator instead and assess performance
def bootstrap_year(da, groups, label, dim="time"):
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

    # Replace `bloc` by every other group
    for i, (key, gsl) in enumerate(gr.items()):
        source = da.isel({dim: gsl})

        if len(source[dim]) == len(bloc):
            out.loc[{bdim: i, dim: bloc}] = source.data
        elif len(bloc) == 365:
            out.loc[{bdim: i, dim: bloc}] = convert_calendar(source, "365_day").data
        elif len(bloc) == 366:
            out.loc[{bdim: i, dim: bloc[:-1]}] = convert_calendar(
                source, "366_day"
            ).data
        else:
            raise NotImplementedError

    return out


def compute_bootstrapped_exceedance_rate(
    da: DataArray, config: PercentileConfig, exceedance_function: ExceedanceFunction
) -> DataArray:
    """Bootsrap function for percentiles.

    Parameters
    ----------
    da : xr.DataArray
    Input data, a daily frequency (or coarser) is required.
    config : PercentileConfig
    bootstrap configuration, including :
    - the percentile value
    - the window size in days
    - the comparison operator
    - the slices for in-base and out of base periodes

    Returns
    -------
    xr.DataArray
    The exceedance rates of a whole period


    ref: Zhang et al https://doi.org/10.1175/JCLI3366.1

    Note: in the text below the base period is choosen to be 30 years for the sake of simplicity,
    In reality any period duration will introduce the same bias and can be fixed by the same algorithm.

    When computing percentile based indices on dataset for a long period,
    we may want to calculate the percentiles of each day for a base periode, for example the first 30 years,
    then we use theses percentiles to see how, in the whole periode, the daily exceedance rate evolves.
    The exceedance rate of a year is the number of days where they individually exceed the thershold (the percentile) for this day of the year.
    For example, if the 95 percentile for the 5 march is 10 °C it means 95% of 5 march in the base period have an averaged tempareture below 10°C.
    Once this is estimated for each day of a year (eventually skipping the 29 feb), we can use theses values to calculate exceedane rate of eac year.
    This can give for example, that the year 2021 has 50 days exceeding the 95 percentile.
    Plotting the evolution of this count for each years gives a good view of how the extreme values are becoming more or less frequent.

    However this analysis introduces a bias because the percentile is computed using the data of the base period and is also used for the calculation of exceedance rates for the same base period.
    Thus the years of the base period are affected by sampling variability.

    To workaround this issue in the in-base period, we can apply the following algorithm
    The data set of in-base period will thereafter be called ds
    1. Split ds in an in-base of 29 years and a single year of out of base period
    2. Add a virtual year in the constructed in-base period by repeating one year.
    3. Get the thershold (your percentile of interest) form this 30 years in-base period
    4. Calculate the exceedance rate of the out base period (the single year)
    5. Repeat 28 times the steps 2, 3, 4 but each time, the repeated year of step 2 must be another year of the in-base
    6. To obtain the final time serie of exceedance rate for the out of base year, average the 29 estimates obtained

    7. Repeat the whole process for each year of the base period

    TODO: show how to bootstrap a custom indice

    """
    in_base_period = da.sel(time=config.in_base_slice)
    if in_base_period.size == 0:
        raise click.BadOptionUsage(
            "percentile_config",
            f"The in base slice {config.in_base_slice} correspond to an empty period in the dataset.\
              Make sure to use a slice of your dataset time serie",
        )
    in_base_exceedance_rates = _bootstrap_period(
        in_base_period, config, exceedance_function
    )
    if config.out_of_base_slice is None:
        return in_base_exceedance_rates
    out_of_base_period = da.sel(time=config.out_of_base_slice)
    in_base_threshold = config.in_base_percentiles
    out_of_base_exceedance = _calculate_exceedances(
        config, exceedance_function, out_of_base_period, in_base_threshold
    )
    return xr.concat([in_base_exceedance_rates, out_of_base_exceedance], dim="time")


def _bootstrap_period(
    ds_in_base_period: DataArray,
    config: PercentileConfig,
    exceedance_function: ExceedanceFunction,
) -> DataArray:
    period_exceedance_rates = []
    for year in np.unique(ds_in_base_period.time.dt.year):
        period_exceedance_rates.append(
            _bootstrap_year(ds_in_base_period, year, config, exceedance_function)
        )
    out = xr.concat(period_exceedance_rates, dim="time")
    # workaround to ensure unit is really "days"
    out.attrs["units"] = "d"
    return out


def _bootstrap_year(
    ds_in_base_period: DataArray,
    out_base_year: int,
    config: PercentileConfig,
    exceedance_function: ExceedanceFunction,
) -> DataArray:
    """ """

    in_base = _build_virtual_in_base_period(ds_in_base_period, out_base_year)
    out_base = ds_in_base_period.sel(time=str(out_base_year))
    exceedance_rates = []
    for year in np.unique(in_base.time.dt.year):
        print(year)
        completed_in_base = _build_completed_in_base(in_base, out_base, year)
        thresholds = _calculate_thresholds(completed_in_base, config)
        exceedance_rate = _calculate_exceedances(
            config, exceedance_function, out_base, thresholds
        )
        exceedance_rates.append(exceedance_rate)
    if len(exceedance_rates) == 1:
        return exceedance_rates[0]
    return xr.concat(exceedance_rates, dim="in_base_period").mean(dim="in_base_period")


def _build_completed_in_base(
    in_base: DataArray,
    out_base_year: DataArray,
    year_to_replicate: int,
):
    out_of_base_calendar = int(out_base_year.time.dt.day.count())
    replicated_year = in_base.sel(time=str(year_to_replicate))
    replicated_year_calendar = int(replicated_year.time.dt.day.count())
    # it is necessary to change the time of the replicated year
    # in order to not skip it in percentile calculation
    if replicated_year_calendar == out_of_base_calendar:
        replicated_year["time"] = out_base_year.time
    elif replicated_year_calendar == 365:
        # TODO that's an ugly instruction to remove 29th Feb...
        out_base_year = out_base_year.drop_sel(
            time=np.datetime64(str(int(out_base_year.time.dt.year[0])) + "-02-29")
        )
        replicated_year["time"] = out_base_year.time
    else:
        replicated_year = calendar.convert_calendar(replicated_year, "noleap")
        replicated_year["time"] = out_base_year.time
    completed_in_base = xr.concat([in_base, replicated_year], dim="time")
    return completed_in_base


# Does not handle computation on a in_base_period of a single year,
# because there would be no out_base_year to exclude
def _build_virtual_in_base_period(
    in_base_period: DataArray, out_base_year: int
) -> DataArray:

    in_base_first_year = in_base_period[0].time.dt.year.item()
    in_base_last_year = in_base_period[-1].time.dt.year.item()
    if in_base_first_year == in_base_last_year:
        raise ValueError(
            "The in_base_period given to _build_virtual_in_base_period must be of at least two years."
        )

    if in_base_first_year == out_base_year:
        return in_base_period.sel(
            time=slice(str(in_base_first_year + 1), str(in_base_last_year))
        )

    if in_base_last_year == out_base_year:
        return in_base_period.sel(
            time=slice(str(in_base_first_year), str(in_base_last_year - 1))
        )

    in_base_time_slice_begin = slice(str(in_base_first_year), str(out_base_year - 1))
    in_base_time_slice_end = slice(str(out_base_year + 1), str(in_base_last_year))
    return xr.concat(
        [
            in_base_period.sel(time=in_base_time_slice_begin),
            in_base_period.sel(time=in_base_time_slice_end),
        ],
        dim="time",
    )


def _calculate_thresholds(
    in_base_period: DataArray, config: PercentileConfig
) -> DataArray:
    return percentile_doy(
        arr=in_base_period, window=config.percentile_window, per=config.percentile
    ).in_base_percentiles.sel(percentiles=config.percentile)


def _calculate_exceedances(
    config: PercentileConfig,
    exceedance_function: ExceedanceFunction,
    out_of_base_period: DataArray,
    in_base_threshold: DataArray,
) -> DataArray:
    if config.indice_window is not None:
        out_of_base_exceedance = exceedance_function(
            out_of_base_period,
            in_base_threshold,
            freq=config.freq,
            window=config.indice_window,
        )
    else:
        out_of_base_exceedance = exceedance_function(
            out_of_base_period, in_base_threshold, freq=config.freq
        )
    return out_of_base_exceedance

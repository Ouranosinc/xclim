from dataclasses import dataclass
from functools import wraps
from inspect import signature
from typing import Callable, Optional

import pandas as pd
import xarray as xr
from xarray.core.dataarray import DataArray

from xclim.core.bootstrap_config import NO_BOOTSRAP, BootstrapConfig
from xclim.core.calendar import percentile_doy


def percentile_bootstrap(func):
    """Decorator for indices which can be bootstrapped.

    Only the percentile based indices may benefit from bootstrapping.

    When the indices function is called with a BootstrapConfig parameter,
    the decarator will take over the computation to iterate over the base period in this configuration.
    @see compute_bootstrapped_exceedance_rate for the full Algorithm.

    Example of declaration:
    @declare_units(tas="[temperature]", t90="[temperature]")
    @percentile_bootstrap
    def tg90p(
        tas: xarray.DataArray,
        t90: xarray.DataArray,
        freq: str = "YS",
        bootstrap_config: BootstrapConfig = None,
    ) -> xarray.DataArray:

    Example when called:
    >>> config = BootstrapConfig(percentile=90,
                                percentile_window=5,
                                in_base_slice=slice("2015-01-01", "2018-12-31"),
                                out_of_base_slice=slice("2019-01-01", "2024-12-31"))
    >>> tg90p(tas = da, t90=None, freq="MS", bootstrap_config=config)
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        bound_args = signature(func).bind(*args, **kwargs)
        config = NO_BOOTSRAP
        indice_window = None
        for name, val in bound_args.arguments.items():
            if name == "window":
                indice_window = val
            elif name == "freq":
                freq = val
            elif isinstance(val, DataArray):
                da = val
            elif isinstance(val, BootstrapConfig):
                config = val
        if config != NO_BOOTSRAP:
            config.indice_window = indice_window
            config.freq = freq
            return compute_bootstrapped_exceedance_rate(
                exceedance_function=func, da=da, config=config
            )
        return func(*args, **kwargs)

    return wrapper


ExceedanceFunction = Callable[[DataArray, DataArray, str, Optional[int]], DataArray]


def compute_bootstrapped_exceedance_rate(
    da: DataArray, config: BootstrapConfig, exceedance_function: ExceedanceFunction
) -> DataArray:
    """Bootsrap function for percentiles.

    Parameters
    ----------
    da : xr.DataArray
    Input data, a daily frequency (or coarser) is required.
    config : BootstrapConfig
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
    in_base_exceedance_rates = _bootstrap_period(
        in_base_period, config, exceedance_function
    )
    if config.out_of_base_slice is None:
        return in_base_exceedance_rates
    out_of_base_period = da.sel(time=config.out_of_base_slice)
    in_base_threshold = _calculate_thresholds(in_base_period, config)
    out_of_base_exceedance = _calculate_exceedances(
        config, exceedance_function, out_of_base_period, in_base_threshold
    )
    return xr.concat([in_base_exceedance_rates, out_of_base_exceedance], dim="time")


def _bootstrap_period(
    ds_in_base_period: DataArray,
    config: BootstrapConfig,
    exceedance_function: ExceedanceFunction,
) -> DataArray:
    period_exceedance_rates = []
    for year_ds in ds_in_base_period.groupby("time.year"):
        period_exceedance_rates.append(
            _bootstrap_year(ds_in_base_period, year_ds[0], config, exceedance_function)
        )
    out = xr.concat(period_exceedance_rates, dim="time")
    # workaround to ensure unit is really "days"
    out.attrs["units"] = "d"
    return out


def _bootstrap_year(
    ds_in_base_period: DataArray,
    out_base_year: int,
    config: BootstrapConfig,
    exceedance_function: ExceedanceFunction,
) -> DataArray:
    print("out base :", out_base_year)
    in_base = _build_virtual_in_base_period(ds_in_base_period, out_base_year)
    out_base = ds_in_base_period.sel(time=str(out_base_year))
    exceedance_rates = []
    for year_ds in in_base.groupby("time.year"):
        print(year_ds[0])
        replicated_year = in_base.sel(time=str(year_ds[0]))
        # it is necessary to change the time of the replicated year
        # in order to not skip it in percentile calculation
        replicated_year["time"] = replicated_year.time + pd.Timedelta(
            str(out_base_year - year_ds[0]) + "y"
        )
        completed_in_base = xr.concat([in_base, replicated_year], dim="time")
        thresholds = _calculate_thresholds(completed_in_base, config)
        exceedance_rate = _calculate_exceedances(
            config, exceedance_function, out_base, thresholds
        )
        exceedance_rates.append(exceedance_rate)
    if len(exceedance_rates) == 1:
        return exceedance_rates[0]
    return xr.concat(exceedance_rates, dim="time").groupby("time").mean()


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
    elif in_base_first_year == out_base_year:
        return in_base_period.sel(
            time=slice(str(in_base_first_year + 1), str(in_base_last_year))
        )
    elif in_base_last_year == out_base_year:
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
    in_base_period: DataArray, config: BootstrapConfig
) -> DataArray:
    return percentile_doy(
        in_base_period, config.percentile_window, config.percentile
    ).sel(percentiles=config.percentile)


def _calculate_exceedances(
    config: BootstrapConfig,
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

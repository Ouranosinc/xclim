from dataclasses import dataclass
from functools import wraps
from inspect import signature
from typing import Callable

from xclim.core.indicator import Daily

import pandas as pd
import xarray as xr
from xarray.core.dataarray import DataArray

from xclim.core.calendar import percentile_doy, resample_doy
from xclim.core.units import convert_units_to, declare_units, to_agg_units
from xclim.indices.generic import threshold_count


ExceedanceFunction = Callable[[DataArray, DataArray, str], DataArray]


@dataclass
class BootstrapConfig:
    percentile: int  # ]0, 100[
    in_base_slice: slice
    exceedance_function: Daily
    out_of_base_slice: slice = None  # when None, only the in-base will be computed
    window: int = 5


def compute_bootstrapped_exceedance_rate(da: DataArray, config: BootstrapConfig, exceedance_function: ExceedanceFunction, *args, **kwargs) -> DataArray:
    """ Bootsrap function for percentiles.

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

        Example :
        ds = xr.open_dataset(
            "tasmax_day_MIROC6_ssp585_r1i1p1f1_gn_20150101-20241231.nc")
        config = bootstrapping.BootstrapConfig(
            operator='>',
            percentile=90,
            window=1,
            in_base_slice=slice("2015-01-01", "2017-12-31"),
            out_of_base_slice=slice("2018-01-01", "2024-12-31")
        )
        result = bootstrapping.compute_bootstrapped_exceedance_rate(ds.tasmax, config)
        result.to_netcdf('bootstrap_results_xclim.nc')

    """
    in_base_period = da.sel(time=config.in_base_slice)
    in_base_exceedance_rates = _bootstrap_period(in_base_period,
                                                 config,
                                                 exceedance_function)
    if config.out_of_base_slice == None:
        return in_base_exceedance_rates
    out_of_base_period = da.sel(time=config.out_of_base_slice)
    in_base_threshold = _calculate_thresholds(in_base_period, config)
    out_of_base_exceedance = exceedance_function(
        out_of_base_period, in_base_threshold, *args, **kwargs)
    return xr.concat([in_base_exceedance_rates, out_of_base_exceedance], dim="time")


def _bootstrap_period(ds_in_base_period: DataArray,
                      config: BootstrapConfig,
                      exceedance_function: ExceedanceFunction,
                      *args, **kwargs) -> DataArray:
    period_exceedance_rates = []
    for year_ds in ds_in_base_period.groupby("time.year"):
        period_exceedance_rates.append(
            _bootstrap_year(ds_in_base_period, year_ds[0], config, exceedance_function, *args, **kwargs))
    out = xr.concat(period_exceedance_rates, dim="time")\
        .resample(time="M")\
        .sum(dim="time")
    return to_agg_units(out, period_exceedance_rates[0], "count")


def _bootstrap_year(ds_in_base_period: DataArray,
                    out_base_year: int,
                    config: BootstrapConfig,
                    exceedance_function: ExceedanceFunction,
                    *args, **kwargs) -> DataArray:
    print("out base :", out_base_year)
    in_base = _build_virtual_in_base_period(ds_in_base_period, out_base_year)
    out_base = ds_in_base_period.sel(time=str(out_base_year))
    exceedance_rates = []
    for year_ds in in_base.groupby("time.year"):
        print(year_ds[0])
        replicated_year = in_base.sel(time=str(year_ds[0]))
        # it is necessary to change the time of the replicated year
        # in order to not skip it in percentile calculation
        replicated_year["time"] = replicated_year.time + \
            pd.Timedelta(str(out_base_year - year_ds[0]) + 'y')
        completed_in_base = xr.concat([in_base, replicated_year], dim="time")
        thresholds = _calculate_thresholds(completed_in_base, config)
        exceedance_rate = exceedance_function(out_base, thresholds, *args, **kwargs)
        exceedance_rates.append(exceedance_rate)
    out = xr.concat(exceedance_rates, dim="time")\
        .groupby('time')\
        .mean()
    return to_agg_units(out, exceedance_rates[0], "count")


# Does not handle computation on a in_base_period of a single year, because there would be no out of base year
def _build_virtual_in_base_period(in_base_period: DataArray, out_base_year: int) -> DataArray:
    in_base_first_year = in_base_period[0].time.dt.year.item()
    in_base_last_year = in_base_period[-1].time.dt.year.item()
    if in_base_first_year == in_base_last_year:
        raise ValueError(
            f"The in_base_period given to _build_virtual_in_base_period must be of at least two years."
        )
    elif in_base_first_year == out_base_year:
        return in_base_period.sel(time=slice(str(in_base_first_year + 1), str(in_base_last_year)))
    elif in_base_last_year == out_base_year:
        return in_base_period.sel(time=slice(str(in_base_first_year), str(in_base_last_year - 1)))
    in_base_time_slice_begin = slice(str(in_base_first_year), str(out_base_year - 1))
    in_base_time_slice_end = slice(str(out_base_year + 1), str(in_base_last_year))
    return xr.concat([in_base_period.sel(time=in_base_time_slice_begin),
                      in_base_period.sel(time=in_base_time_slice_end)],
                     dim="time")


def _calculate_thresholds(in_base_period: DataArray, config: BootstrapConfig) -> DataArray:
    return percentile_doy(in_base_period, config.window, config.percentile)\
        .sel(percentiles=config.percentile)


def percentile_bootstrap(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        bound_args = signature(func).bind(*args, **kwargs)
        config = None
        for _, val in bound_args.arguments.items():
            if isinstance(val, DataArray):
                da = val
            elif isinstance(val, BootstrapConfig):
                config = val
        if config != None:
            return compute_bootstrapped_exceedance_rate(exceedance_function=func, da=da, config=config, *args, **kwargs)
        return func(*args, **kwargs)
    return wrapper

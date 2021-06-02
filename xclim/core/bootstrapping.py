from xclim.core.units import convert_units_to, to_agg_units
from numpy.lib.function_base import percentile
import xarray as xr
from xarray.core.dataarray import DataArray
from xarray.core.dataset import calculate_dimensions

from xclim.core.calendar import percentile_doy, resample_doy
from xclim.indices.generic import threshold_count

# Bootsrap function for percentiles.
# ref: Zhang et al https://doi.org/10.1175/JCLI3366.1
#
# Note: in the text below the base period is choosen to be 30 years for the sake of simplicity,
# In reality any period duration will introduce the same bias and can be fixed by the same algorithm.
#
# When computing percentile based indices on dataset for a long period,
# we usually calculate the percentiles of each day for a base periode, for example the first 30 years,
# then we use theses percentiles to see how, in the whole periode, the daily exceedance rate evolves.
# The exceedance rate of a year is the number of days where they individually exceed the thershold (the percentile) for this day of the year.
# For example, if the 95 percentile for the 5 march is 10 °C it means 95% of 5 march in the base period have an averaged tempareture below 10°C.
# Once this is estimated for each day of a year (eventually skipping the 29 feb), we can use theses values to calculate exceedane rate of eac year.
# This can give for example, that the year 2021 has 50 days exceeding the 95 percentile.
# Plotting the evolution of this count for each years gives a good view of how the extreme values are becoming more or less frequent.
#
# However this analysis introduces a bias because the percentile is computed using the data of the base period and is also used for the calculus of exceedance rate for the same base period.
# Thus the years of the base period are affected by sampling variability.
#
# To workaround this issue in the in-base period, we can apply the following algorithm
# The data set of in-base period will thereafter be called ds
# 1. Split ds in an in-base of 29 years and a single year of out of base period
# 2. Add a virtual year in the constructed in-base period by repeating one year.
# 3. Get the thershold (your percentile of interest) form this 30 years in-base period
# 4. Calculate the exceedance rate of the out base period (the single year)
# 5. Repeat 28 times the steps 2, 3, 4 but each time, the repeated year of step 2 must be another year of the in-base
# 6. To obtain the final time serie of exceedance rate for the out of base year, average the 29 estimates obtained
#
# 7. Repeat the whole process for each year of the base period


# public
def percentile_with_bootstrap(ds_in_base_period: DataArray,
                              out_of_base_period: DataArray,
                              expected_percentile: int) -> DataArray:
    in_base = bootstrap_period(ds_in_base_period, expected_percentile)
    in_base_threshold = calculate_tresholds(ds_in_base_period, expected_percentile)
    thresh = resample_doy(in_base_threshold, out_of_base_period)
    # Identify the days with max temp above 90th percentile.
    out = threshold_count(out_of_base_period, ">", thresh, 'D')
    out_of_base = to_agg_units(out, out_of_base_period, "count")
    return xr.concat([in_base, out_of_base], dim="time")


def bootstrap_period(ds_in_base_period: DataArray, expected_percentile: int) -> DataArray:
    # public
    period_exceedance_rates = []
    for year_ds in ds_in_base_period.groupby("time.year"):
        period_exceedance_rates.append(bootstrap_year(
            ds_in_base_period, year_ds[0], expected_percentile))
    return xr.concat(period_exceedance_rates, dim="time")\
        .resample(time="M")\
        .sum(dim="time")


def bootstrap_year(ds_in_base_period: DataArray, out_base_year: int, expected_percentile: int) -> DataArray:
    print("out base :", out_base_year)
    in_base = build_virtual_in_base_period(ds_in_base_period, out_base_year)
    out_base = ds_in_base_period.sel(time=str(out_base_year))
    exceedance_rates = []
    for year_ds in in_base.groupby("time.year"):
        print(year_ds[0])
        completed_in_base = xr.concat([in_base, in_base.sel(time=str(year_ds[0]))],
                                      dim="time")
        thresholds = calculate_tresholds(completed_in_base, expected_percentile)
        exceedance_rate = calculate_exceedances(thresholds, out_base)
        exceedance_rates.append(exceedance_rate)
    return xr.concat(exceedance_rates, dim="time").groupby('time').mean()


# Does not handle computation on a in_base_period of a single year
def build_virtual_in_base_period(in_base_period: DataArray, out_base_year: int) -> DataArray:
    first_year = in_base_period[0].time.dt.year.item()
    last_year = in_base_period[-1].time.dt.year.item()
    if first_year == out_base_year:
        return in_base_period.sel(time=slice(str(first_year + 1), str(last_year)))
    elif last_year == out_base_year:
        return in_base_period.sel(time=slice(str(first_year), str(last_year - 1)))
    in_base_time_slice_begin = slice(str(first_year), str(out_base_year - 1))
    in_base_time_slice_end = slice(str(out_base_year + 1), str(last_year))
    return xr.concat([in_base_period.sel(time=in_base_time_slice_begin),
                      in_base_period.sel(time=in_base_time_slice_end)],
                     dim="time")


def calculate_tresholds(in_base_period: DataArray, expected_percentile: str) -> DataArray:
    threshold = percentile_doy(in_base_period, 5, expected_percentile)\
        .sel(percentiles=expected_percentile)
    return threshold


def calculate_exceedances(thresholds: DataArray, out_base: DataArray) -> DataArray:
    # thresholds = convert_units_to(thresholds, out_base)
    thresh = resample_doy(thresholds, out_base)
    out = threshold_count(out_base, ">", thresh, 'D')
    return to_agg_units(out, out_base, "count")

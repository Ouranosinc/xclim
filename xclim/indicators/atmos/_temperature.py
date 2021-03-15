# -*- coding: utf-8 -*-
"""Temperature indicator definitions."""

from xclim import indices
from xclim.core import cfchecks
from xclim.core.indicator import Daily, Daily2D
from xclim.core.units import check_units
from xclim.core.utils import wrapped_partial

__all__ = [
    "tn_days_below",
    "tx_days_above",
    "tx_tn_days_above",
    "heat_wave_frequency",
    "heat_wave_max_length",
    "heat_wave_total_length",
    "heat_wave_index",
    "hot_spell_frequency",
    "hot_spell_max_length",
    "tg_mean",
    "tg10p",
    "tg90p",
    "tn_min",
    "tn_max",
    "tn_mean",
    "tn10p",
    "tn90p",
    "tx_min",
    "tx_max",
    "tx_mean",
    "tx10p",
    "tx90p",
    "daily_temperature_range",
    "max_daily_temperature_range",
    "daily_temperature_range_variability",
    "extreme_temperature_range",
    "cold_spell_duration_index",
    "cold_spell_days",
    "cold_spell_frequency",
    "daily_freezethaw_cycles",
    "cooling_degree_days",
    "heating_degree_days",
    "growing_degree_days",
    "frost_season_length",
    "freshet_start",
    "frost_days",
    "last_spring_frost",
    "first_day_below",
    "first_day_above",
    "ice_days",
    "consecutive_frost_days",
    "maximum_consecutive_frost_free_days",
    "growing_season_length",
    "growing_season_end",
    "tropical_nights",
    "degree_days_exceedance_date",
    "warm_spell_duration_index",
    "maximum_consecutive_warm_days",
]


# TODO: Should we reference the standard vocabulary we're using ?
# E.g. http://vocab.nerc.ac.uk/collection/P07/current/BHMHISG2/


class Tas(Daily):
    """Class for univariate indices using mean daily temperature as the input."""

    @staticmethod
    def cfcheck(tas):
        cfchecks.check_valid(tas, "cell_methods", "*time: mean within days*")
        cfchecks.check_valid(tas, "standard_name", "air_temperature")


class Tasmin(Daily):
    """Class for univariate indices using min daily temperature as the input."""

    @staticmethod
    def cfcheck(tasmin):
        cfchecks.check_valid(tasmin, "cell_methods", "*time: minimum within days*")
        cfchecks.check_valid(tasmin, "standard_name", "air_temperature")


class Tasmax(Daily):
    """Class for univariate indices using max daily temperature as the input."""

    @staticmethod
    def cfcheck(tasmax):
        cfchecks.check_valid(tasmax, "cell_methods", "*time: maximum within days*")
        cfchecks.check_valid(tasmax, "standard_name", "air_temperature")


class TasminTasmax(Daily2D):
    @staticmethod
    def cfcheck(tasmin, tasmax):
        for da in (tasmin, tasmax):
            cfchecks.check_valid(da, "standard_name", "air_temperature")
        cfchecks.check_valid(tasmin, "cell_methods", "*time: minimum within days*")
        cfchecks.check_valid(tasmax, "cell_methods", "*time: maximum within days*")
        check_units(tasmax, tasmin.attrs["units"])


tn_days_below = Tasmin(
    identifier="tn_days_below",
    units="days",
    standard_name="number_of_days_with_air_temperature_below_threshold",
    long_name="Number of days with Tmin < {thresh}",
    description="{freq} number of days where daily minimum temperature is below {thresh}.",
    cell_methods="time: minimum within days time: sum over days",
    compute=indices.tn_days_below,
)

tx_days_above = Tasmax(
    identifier="tx_days_above",
    units="days",
    standard_name="number_of_days_with_air_temperature_above_threshold",
    long_name="Number of days with Tmax > {thresh}",
    description="{freq} number of days where daily maximum temperature exceeds {thresh}.",
    cell_methods="time: maximum within days time: sum over days",
    compute=indices.tx_days_above,
)

tx_tn_days_above = TasminTasmax(
    identifier="tx_tn_days_above",
    units="days",
    standard_name="number_of_days_with_air_temperature_above_threshold",
    long_name="Number of days with Tmax > {thresh_tasmax} and Tmin > {thresh_tasmin}",
    description="{freq} number of days where daily maximum temperature exceeds "
    "{thresh_tasmax} and minimum temperature exceeds {thresh_tasmin}.",
    cell_methods="",
    compute=indices.tx_tn_days_above,
)

heat_wave_frequency = TasminTasmax(
    identifier="heat_wave_frequency",
    units="",
    standard_name="heat_wave_events",
    long_name="Number of heat wave events (Tmin > {thresh_tasmin} "
    "and Tmax > {thresh_tasmax} for >= {window} days)",
    description="{freq} number of heat wave events over a given period. "
    "An event occurs when the minimum and maximum daily "
    "temperature both exceeds specific thresholds : "
    "(Tmin > {thresh_tasmin} and Tmax > {thresh_tasmax}) "
    "over a minimum number of days ({window}).",
    cell_methods="",
    keywords="health,",
    compute=indices.heat_wave_frequency,
)

heat_wave_max_length = TasminTasmax(
    identifier="heat_wave_max_length",
    units="days",
    standard_name="spell_length_of_days_with_air_temperature_above_threshold",
    long_name="Maximum length of heat wave events (Tmin > {thresh_tasmin}"
    "and Tmax > {thresh_tasmax} for >= {window} days)",
    description="{freq} maximum length of heat wave events occurring in a given period. "
    "An event occurs when the minimum and maximum daily "
    "temperature both exceeds specific thresholds "
    "(Tmin > {thresh_tasmin} and Tmax > {thresh_tasmax}) over "
    "a minimum number of days ({window}).",
    cell_methods="",
    keywords="health,",
    compute=indices.heat_wave_max_length,
)

heat_wave_total_length = TasminTasmax(
    identifier="heat_wave_total_length",
    units="days",
    standard_name="spell_length_of_days_with_air_temperature_above_threshold",
    long_name="Total length of heat wave events (Tmin > {thresh_tasmin} "
    "and Tmax > {thresh_tasmax} for >= {window} days)",
    description="{freq} total length of heat wave events occurring in a given period. "
    "An event occurs when the minimum and maximum daily "
    "temperature both exceeds specific thresholds "
    "(Tmin > {thresh_tasmin} and Tmax > {thresh_tasmax}) over "
    "a minimum number of days ({window}).",
    cell_methods="",
    keywords="health,",
    compute=indices.heat_wave_total_length,
)


heat_wave_index = Tasmax(
    identifier="heat_wave_index",
    units="days",
    standard_name="heat_wave_index",
    long_name="Number of days that are part of a heatwave",
    description="{freq} number of days that are part of a heatwave, "
    "defined as five or more consecutive days over {thresh}.",
    cell_methods="",
    compute=indices.heat_wave_index,
)


hot_spell_frequency = Tasmax(
    identifier="hot_spell_frequency",
    units="",
    standard_name="hot_spell_events",
    long_name="Number of hot spell events (Tmax > {thresh_tasmax} for >= {window} days)",
    description="{freq} number of hot spell events over a given period. "
    "An event occurs when the maximum daily temperature exceeds a specific threshold: (Tmax > {thresh_tasmax}) "
    "over a minimum number of days ({window}).",
    cell_methods="",
    keywords="health,",
    compute=indices.hot_spell_frequency,
)

hot_spell_max_length = Tasmax(
    identifier="hot_spell_max_length",
    units="days",
    standard_name="spell_length_of_days_with_air_temperature_above_threshold",
    long_name="Maximum length of hot spell events (Tmax > {thresh_tasmax} for >= {window} days)",
    description="{freq} maximum length of hot spell events occurring in a given period. "
    "An event occurs when the maximum daily temperature exceeds a specific threshold: (Tmax > {thresh_tasmax}) "
    "over a minimum number of days ({window}).",
    cell_methods="",
    keywords="health,",
    compute=indices.hot_spell_max_length,
)

tg_mean = Tas(
    identifier="tg_mean",
    units="K",
    standard_name="air_temperature",
    long_name="Mean daily mean temperature",
    description="{freq} mean of daily mean temperature.",
    cell_methods="time: mean within days time: mean over days",
    compute=indices.tg_mean,
)

tg_max = Tas(
    identifier="tg_max",
    units="K",
    standard_name="air_temperature",
    long_name="Maximum daily mean temperature",
    description="{freq} maximum of daily mean temperature.",
    cell_methods="time: mean within days time: maximum over days",
    compute=indices.tg_max,
)

tg_min = Tas(
    identifier="tg_min",
    units="K",
    standard_name="air_temperature",
    long_name="Minimum daily mean temperature",
    description="{freq} minimum of daily mean temperature.",
    cell_methods="time: mean within days time: minimum over days",
    compute=indices.tg_min,
)

tx_mean = Tasmax(
    identifier="tx_mean",
    units="K",
    standard_name="air_temperature",
    long_name="Mean daily maximum temperature",
    description="{freq} mean of daily maximum temperature.",
    cell_methods="time: maximum within days time: mean over days",
    compute=indices.tx_mean,
)

tx_max = Tasmax(
    identifier="tx_max",
    units="K",
    standard_name="air_temperature",
    long_name="Maximum daily maximum temperature",
    description="{freq} maximum of daily maximum temperature.",
    cell_methods="time: maximum within days time: maximum over days",
    compute=indices.tx_max,
)

tx_min = Tasmax(
    identifier="tx_min",
    units="K",
    standard_name="air_temperature",
    long_name="Minimum daily maximum temperature",
    description="{freq} minimum of daily maximum temperature.",
    cell_methods="time: maximum within days time: minimum over days",
    compute=indices.tx_min,
)

tn_mean = Tasmin(
    identifier="tn_mean",
    units="K",
    standard_name="air_temperature",
    long_name="Mean daily minimum temperature",
    description="{freq} mean of daily minimum temperature.",
    cell_methods="time: minimum within days time: mean over days",
    compute=indices.tn_mean,
)

tn_max = Tasmin(
    identifier="tn_max",
    units="K",
    standard_name="air_temperature",
    long_name="Maximum daily minimum temperature",
    description="{freq} maximum of daily minimum temperature.",
    cell_methods="time: minimum within days time: maximum over days",
    compute=indices.tn_max,
)

tn_min = Tasmin(
    identifier="tn_min",
    units="K",
    standard_name="air_temperature",
    long_name="Minimum daily minimum temperature",
    description="{freq} minimum of daily minimum temperature.",
    cell_methods="time: minimum within days time: minimum over days",
    compute=indices.tn_min,
)

daily_temperature_range = TasminTasmax(
    title="Mean of daily temperature range.",
    identifier="dtr",
    units="K",
    standard_name="air_temperature",
    long_name="Mean Diurnal Temperature Range",
    description="{freq} mean diurnal temperature range.",
    cell_methods="time range within days time: mean over days",
    compute=wrapped_partial(indices.daily_temperature_range, op="mean"),
)

max_daily_temperature_range = TasminTasmax(
    title="Maximum of daily temperature range.",
    identifier="dtrmax",
    units="K",
    standard_name="air_temperature",
    long_name="Maximum Diurnal Temperature Range",
    description="{freq} maximum diurnal temperature range.",
    cell_methods="time range within days time: max over days",
    compute=wrapped_partial(indices.daily_temperature_range, op="max"),
)

daily_temperature_range_variability = TasminTasmax(
    identifier="dtrvar",
    units="K",
    standard_name="air_temperature",
    long_name="Mean Diurnal Temperature Range Variability",
    description="{freq} mean diurnal temparature range variability "
    "(defined as the average day-to-day variation "
    "in daily temperature range "
    "for the given time period)",
    cell_methods="time range within days time: difference "
    "over days time: mean over days",
    compute=indices.daily_temperature_range_variability,
)

extreme_temperature_range = TasminTasmax(
    identifier="etr",
    units="K",
    standard_name="air_temperature",
    long_name="Intra-period Extreme Temperature Range",
    description="{freq} range between the maximum of daily max temperature "
    "(tx_max) and the minimum of daily min temperature (tn_min)",
    compute=indices.extreme_temperature_range,
)

cold_spell_duration_index = Tasmin(
    identifier="cold_spell_duration_index",
    var_name="csdi_{window}",
    units="days",
    standard_name="cold_spell_duration_index",
    long_name="Number of days part of a percentile-defined cold spell",
    description="{freq} number of days with at least {window} consecutive days "
    "where the daily minimum temperature is below the 10th "
    "percentile. The 10th percentile should be computed for "
    "a 5-day window centred on each calendar day in the  1961-1990 period",
    cell_methods="",
    compute=indices.cold_spell_duration_index,
)

cold_spell_days = Tas(
    identifier="cold_spell_days",
    units="days",
    standard_name="cold_spell_days",
    long_name="Number of days part of a cold spell",
    description="{freq} number of days that are part of a cold spell, defined as {window} "
    "or more consecutive days with mean daily "
    "temperature below  {thresh}.",
    cell_methods="",
    compute=indices.cold_spell_days,
)

cold_spell_frequency = Tas(
    identifier="cold_spell_frequency",
    units="",
    standard_name="cold_spell_frequency",
    long_name="Number of cold spell events",
    description="{freq} number cold spell events, defined as {window} "
    "or more consecutive days with mean daily "
    "temperature below  {thresh}.",
    cell_methods="",
    compute=indices.cold_spell_frequency,
)


daily_freezethaw_cycles = TasminTasmax(
    identifier="dlyfrzthw",
    units="days",
    standard_name="daily_freezethaw_cycles",
    long_name="daily freezethaw cycles",
    description="{freq} number of days with a diurnal freeze-thaw cycle "
    ": Tmax > {thresh_tasmax} and Tmin <= {thresh_tasmin}.",
    cell_methods="",
    compute=indices.daily_freezethaw_cycles,
)

cooling_degree_days = Tas(
    identifier="cooling_degree_days",
    units="K days",
    standard_name="integral_of_air_temperature_excess_wrt_time",
    long_name="Cooling Degree Days (Tmean > {thresh})",
    description="{freq} cooling degree days above {thresh}.",
    cell_methods="time: mean within days time: sum over days",
    compute=indices.cooling_degree_days,
)

heating_degree_days = Tas(
    identifier="heating_degree_days",
    units="K days",
    standard_name="integral_of_air_temperature_deficit_wrt_time",
    long_name="Heating Degree Days (Tmean < {thresh})",
    description="{freq} heating degree days below {thresh}.",
    cell_methods="time: mean within days time: sum over days",
    compute=indices.heating_degree_days,
)

growing_degree_days = Tas(
    identifier="growing_degree_days",
    units="K days",
    standard_name="integral_of_air_temperature_excess_wrt_time",
    long_name="growing degree days above {thresh}",
    description="{freq} growing degree days above {thresh}",
    cell_methods="time: mean within days time: sum over days",
    compute=indices.growing_degree_days,
)

freshet_start = Tas(
    identifier="freshet_start",
    units="",
    standard_name="day_of_year",
    long_name="Day of year of spring freshet start",
    description="Day of year of spring freshet start, defined as the first day a temperature "
    "threshold of {thresh} is exceeded for at least {window} days.",
    compute=indices.freshet_start,
)

frost_days = Tasmin(
    identifier="frost_days",
    units="days",
    standard_name="days_with_air_temperature_below_threshold",
    long_name="Number of Frost Days (Tmin < 0C)",
    description="{freq} number of days with minimum daily temperature below 0℃.",
    cell_methods="time: minimum within days time: sum over days",
    compute=indices.frost_days,
)

frost_season_length = Tasmin(
    identifier="frost_season_length",
    units="days",
    standard_name="days_with_air_temperature_below_threshold",
    long_name="Length of the frost season",
    description="{freq} number of days between the first occurrence of at least "
    "{window} consecutive days with minimum daily temperature below freezing and "
    "the first occurrence of at least {window} consecutive days with "
    "minimuim daily temperature above freezing after {mid_date}.",
    cell_methods="time: minimum within days time: sum over days",
    compute=wrapped_partial(indices.frost_season_length, thresh="0 degC"),
)

last_spring_frost = Tasmin(
    identifier="last_spring_frost",
    units="",
    standard_name="day_of_year",
    long_name="Day of year of last spring frost",
    description="Day of year of last spring frost, defined as the last day a minimum temperature "
    "threshold of {thresh} is not exceeded before a given date.",
    compute=indices.last_spring_frost,
)

first_day_below = Tasmin(
    identifier="first_day_below",
    units="",
    standard_name="day_of_year",
    long_name="First day of year with temperature below {thresh}",
    description="First day of year with temperature below {thresh} for at least {window} days.",
    compute=indices.first_day_below,
)

first_day_above = Tasmin(
    identifier="first_day_above",
    units="",
    standard_name="day_of_year",
    long_name="First day of year with temperature above {thresh}",
    description="First day of year with temperature above {thresh} for at least {window} days.",
    compute=indices.first_day_above,
)


ice_days = Tasmax(
    identifier="ice_days",
    standard_name="days_with_air_temperature_below_threshold",
    units="days",
    long_name="Number of Ice Days (Tmax < 0℃)",
    description="{freq} number of days with maximum daily temperature below 0℃",
    cell_methods="time: maximum within days time: sum over days",
    compute=indices.ice_days,
)

consecutive_frost_days = Tasmin(
    identifier="consecutive_frost_days",
    units="days",
    standard_name="spell_length_of_days_with_air_temperature_below_threshold",
    long_name="Maximum number of consecutive days with Tmin < {thresh}",
    description="{freq} maximum number of consecutive days with "
    "minimum daily temperature below {thresh}",
    cell_methods="time: min within days time: maximum over days",
    compute=indices.maximum_consecutive_frost_days,
)

maximum_consecutive_frost_free_days = Tasmin(
    identifier="consecutive_frost_free_days",
    units="days",
    standard_name="spell_length_of_days_with_air_temperature_above_threshold",
    long_name="Maximum number of consecutive days with Tmin > {thresh}",
    description="{freq} maximum number of consecutive days with "
    "minimum daily temperature above {thresh}.",
    cell_methods="time: min within days time: maximum over days",
    compute=indices.maximum_consecutive_frost_free_days,
)

growing_season_length = Tas(
    identifier="growing_season_length",
    units="days",
    standard_name="growing_season_length",
    long_name="ETCCDI Growing Season Length (Tmean > {thresh})",
    description="{freq} number of days between the first occurrence of at least "
    "{window} consecutive days with mean daily temperature over {thresh} and "
    "the first occurrence of at least {window} consecutive days with "
    "mean daily temperature below {thresh} after {mid_date}.",
    cell_methods="",
    compute=indices.growing_season_length,
)

growing_season_end = Tas(
    identifier="growing_season_end",
    units="",
    standard_name="day_of_year",
    long_name="Day of year of growing season end",
    description="Day of year of end of growing season, defined as the first day of "
    "consistent inferior threshold temperature of {thresh} after a run of "
    "{window} days superior to threshold temperature.",
    cell_methods="",
    compute=indices.growing_season_end,
)

tropical_nights = Tasmin(
    identifier="tropical_nights",
    units="days",
    standard_name="number_of_days_with_air_temperature_above_threshold",
    long_name="Number of Tropical Nights (Tmin > {thresh})",
    description="{freq} number of Tropical Nights : defined as days with minimum daily temperature"
    " above {thresh}",
    cell_methods="time: minimum within days time: sum over days",
    compute=indices.tropical_nights,
)

tg90p = Tas(
    identifier="tg90p",
    units="days",
    standard_name="days_with_air_temperature_above_threshold",
    long_name="Number of days when Tmean > 90th percentile",
    description="{freq} number of days with mean daily temperature above the 90th percentile."
    "The 90th percentile is to be computed for a 5 day moving window centered on each calendar day "
    "for a reference period.",
    cell_methods="time: mean within days time: sum over days",
    compute=indices.tg90p,
)

tg10p = Tas(
    identifier="tg10p",
    units="days",
    standard_name="days_with_air_temperature_below_threshold",
    long_name="Number of days when Tmean < 10th percentile",
    description="{freq} number of days with mean daily temperature below the 10th percentile."
    "The 10th percentile is to be computed for a 5 day moving window centered on each calendar day "
    "for a reference period.",
    cell_methods="time: mean within days time: sum over days",
    compute=indices.tg10p,
)

tx90p = Tasmax(
    identifier="tx90p",
    units="days",
    standard_name="days_with_air_temperature_above_threshold",
    long_name="Number of days when Tmax > 90th percentile",
    description="{freq} number of days with maximum daily temperature above the 90th percentile."
    "The 90th percentile is to be computed for a 5 day moving window centered on each calendar day "
    "for a reference period.",
    cell_methods="time: maximum within days time: sum over days",
    compute=indices.tx90p,
)

tx10p = Tasmax(
    identifier="tx10p",
    units="days",
    standard_name="days_with_air_temperature_below_threshold",
    long_name="Number of days when Tmax < 10th percentile",
    description="{freq} number of days with maximum daily temperature below the 10th percentile."
    "The 10th percentile is to be computed for a 5 day moving window centered on each calendar day "
    "for a reference period.",
    cell_methods="time: maximum within days time: sum over days",
    compute=indices.tx10p,
)

tn90p = Tasmin(
    identifier="tn90p",
    units="days",
    standard_name="days_with_air_temperature_above_threshold",
    long_name="Number of days when Tmin > 90th percentile",
    description="{freq} number of days with minimum daily temperature above the 90th percentile."
    "The 90th percentile is to be computed for a 5 day moving window centered on each calendar day "
    "for a reference period.",
    cell_methods="time: minimum within days time: sum over days",
    compute=indices.tn90p,
)

tn10p = Tasmin(
    identifier="tn10p",
    units="days",
    standard_name="days_with_air_temperature_below_threshold",
    long_name="Number of days when Tmin < 10th percentile",
    description="{freq} number of days with minimum daily temperature below the 10th percentile."
    "The 10th percentile is to be computed for a 5 day moving window centered on each calendar day "
    "for a reference period.",
    cell_methods="time: minimum within days time: sum over days",
    compute=indices.tn10p,
)


degree_days_exceedance_date = Tas(
    identifier="degree_days_exceedance_date",
    units="",
    standard_name="day_of_year",
    long_name="Day of year when cumulative degree days exceed {sum_thresh}.",
    description="Day of year when the integral of degree days (tmean {op} {thresh})"
    " exceeds {sum_thresh}, the cumulative sum starts on {after_date}.",
    cell_methods="",
    compute=indices.degree_days_exceedance_date,
)


warm_spell_duration_index = Tasmax(
    identifier="warm_spell_duration_index",
    description="{freq} total number of days within spells of at least {window} days with tmax above the 90th daily percentile.",
    units="days",
    standard_name="number_of_days_with_air_temperature_above_threshold",
    cell_methods="time: sum over days",
    compute=indices.warm_spell_duration_index,
)


maximum_consecutive_warm_days = Tasmax(
    identifier="maximum_consecutive_warm_days",
    description="{freq} longest spell of consecutive days with Tmax above {thresh}.",
    units="days",
    standard_name="spell_length_of_days_with_air_temperature_above_threshold",
    cell_methods="time: maximum over days",
    compute=indices.maximum_consecutive_tx_days,
)

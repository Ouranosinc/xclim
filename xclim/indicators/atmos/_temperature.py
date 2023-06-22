"""Temperature indicator definitions."""
from __future__ import annotations

from xclim import indices
from xclim.core import cfchecks
from xclim.core.indicator import Daily, Indicator, ResamplingIndicatorWithIndexing
from xclim.core.utils import InputKind

__all__ = [
    "australian_hardiness_zones",
    "biologically_effective_degree_days",
    "cold_spell_days",
    "cold_spell_duration_index",
    "cold_spell_frequency",
    "cold_spell_max_length",
    "cold_spell_total_length",
    "consecutive_frost_days",
    "cool_night_index",
    "cooling_degree_days",
    "daily_freezethaw_cycles",
    "daily_temperature_range",
    "daily_temperature_range_variability",
    "degree_days_exceedance_date",
    "extreme_temperature_range",
    "fire_season",
    "first_day_tg_above",
    "first_day_tg_below",
    "first_day_tn_above",
    "first_day_tn_below",
    "first_day_tx_above",
    "first_day_tx_below",
    "freezethaw_spell_frequency",
    "freezethaw_spell_max_length",
    "freezethaw_spell_mean_length",
    "freezing_degree_days",
    "freshet_start",
    "frost_days",
    "frost_free_season_end",
    "frost_free_season_length",
    "frost_free_season_start",
    "frost_season_length",
    "growing_degree_days",
    "growing_season_end",
    "growing_season_length",
    "growing_season_start",
    "heat_wave_frequency",
    "heat_wave_index",
    "heat_wave_max_length",
    "heat_wave_total_length",
    "heating_degree_days",
    "hot_spell_frequency",
    "hot_spell_max_length",
    "hot_spell_total_length",
    "huglin_index",
    "ice_days",
    "last_spring_frost",
    "late_frost_days",
    "latitude_temperature_index",
    "max_daily_temperature_range",
    "maximum_consecutive_frost_free_days",
    "maximum_consecutive_warm_days",
    "tg10p",
    "tg90p",
    "tg_days_above",
    "tg_days_below",
    "tg_max",
    "tg_mean",
    "tg_min",
    "thawing_degree_days",
    "tn10p",
    "tn90p",
    "tn_days_above",
    "tn_days_below",
    "tn_max",
    "tn_mean",
    "tn_min",
    "tropical_nights",
    "tx10p",
    "tx90p",
    "tx_days_above",
    "tx_days_below",
    "tx_max",
    "tx_mean",
    "tx_min",
    "tx_tn_days_above",
    "usda_hardiness_zones",
    "warm_spell_duration_index",
]


# We need to declare the class here so that the `atmos` realm is automatically detected.
class Temp(Daily):
    """Indicators involving daily temperature."""


class TempWithIndexing(ResamplingIndicatorWithIndexing):
    """Indicators involving daily temperature and adding an indexing possibility."""

    src_freq = "D"


tn_days_above = TempWithIndexing(
    title="Number of days with minimum temperature above a given threshold",
    identifier="tn_days_above",
    units="days",
    standard_name="number_of_days_with_air_temperature_above_threshold",
    long_name="The number of days with minimum temperature above {thresh}",
    description="{freq} number of days where daily minimum temperature exceeds {thresh}.",
    abstract="The number of days with minimum temperature above a given threshold.",
    cell_methods="time: sum over days",
    compute=indices.tn_days_above,
    parameters={"op": {"default": ">"}},
)

tn_days_below = TempWithIndexing(
    title="Number of days with minimum temperature below a given threshold",
    identifier="tn_days_below",
    units="days",
    standard_name="number_of_days_with_air_temperature_below_threshold",
    long_name="The number of days with minimum temperature below {thresh}",
    description="{freq} number of days where daily minimum temperature is below {thresh}.",
    abstract="The number of days with minimum temperature below a given threshold.",
    cell_methods="time: sum over days",
    compute=indices.tn_days_below,
    parameters={"op": {"default": "<"}},
)

tg_days_above = TempWithIndexing(
    title="Number of days with mean temperature above a given threshold",
    identifier="tg_days_above",
    units="days",
    standard_name="number_of_days_with_air_temperature_above_threshold",
    long_name="The number of days with mean temperature above {thresh}",
    description="{freq} number of days where daily mean temperature exceeds {thresh}.",
    abstract="The number of days with mean temperature above a given threshold.",
    cell_methods="time: sum over days",
    compute=indices.tg_days_above,
    parameters={"op": {"default": ">"}},
)

tg_days_below = TempWithIndexing(
    title="Number of days with mean temperature below a given threshold",
    identifier="tg_days_below",
    units="days",
    standard_name="number_of_days_with_air_temperature_below_threshold",
    long_name="The number of days with mean temperature below {thresh}",
    description="{freq} number of days where daily mean temperature is below {thresh}.",
    abstract="The number of days with mean temperature below a given threshold.",
    cell_methods="time: sum over days",
    compute=indices.tg_days_below,
    parameters={"op": {"default": "<"}},
)

tx_days_above = TempWithIndexing(
    title="Number of days with maximum temperature above a given threshold",
    identifier="tx_days_above",
    units="days",
    standard_name="number_of_days_with_air_temperature_above_threshold",
    long_name="The number of days with maximum temperature above {thresh}",
    description="{freq} number of days where daily maximum temperature exceeds {thresh}.",
    abstract="The number of days with maximum temperature above a given threshold.",
    cell_methods="time: sum over days",
    compute=indices.tx_days_above,
    parameters={"op": {"default": ">"}},
)

tx_days_below = TempWithIndexing(
    title="Number of days with maximum temperature below a given threshold",
    identifier="tx_days_below",
    units="days",
    standard_name="number_of_days_with_air_temperature_below_threshold",
    long_name="The number of days with maximum temperature below {thresh}",
    description="{freq} number of days where daily max temperature is below {thresh}.",
    abstract="The number of days with maximum temperature below a given threshold.",
    cell_methods="time: sum over days",
    compute=indices.tx_days_below,
    parameters={"op": {"default": "<"}},
)

tx_tn_days_above = TempWithIndexing(
    title="Number of days with daily minimum and maximum temperatures exceeding thresholds",
    identifier="tx_tn_days_above",
    units="days",
    standard_name="number_of_days_with_air_temperature_above_threshold",
    long_name="Number of days with daily minimum above {thresh_tasmin} "
    "and daily maximum temperatures above {thresh_tasmax}",
    description="{freq} number of days where daily maximum temperature exceeds {thresh_tasmax} and minimum temperature "
    "exceeds {thresh_tasmin}.",
    abstract="Number of days with daily maximum and minimum temperatures above given thresholds.",
    cell_methods="",
    compute=indices.tx_tn_days_above,
)


heat_wave_frequency = Temp(
    title="Heat wave frequency",
    identifier="heat_wave_frequency",
    units="",
    standard_name="heat_wave_events",
    long_name="Total number of series of at least {window} consecutive days with daily minimum temperature above "
    "{thresh_tasmin} and daily maximum temperature above {thresh_tasmax}",
    description="{freq} number of heat wave events within a given period. A heat wave occurs when daily minimum and "
    "maximum temperatures exceed {thresh_tasmin} and {thresh_tasmax}, respectively, over at least {window} days.",
    abstract="Number of heat waves. A heat wave occurs when daily minimum and maximum temperatures exceed given "
    "thresholds for a number of days.",
    cell_methods="",
    keywords="health,",
    compute=indices.heat_wave_frequency,
)

heat_wave_max_length = Temp(
    title="Heat wave maximum length",
    identifier="heat_wave_max_length",
    units="days",
    standard_name="spell_length_of_days_with_air_temperature_above_threshold",
    long_name="Longest series of at least {window} consecutive days with daily minimum temperature above "
    "{thresh_tasmin} and daily maximum temperature above {thresh_tasmax}",
    description="{freq} maximum length of heat wave events occurring within a given period. "
    "A heat wave occurs when daily minimum and maximum temperatures exceed {thresh_tasmin} and {thresh_tasmax}, "
    "respectively, over at least {window} days.",
    abstract="Total duration of heat waves. A heat wave occurs when daily minimum and maximum temperatures exceed "
    "given thresholds for a number of days.",
    cell_methods="",
    keywords="health,",
    compute=indices.heat_wave_max_length,
)

heat_wave_total_length = Temp(
    title="Heat wave total length",
    identifier="heat_wave_total_length",
    units="days",
    standard_name="spell_length_of_days_with_air_temperature_above_threshold",
    long_name="Total length of events of at least {window} consecutive days with daily minimum temperature above "
    "{thresh_tasmin} and daily maximum temperature above {thresh_tasmax}",
    description="{freq} total length of heat wave events occurring within a given period. "
    "A heat wave occurs when daily minimum and maximum temperatures exceed {thresh_tasmin} and {thresh_tasmax}, "
    "respectively, over at least {window} days.",
    abstract="Maximum length of heat waves. A heat wave occurs when daily minimum and maximum temperatures exceed "
    "given thresholds for a number of days.",
    cell_methods="",
    keywords="health,",
    compute=indices.heat_wave_total_length,
)


heat_wave_index = Temp(
    title="Heat wave index",
    identifier="heat_wave_index",
    units="days",
    standard_name="heat_wave_index",
    long_name="Total number of days constituting events of at least {window} consecutive days "
    "with daily maximum temperature above {thresh}",
    description="{freq} total number of days that are part of a heatwave within a given period. "
    "A heat wave occurs when daily maximum temperatures exceed {thresh} over at least {window} days.",
    abstract="Number of days that constitute heatwave events. A heat wave occurs when daily minimum and maximum "
    "temperatures exceed given thresholds for a number of days.",
    cell_methods="",
    compute=indices.heat_wave_index,
)

hot_spell_frequency = Temp(
    title="Hot spell frequency",
    identifier="hot_spell_frequency",
    long_name="Number of hot periods of {window} day(s) or more, during which the temperature on a "
    "window of {window} day(s) is above {thresh}.",
    description="The {freq} number of hot periods of {window} day(s) or more, during which the temperature on a "
    "window of {window} day(s) is above {thresh}.",
    abstract="The frequency of hot periods of `N` days or more, during which the temperature "
    "over a given time window of days is above a given threshold.",
    units="",
    cell_methods="",
    compute=indices.hot_spell_frequency,
)

hot_spell_max_length = Temp(
    title="Hot spell maximum length",
    identifier="hot_spell_max_length",
    long_name="Maximum consecutive number of days in a hot period of {window} day(s) or more, during which the "
    "temperature within windows of {window} day(s) is above {thresh}.",
    description="The maximum {freq} number of consecutive days in a hot period of {window} day(s) or more"
    ", during which the precipitation within windows of {window} day(s) is above {thresh}.",
    abstract="The maximum length of a hot period of `N` days or more, during which the "
    "temperature over a given time window of days is above a given threshold.",
    units="days",
    cell_methods="",
    compute=indices.hot_spell_max_length,
)

hot_spell_total_length = Temp(
    title="Hot spell total length",
    identifier="hot_spell_total_length",
    long_name="Number of days in hot periods of {window} day(s) or more, during which the"
    "temperature within windows of {window} day(s) is above {thresh}.",
    description="The {freq} number of days in hot periods of {window} day(s) or more, during which the "
    "temperature within windows of {window} day(s) is above {thresh}.",
    abstract="The total length of hot periods of `N` days or more, during which the "
    "temperature over a given time window of days is above a given threshold.",
    units="days",
    cell_methods="",
    compute=indices.hot_spell_total_length,
)

tg_mean = TempWithIndexing(
    title="Mean temperature",
    identifier="tg_mean",
    units="K",
    standard_name="air_temperature",
    long_name="Mean daily mean temperature",
    description="{freq} mean of daily mean temperature.",
    abstract="Mean of daily mean temperature.",
    cell_methods="time: mean over days",
    compute=indices.tg_mean,
)

tg_max = TempWithIndexing(
    title="Maximum of mean temperature",
    identifier="tg_max",
    units="K",
    standard_name="air_temperature",
    long_name="Maximum daily mean temperature",
    description="{freq} maximum of daily mean temperature.",
    abstract="Maximum of daily mean temperature.",
    cell_methods="time: maximum over days",
    compute=indices.tg_max,
)

tg_min = TempWithIndexing(
    title="Minimum of mean temperature",
    identifier="tg_min",
    units="K",
    standard_name="air_temperature",
    long_name="Minimum daily mean temperature",
    description="{freq} minimum of daily mean temperature.",
    abstract="Minimum of daily mean temperature.",
    cell_methods="time: minimum over days",
    compute=indices.tg_min,
)

tx_mean = TempWithIndexing(
    title="Mean of maximum temperature",
    identifier="tx_mean",
    units="K",
    standard_name="air_temperature",
    long_name="Mean daily maximum temperature",
    description="{freq} mean of daily maximum temperature.",
    abstract="Mean of daily maximum temperature.",
    cell_methods="time: mean over days",
    compute=indices.tx_mean,
)

tx_max = TempWithIndexing(
    title="Maximum temperature",
    identifier="tx_max",
    units="K",
    standard_name="air_temperature",
    long_name="Maximum daily maximum temperature",
    description="{freq} maximum of daily maximum temperature.",
    abstract="Maximum of daily maximum temperature.",
    cell_methods="time: maximum over days",
    compute=indices.tx_max,
)

tx_min = TempWithIndexing(
    title="Minimum of maximum temperature",
    identifier="tx_min",
    units="K",
    standard_name="air_temperature",
    long_name="Minimum daily maximum temperature",
    description="{freq} minimum of daily maximum temperature.",
    abstract="Minimum of daily maximum temperature.",
    cell_methods="time: minimum over days",
    compute=indices.tx_min,
)

tn_mean = TempWithIndexing(
    title="Mean of minimum temperature",
    identifier="tn_mean",
    units="K",
    standard_name="air_temperature",
    long_name="Mean daily minimum temperature",
    description="{freq} mean of daily minimum temperature.",
    abstract="Mean of daily minimum temperature.",
    cell_methods="time: mean over days",
    compute=indices.tn_mean,
)

tn_max = TempWithIndexing(
    title="Maximum of minimum temperature",
    identifier="tn_max",
    units="K",
    standard_name="air_temperature",
    long_name="Maximum daily minimum temperature",
    description="{freq} maximum of daily minimum temperature.",
    abstract="Maximum of daily minimum temperature.",
    cell_methods="time: maximum over days",
    compute=indices.tn_max,
)

tn_min = TempWithIndexing(
    title="Minimum temperature",
    identifier="tn_min",
    units="K",
    standard_name="air_temperature",
    long_name="Minimum daily minimum temperature",
    description="{freq} minimum of daily minimum temperature.",
    abstract="Minimum of daily minimum temperature.",
    cell_methods="time: minimum over days",
    compute=indices.tn_min,
)

daily_temperature_range = TempWithIndexing(
    title="Mean of daily temperature range",
    identifier="dtr",
    units="K",
    standard_name="air_temperature",
    long_name="Mean diurnal temperature range",
    description="{freq} mean diurnal temperature range.",
    cell_methods="time range within days time: mean over days",
    abstract="The average difference between the daily maximum and minimum temperatures.",
    compute=indices.daily_temperature_range,
    parameters={"op": "mean"},
)

max_daily_temperature_range = TempWithIndexing(
    title="Maximum of daily temperature range",
    identifier="dtrmax",
    units="K",
    standard_name="air_temperature",
    long_name="Maximum diurnal temperature range",
    description="{freq} maximum diurnal temperature range.",
    cell_methods="time range within days time: max over days",
    abstract="The maximum difference between the daily maximum and minimum temperatures.",
    compute=indices.daily_temperature_range,
    parameters={"op": "max"},
)

daily_temperature_range_variability = TempWithIndexing(
    title="Variability of daily temperature range",
    identifier="dtrvar",
    units="K",
    standard_name="air_temperature",
    long_name="Mean diurnal temperature range variability",
    description="{freq} mean diurnal temperature range variability, defined as the average day-to-day variation "
    "in daily temperature range for the given time period.",
    abstract="The average day-to-day variation in daily temperature range.",
    cell_methods="time range within days time: difference over days time: mean over days",
    compute=indices.daily_temperature_range_variability,
)

extreme_temperature_range = TempWithIndexing(
    title="Extreme temperature range",
    identifier="etr",
    units="K",
    standard_name="air_temperature",
    long_name="Intra-period extreme temperature range",
    description="{freq} range between the maximum of daily maximum temperature and the minimum of daily"
    "minimum temperature.",
    abstract="The maximum of the maximum temperature minus the minimum of the minimum temperature.",
    compute=indices.extreme_temperature_range,
)

cold_spell_duration_index = Temp(
    title="Cold Spell Duration Index (CSDI)",
    identifier="cold_spell_duration_index",
    var_name="csdi_{window}",
    units="days",
    standard_name="cold_spell_duration_index",
    long_name="Total number of days constituting events of at least {window} consecutive days "
    "where the daily minimum temperature is below the {tasmin_per_thresh}th percentile",
    description="{freq} number of days with at least {window} consecutive days where the daily minimum temperature "
    "is below the {tasmin_per_thresh}th percentile. A {tasmin_per_window} day(s) window, centred on each calendar day "
    "in the {tasmin_per_period} period, is used to compute the {tasmin_per_thresh}th percentile(s).",
    abstract="Number of days part of a percentile-defined cold spell. A cold spell occurs when the daily minimum "
    "temperature is below a given percentile for a given number of consecutive days.",
    cell_methods="",
    compute=indices.cold_spell_duration_index,
)

cold_spell_days = Temp(
    title="Cold spell days",
    identifier="cold_spell_days",
    units="days",
    standard_name="cold_spell_days",
    long_name="Total number of days constituting events of at least {window} consecutive days "
    "where the mean daily temperature is below {thresh}",
    description="{freq} number of days that are part of a cold spell. A cold spell is defined as {window} or more "
    "consecutive days with mean daily temperature below {thresh}.",
    abstract="The number of days that are part of a cold spell. A cold spell is defined as a minimum number of "
    "consecutive days with mean daily temperature below a given threshold.",
    cell_methods="",
    compute=indices.cold_spell_days,
)

cold_spell_frequency = Temp(
    title="Cold spell frequency",
    identifier="cold_spell_frequency",
    long_name="Number of cold periods of {window} day(s) or more, during which the temperature on a "
    "window of {window} day(s) is below {thresh}.",
    description="The {freq} number of cold periods of {window} day(s) or more, during which the temperature on a "
    "window of {window} day(s) is below {thresh}.",
    abstract="The frequency of cold periods of `N` days or more, during which the temperature "
    "over a given time window of days is below a given threshold.",
    units="",
    cell_methods="",
    compute=indices.cold_spell_frequency,
)

cold_spell_max_length = Temp(
    title="Cold spell maximum length",
    identifier="cold_spell_max_length",
    long_name="Maximum consecutive number of days in a cold period of {window} day(s) or more, during which the "
    "temperature within windows of {window} day(s) is under {thresh}.",
    description="The maximum {freq} number of consecutive days in a cold period of {window} day(s) or more"
    ", during which the precipitation within windows of {window} day(s) is under {thresh}.",
    abstract="The maximum length of a cold period of `N` days or more, during which the "
    "temperature over a given time window of days is below a given threshold.",
    units="days",
    cell_methods="",
    compute=indices.cold_spell_max_length,
)

cold_spell_total_length = Temp(
    title="Cold spell total length",
    identifier="cold_spell_total_length",
    long_name="Number of days in cold periods of {window} day(s) or more, during which the"
    "temperature within windows of {window} day(s) is under {thresh}.",
    description="The {freq} number of days in cold periods of {window} day(s) or more, during which the "
    "temperature within windows of {window} day(s) is under {thresh}.",
    abstract="The total length of cold periods of `N` days or more, during which the "
    "temperature over a given time window of days is below a given threshold.",
    units="days",
    cell_methods="",
    compute=indices.cold_spell_total_length,
)

cool_night_index = Temp(
    title="Cool night index",
    identifier="cool_night_index",
    units="degC",
    long_name="Mean minimum temperature in late summer",
    description="Mean minimum temperature for September (Northern hemisphere) or March (Southern hemisphere).",
    abstract="A night coolness variable which takes into account the mean minimum night temperatures during the "
    "month when ripening usually occurs beyond the ripening period.",
    cell_methods="time: mean over days",
    src_freq=["D", "M"],
    compute=indices.cool_night_index,
)

daily_freezethaw_cycles = TempWithIndexing(
    title="Daily freeze-thaw cycles",
    identifier="dlyfrzthw",
    units="days",
    long_name="Number of days where maximum daily temperatures are above {thresh_tasmax} "
    "and minimum daily temperatures are at or below {thresh_tasmin}",
    description="{freq} number of days with a diurnal freeze-thaw cycle, where maximum daily temperatures are above "
    "{thresh_tasmax} and minimum daily temperatures are at or below {thresh_tasmin}.",
    abstract="The number of days with a freeze-thaw cycle. A freeze-thaw cycle is defined as a day where maximum daily "
    "temperature is above a given threshold and minimum daily temperature is at or below a given threshold, "
    "usually 0°C for both.",
    cell_methods="",
    compute=indices.multiday_temperature_swing,
    parameters={
        "op": "sum",
        "window": 1,
        "thresh_tasmax": {"default": "0 degC"},
        "thresh_tasmin": {"default": "0 degC"},
        "op_tasmax": {"default": ">"},
        "op_tasmin": {"default": "<="},
    },
)


freezethaw_spell_frequency = Temp(
    title="Freeze-thaw spell frequency",
    identifier="freezethaw_spell_frequency",
    units="days",
    long_name="Frequency of events where maximum daily temperatures are above {thresh_tasmax} "
    "and minimum daily temperatures are at or below {thresh_tasmin} for at least {window} consecutive day(s).",
    description="{freq} number of freeze-thaw spells, where maximum daily temperatures are above {thresh_tasmax} "
    "and minimum daily temperatures are at or below {thresh_tasmin} for at least {window} consecutive day(s).",
    abstract="Frequency of daily freeze-thaw spells. A freeze-thaw spell is defined as a number of consecutive days "
    "where maximum daily temperatures are above a given threshold and minimum daily temperatures are at or below a "
    "given threshold, usually 0°C for both.",
    cell_methods="",
    compute=indices.multiday_temperature_swing,
    parameters={
        "op": "count",
        "thresh_tasmax": {"default": "0 degC"},
        "thresh_tasmin": {"default": "0 degC"},
        "op_tasmax": {"default": ">"},
        "op_tasmin": {"default": "<="},
    },
)


freezethaw_spell_mean_length = Temp(
    title="Freeze-thaw spell mean length",
    identifier="freezethaw_spell_mean_length",
    units="days",
    long_name="Average length of events where maximum daily temperatures are above {thresh_tasmax} "
    "and minimum daily temperatures are at or below {thresh_tasmin} for at least {window} consecutive day(s).",
    description="{freq} average length of freeze-thaw spells, where maximum daily temperatures are above "
    "{thresh_tasmax} and minimum daily temperatures are at or below {thresh_tasmin} for at least {window} consecutive "
    "day(s).",
    abstract="Average length of daily freeze-thaw spells. A freeze-thaw spell is defined as a number of consecutive "
    "days where maximum daily temperatures are above a given threshold and minimum daily temperatures are at or below "
    "a given threshold, usually 0°C for both.",
    cell_methods="",
    compute=indices.multiday_temperature_swing,
    parameters={
        "op": "mean",
        "thresh_tasmax": {"default": "0 degC"},
        "thresh_tasmin": {"default": "0 degC"},
        "op_tasmax": ">",
        "op_tasmin": "<=",
    },
)


freezethaw_spell_max_length = Temp(
    title="Maximal length of freeze-thaw spells",
    identifier="freezethaw_spell_max_length",
    units="days",
    long_name="Maximal length of events where maximum daily temperatures are above {thresh_tasmax} "
    "and minimum daily temperatures are at or below {thresh_tasmin} for at least {window} consecutive day(s).",
    description="{freq} maximal length of freeze-thaw spells, where maximum daily temperatures are above "
    "{thresh_tasmax} and minimum daily temperatures are at or below {thresh_tasmin} for at least {window} consecutive "
    "day(s).",
    abstract="Maximal length of daily freeze-thaw spells. A freeze-thaw spell is defined as a number of consecutive "
    "days where maximum daily temperatures are above a given threshold and minimum daily temperatures are at or below "
    "a threshold, usually 0°C for both.",
    cell_methods="",
    compute=indices.multiday_temperature_swing,
    parameters={
        "op": "max",
        "thresh_tasmax": {"default": "0 degC"},
        "thresh_tasmin": {"default": "0 degC"},
        "op_tasmax": {"default": ">"},
        "op_tasmin": {"default": "<="},
    },
)


cooling_degree_days = TempWithIndexing(
    title="Cooling degree days",
    identifier="cooling_degree_days",
    units="K days",
    standard_name="integral_of_air_temperature_excess_wrt_time",
    long_name="Cumulative sum of temperature degrees for mean daily temperature above {thresh}",
    description="{freq} cumulative cooling degree days (mean temperature above {thresh}).",
    abstract="The cumulative degree days for days when the mean daily temperature is above a given threshold and "
    "buildings must be air conditioned.",
    cell_methods="time: sum over days",
    compute=indices.cooling_degree_days,
    parameters={"thresh": {"default": "18.0 degC"}},
)

heating_degree_days = TempWithIndexing(
    title="Heating degree days",
    identifier="heating_degree_days",
    units="K days",
    standard_name="integral_of_air_temperature_deficit_wrt_time",
    long_name="Cumulative sum of temperature degrees for mean daily temperature below {thresh}",
    description="{freq} cumulative heating degree days (mean temperature below {thresh}).",
    abstract="The cumulative degree days for days when the mean daily temperature is below a given threshold and "
    "buildings must be heated.",
    cell_methods="time: sum over days",
    compute=indices.heating_degree_days,
    parameters={"thresh": {"default": "17.0 degC"}},
)

growing_degree_days = TempWithIndexing(
    title="Growing degree days",
    identifier="growing_degree_days",
    units="K days",
    standard_name="integral_of_air_temperature_excess_wrt_time",
    long_name="Cumulative sum of temperature degrees for mean daily temperature above {thresh}",
    description="{freq} growing degree days (mean temperature above {thresh}).",
    abstract="The cumulative degree days for days when the average temperature is above a given threshold.",
    cell_methods="time: sum over days",
    compute=indices.growing_degree_days,
    parameters={"thresh": {"default": "4.0 degC"}},
)

freezing_degree_days = TempWithIndexing(
    title="Freezing degree days",
    identifier="freezing_degree_days",
    units="K days",
    standard_name="integral_of_air_temperature_deficit_wrt_time",
    long_name="Cumulative sum of temperature degrees for mean daily temperature below {thresh}",
    description="{freq} freezing degree days (mean temperature below {thresh}).",
    abstract="The cumulative degree days for days when the average temperature is below a given threshold, "
    "typically 0°C.",
    cell_methods="time: sum over days",
    compute=indices.heating_degree_days,
    parameters={"thresh": {"default": "0 degC"}},
)

thawing_degree_days = TempWithIndexing(
    title="Thawing degree days",
    identifier="thawing_degree_days",
    units="K days",
    standard_name="integral_of_air_temperature_excess_wrt_time",
    long_name="Cumulative sum of temperature degrees for mean daily temperature above {thresh}",
    description="{freq} thawing degree days (mean temperature above {thresh}).",
    abstract="The cumulative degree days for days when the average temperature is above a given threshold, "
    "typically 0°C.",
    cell_methods="time: sum over days",
    compute=indices.growing_degree_days,
    parameters={"thresh": {"default": "0 degC"}},
)

freshet_start = Temp(
    title="Day of year of spring freshet start",
    identifier="freshet_start",
    units="",
    standard_name="day_of_year",
    long_name="First day where temperature threshold of {thresh} is exceeded for at least {window} days",
    description="Day of year of the spring freshet start, defined as the first day a temperature threshold of {thresh} "
    "is exceeded for at least {window} days.",
    abstract="Day of year of the spring freshet start, defined as the first day when the temperature exceeds a certain "
    "threshold for a given number of consecutive days.",
    compute=indices.first_day_temperature_above,
    parameters={"thresh": {"default": "0 degC"}, "window": {"default": 5}},
)

frost_days = Temp(
    title="Frost days",
    identifier="frost_days",
    units="days",
    standard_name="days_with_air_temperature_below_threshold",
    long_name="Number of days where the daily minimum temperature is below {thresh}",
    description="{freq} number of days where the daily minimum temperature is below {thresh}.",
    abstract="Number of days where the daily minimum temperature is below a given threshold.",
    cell_methods="time: sum over days",
    compute=indices.frost_days,
)

frost_season_length = Temp(
    title="Frost season length",
    identifier="frost_season_length",
    units="days",
    standard_name="days_with_air_temperature_below_threshold",
    long_name="Number of days between the first occurrence of at least {window} consecutive days with "
    "minimum daily temperature below {thresh} and the first occurrence of at least {window} consecutive days with "
    "minimum daily temperature at or above {thresh} after {mid_date}",
    description="{freq} number of days between the first occurrence of at least {window} consecutive days with "
    "minimum daily temperature below {thresh} and the first occurrence of at least {window} consecutive days with "
    "minimum daily temperature at or above {thresh} after {mid_date}.",
    abstract="Duration of the freezing season, defined as the period when the daily minimum temperature is below 0°C "
    "without a thawing window of days, with the thaw occurring after a median calendar date.",
    cell_methods="time: sum over days",
    compute=indices.frost_season_length,
    parameters={"thresh": {"default": "0 degC"}},
)

last_spring_frost = Temp(
    title="Last spring frost",
    identifier="last_spring_frost",
    units="",
    standard_name="day_of_year",
    long_name="Last day of minimum daily temperature below a threshold of {thresh} "
    "for at least {window} days before a given date ({before_date})",
    description="Day of year of last spring frost, defined as the last day a minimum temperature "
    "remains below a threshold of {thresh} for at least {window} days before a given date ({before_date}).",
    abstract="The last day when minimum temperature is below a given threshold for a certain number of days, "
    "limited by a final calendar date.",
    cell_methods="",
    compute=indices.last_spring_frost,
    parameters={"before_date": {"default": "07-01"}},
)

first_day_tn_below = Temp(
    identifier="first_day_tn_below",
    units="",
    standard_name="day_of_year",
    long_name="First day of year with a period of at least {window} days of minimum temperature below {thresh}",
    description="First day of year with minimum temperature below {thresh} for at least {window} days.",
    compute=indices.first_day_temperature_below,
    input=dict(tas="tasmin"),
    parameters=dict(
        thresh={"default": "0 degC"},
        after_date={"default": "07-01"},
        op={"default": "<"},
    ),
)

first_day_tg_below = Temp(
    identifier="first_day_tg_below",
    units="",
    standard_name="day_of_year",
    long_name="First day of year with a period of at least {window} days of mean temperature below {thresh}",
    description="First day of year with mean temperature below {thresh} for at least {window} days.",
    compute=indices.first_day_temperature_below,
    parameters=dict(
        thresh={"default": "0 degC"},
        after_date={"default": "07-01"},
        op={"default": "<"},
    ),
)

first_day_tx_below = Temp(
    identifier="first_day_tx_below",
    units="",
    standard_name="day_of_year",
    long_name="First day of year with a period of at least {window} days of maximum temperature below {thresh}",
    description="First day of year with maximum temperature below {thresh} for at least {window} days.",
    compute=indices.first_day_temperature_below,
    input=dict(tas="tasmax"),
    parameters=dict(
        thresh={"default": "0 degC"},
        after_date={"default": "07-01"},
        op={"default": "<"},
    ),
)

first_day_tn_above = Temp(
    identifier="first_day_tn_above",
    units="",
    standard_name="day_of_year",
    long_name="First day of year with a period of at least {window} days of minimum temperature above {thresh}",
    description="First day of year with minimum temperature above {thresh} for at least {window} days.",
    compute=indices.first_day_temperature_above,
    input=dict(tas="tasmin"),
    parameters=dict(
        thresh={"default": "0 degC"},
        after_date={"default": "01-01"},
        op={"default": ">"},
    ),
)


first_day_tg_above = Temp(
    identifier="first_day_tg_above",
    units="",
    standard_name="day_of_year",
    long_name="First day of year with a period of at least {window} days of mean temperature above {thresh}",
    description="First day of year with mean temperature above {thresh} for at least {window} days.",
    compute=indices.first_day_temperature_above,
    parameters=dict(
        thresh={"default": "0 degC"},
        after_date={"default": "01-01"},
        op={"default": ">"},
    ),
)

first_day_tx_above = Temp(
    identifier="first_day_tx_above",
    units="",
    standard_name="day_of_year",
    long_name="First day of year with a period of at least {window} days of maximum temperature above {thresh}",
    description="First day of year with maximum temperature above {thresh} for at least {window} days.",
    compute=indices.first_day_temperature_above,
    input=dict(tas="tasmax"),
    parameters=dict(
        thresh={"default": "0 degC"},
        after_date={"default": "01-01"},
        op={"default": ">"},
    ),
)

ice_days = TempWithIndexing(
    title="Ice days",
    identifier="ice_days",
    standard_name="days_with_air_temperature_below_threshold",
    units="days",
    long_name="Number of days with maximum daily temperature below {thresh}",
    description="{freq} number of days where the maximum daily temperature is below {thresh}.",
    abstract="Number of days where the daily maximum temperature is below 0°C",
    cell_methods="time: sum over days",
    compute=indices.ice_days,
)

consecutive_frost_days = Temp(
    title="Consecutive frost days",
    identifier="consecutive_frost_days",
    units="days",
    standard_name="spell_length_of_days_with_air_temperature_below_threshold",
    long_name="Maximum number of consecutive days where minimum daily temperature is below {thresh}",
    description="{freq} maximum number of consecutive days where minimum daily temperature is below {thresh}.",
    abstract="Maximum number of consecutive days where the daily minimum temperature is below 0°C",
    cell_methods="time: maximum over days",
    compute=indices.maximum_consecutive_frost_days,
    parameters={"thresh": {"default": "0 degC"}},
)

frost_free_season_length = Temp(
    title="Frost free season length",
    identifier="frost_free_season_length",
    units="days",
    standard_name="days_with_air_temperature_above_threshold",
    long_name="Number of days between the first occurrence of at least {window} consecutive days "
    "with minimum daily temperature at or above {thresh} and the first occurrence of at least "
    "{window} consecutive days with minimum daily temperature below {thresh} after {mid_date}",
    description="{freq} number of days between the first occurrence of at least {window} consecutive days "
    "with minimum daily temperature at or above {thresh} and the first occurrence of at least "
    "{window} consecutive days with minimum daily temperature below {thresh} after {mid_date}.",
    abstract="Duration of the frost free season, defined as the period when the minimum daily temperature is above 0°C "
    "without a freezing window of `N` days, with freezing occurring after a median calendar date.",
    cell_methods="time: sum over days",
    compute=indices.frost_free_season_length,
    parameters={"thresh": {"default": "0 degC"}},
)

frost_free_season_start = Temp(
    title="Frost free season start",
    identifier="frost_free_season_start",
    units="",
    standard_name="day_of_year",
    long_name="First day following a period of {window} days with minimum daily temperature at or above {thresh}",
    description="Day of the year of the beginning of the frost-free season, defined as the {window}th consecutive day "
    "when minimum daily temperature exceeds {thresh}.",
    abstract="First day when minimum daily temperature exceeds a given threshold for a given number of consecutive days",
    compute=indices.frost_free_season_start,
    parameters={"thresh": {"default": "0 degC"}},
)

frost_free_season_end = Temp(
    title="Frost free season end",
    identifier="frost_free_season_end",
    units="",
    standard_name="day_of_year",
    long_name="First day, after {mid_date}, following a period of {window} days "
    "with minimum daily temperature below {thresh}",
    description="Day of the year of the end of the frost-free season, defined as the interval between the first set of "
    "{window} days when the minimum daily temperature is at or above {thresh} "
    "and the first set (after {mid_date}) of {window} days when it is below {thresh}.",
    abstract="First day when the temperature is below a given threshold for a given number of consecutive days after "
    "a median calendar date.",
    cell_methods="",
    compute=indices.frost_free_season_end,
    parameters={"thresh": {"default": "0 degC"}},
)

maximum_consecutive_frost_free_days = Temp(
    title="Maximum consecutive frost free days",
    # FIXME: shouldn't this be `maximum_`? Breaking changes needed.
    identifier="consecutive_frost_free_days",
    units="days",
    standard_name="spell_length_of_days_with_air_temperature_above_threshold",
    long_name="Maximum number of consecutive days with minimum temperature at or above {thresh}",
    description="{freq} maximum number of consecutive days with minimum daily temperature at or above {thresh}.",
    abstract="Maximum number of consecutive frost-free days: where the daily minimum temperature is above "
    "or equal to 0°C",
    cell_methods="time: maximum over days",
    compute=indices.maximum_consecutive_frost_free_days,
    parameters={"thresh": {"default": "0 degC"}},
)

growing_season_start = Temp(
    title="Growing season start",
    identifier="growing_season_start",
    units="",
    standard_name="day_of_year",
    long_name="First day of the first series of {window} days with mean daily temperature above or equal to {thresh}",
    description="Day of the year marking the beginning of the growing season, defined as the first day of the first "
    "series of {window} days with mean daily temperature above or equal to {thresh}.",
    abstract="The first day when the temperature exceeds a certain threshold for a given number of consecutive days.",
    cell_methods="",
    compute=indices.growing_season_start,
    parameters={"thresh": {"default": "5.0 degC"}},
)

growing_season_length = Temp(
    title="Growing season length",
    identifier="growing_season_length",
    units="days",
    standard_name="growing_season_length",
    long_name="Number of days between the first occurrence of at least {window} consecutive days with mean "
    "daily temperature over {thresh} and the first occurrence of at least {window} consecutive days with "
    "mean daily temperature below {thresh}, occurring after {mid_date}",
    description="{freq} number of days between the first occurrence of at least {window} consecutive days "
    "with mean daily temperature over {thresh} and the first occurrence of at least {window} consecutive days with "
    "mean daily temperature below {thresh}, occurring after {mid_date}.",
    abstract="Number of days between the first occurrence of a series of days with a daily average temperature above a "
    "threshold and the first occurrence of a series of days with a daily average temperature below that same "
    "threshold, occurring after a given calendar date.",
    cell_methods="",
    compute=indices.growing_season_length,
    parameters={"thresh": {"default": "5.0 degC"}, "mid_date": {"default": "07-01"}},
)

growing_season_end = Temp(
    title="Growing season end",
    identifier="growing_season_end",
    units="",
    standard_name="day_of_year",
    long_name="First day of the first series of {window} days with mean daily temperature below {thresh}, "
    "occurring after {mid_date}",
    description="Day of year of end of growing season, defined as the first day of consistent inferior threshold "
    "temperature of {thresh} after a run of {window} days superior to threshold temperature, occurring after "
    "{mid_date}.",
    abstract="The first day when the temperature is below a certain threshold for a certain number of consecutive days "
    "after a given calendar date.",
    cell_methods="",
    compute=indices.growing_season_end,
    parameters={"thresh": {"default": "5.0 degC"}, "mid_date": {"default": "07-01"}},
)

tropical_nights = TempWithIndexing(
    title="Tropical nights",
    identifier="tropical_nights",
    units="days",
    standard_name="number_of_days_with_air_temperature_above_threshold",
    long_name="Number of days with minimum daily temperature above {thresh}",
    description="{freq} number of Tropical Nights, defined as days with minimum daily temperature above {thresh}.",
    abstract="Number of days where minimum temperature is above a given threshold.",
    cell_methods="time: sum over days",
    compute=indices.tn_days_above,
    parameters={"thresh": {"default": "20.0 degC"}},
)

tg90p = TempWithIndexing(
    title="Days with mean temperature above the 90th percentile",
    identifier="tg90p",
    units="days",
    standard_name="days_with_air_temperature_above_threshold",
    long_name="Number of days with mean temperature above the 90th percentile",
    description="{freq} number of days with mean temperature above the 90th percentile. "
    "A {tas_per_window} day(s) window, centered on each calendar day in the {tas_per_period} period, "
    "is used to compute the 90th percentile.",
    abstract="Number of days with mean temperature above the 90th percentile.",
    cell_methods="time: sum over days",
    compute=indices.tg90p,
)

tg10p = TempWithIndexing(
    title="Days with mean temperature below the 10th percentile",
    identifier="tg10p",
    units="days",
    standard_name="days_with_air_temperature_below_threshold",
    long_name="Number of days with mean temperature below the 10th percentile",
    description="{freq} number of days with mean temperature below the 10th percentile. "
    "A {tas_per_window} day(s) window, centered on each calendar day in the {tas_per_period} period, "
    "is used to compute the 10th percentile.",
    abstract="Number of days with mean temperature below the 10th percentile.",
    cell_methods="time: sum over days",
    compute=indices.tg10p,
)

tx90p = TempWithIndexing(
    title="Days with maximum temperature above the 90th percentile",
    identifier="tx90p",
    units="days",
    standard_name="days_with_air_temperature_above_threshold",
    long_name="Number of days with maximum temperature above the 90th percentile",
    description="{freq} number of days with maximum temperature above the 90th percentile. "
    "A {tasmax_per_window} day(s) window, centered on each calendar day in the {tasmax_per_period} period, "
    "is used to compute the 90th percentile.",
    abstract="Number of days with maximum temperature above the 90th percentile.",
    cell_methods="time: sum over days",
    compute=indices.tx90p,
)

tx10p = TempWithIndexing(
    title="Days with maximum temperature below the 10th percentile",
    identifier="tx10p",
    units="days",
    standard_name="days_with_air_temperature_below_threshold",
    long_name="Number of days with maximum temperature below the 10th percentile",
    description="{freq} number of days with maximum temperature below the 10th percentile. "
    "A {tasmax_per_window} day(s) window, centered on each calendar day in the {tasmax_per_period} period, "
    "is used to compute the 10th percentile.",
    abstract="Number of days with maximum temperature below the 10th percentile.",
    cell_methods="time: sum over days",
    compute=indices.tx10p,
)

tn90p = TempWithIndexing(
    title="Days with minimum temperature above the 90th percentile",
    identifier="tn90p",
    units="days",
    standard_name="days_with_air_temperature_above_threshold",
    long_name="Number of days with minimum temperature above the 90th percentile",
    description="{freq} number of days with minimum temperature above the 90th percentile. "
    "A {tasmin_per_window} day(s) window, centered on each calendar day in the {tasmin_per_period} period, "
    "is used to compute the 90th percentile.",
    abstract="Number of days with minimum temperature above the 90th percentile.",
    cell_methods="time: sum over days",
    compute=indices.tn90p,
)

tn10p = TempWithIndexing(
    title="Days with minimum temperature below the 10th percentile",
    identifier="tn10p",
    units="days",
    standard_name="days_with_air_temperature_below_threshold",
    long_name="Number of days with minimum temperature below the 10th percentile",
    description="{freq} number of days with minimum temperature below the 10th percentile. "
    "A {tasmin_per_window} day(s) window, centered on each calendar day in the {tasmin_per_period} period, "
    "is used to compute the 10th percentile.",
    abstract="Number of days with minimum temperature below the 10th percentile.",
    cell_methods="time: sum over days",
    compute=indices.tn10p,
)


degree_days_exceedance_date = Temp(
    title="Degree day exceedance date",
    identifier="degree_days_exceedance_date",
    units="",
    standard_name="day_of_year",
    long_name="Day of year when the integral of mean daily temperature {op} {thresh} exceeds {sum_thresh}",
    description=lambda **kws: "Day of year when the integral of degree days (mean daily temperature {op} {thresh}) "
    "exceeds {sum_thresh}"
    + (
        ", with the cumulative sum starting from {after_date}."
        if kws["after_date"] is not None
        else "."
    ),
    abstract="The day of the year when the sum of degree days exceeds a threshold, occurring after a given date. "
    "Degree days are calculated above or below a given temperature threshold.",
    cell_methods="",
    compute=indices.degree_days_exceedance_date,
)


warm_spell_duration_index = Temp(
    title="Warm spell duration index",
    identifier="warm_spell_duration_index",
    units="days",
    standard_name="number_of_days_with_air_temperature_above_threshold",
    long_name="Number of days with at least {window} consecutive days where the maximum daily temperature is above "
    "the {tasmax_per_thresh}th percentile(s)",
    description="{freq} number of days with at least {window} consecutive days where the maximum daily temperature is "
    "above the {tasmax_per_thresh}th percentile(s). A {tasmax_per_window} day(s) window, centred on each calendar day "
    "in the {tasmax_per_period} period, is used to compute the {tasmax_per_thresh}th percentile(s).",
    abstract="Number of days part of a percentile-defined warm spell. A warm spell occurs when the maximum daily "
    "temperature is above a given percentile for a given number of consecutive days.",
    cell_methods="time: sum over days",
    compute=indices.warm_spell_duration_index,
)


maximum_consecutive_warm_days = Temp(
    title="Maximum consecutive warm days",
    identifier="maximum_consecutive_warm_days",
    units="days",
    standard_name="spell_length_of_days_with_air_temperature_above_threshold",
    long_name="Maximum number of consecutive days with maximum daily temperature above {thresh}",
    description="{freq} longest spell of consecutive days with maximum daily temperature above {thresh}.",
    abstract="Maximum number of consecutive days where the maximum daily temperature exceeds a certain threshold.",
    cell_methods="time: maximum over days",
    compute=indices.maximum_consecutive_tx_days,
)


class FireSeasonBase(Indicator):
    """Special Indicator class for FireSeason that accepts any tas[min/max] and optional snd and is not resampling."""

    def cfcheck(self, tas, snd=None):
        cfchecks.check_valid(tas, "standard_name", "air_temperature")
        cfchecks.cfcheck_from_name("snd", snd)


fire_season = FireSeasonBase(
    identifier="fire_season",
    description="Fire season mask, computed with method {method}.",
    units="",
    compute=indices.fire_season,
)


huglin_index = Temp(
    title="Huglin heliothermal index",
    identifier="huglin_index",
    units="",
    long_name="Integral of mean daily temperature above {thresh} multiplied by day-length coefficient with {method} "
    "method for days between {start_date} and {end_date}",
    description="Heat-summation index for agroclimatic suitability estimation, developed specifically for viticulture, "
    "computed with {method} formula (Summation of ((Tn + Tx)/2 - {thresh}) * k), where coefficient `k` is a "
    "latitude-based day-length for days between {start_date} and {end_date}.",
    abstract="Heat-summation index for agroclimatic suitability estimation, developed specifically for viticulture. "
    "Considers daily minimum and maximum temperature with a given base threshold, typically between 1 April and 30"
    "September, and integrates a day-length coefficient calculation for higher latitudes. "
    "Metric originally published in Huglin (1978). Day-length coefficient based on Hall & Jones (2010).",
    cell_methods="",
    var_name="hi",
    compute=indices.huglin_index,
    parameters={
        "lat": {"kind": InputKind.VARIABLE},
        "method": {"default": "jones"},
        "start_date": {"default": "04-01"},
        "end_date": {"default": "10-01"},
    },
)


biologically_effective_degree_days = Temp(
    title="Biologically effective degree days",
    identifier="biologically_effective_degree_days",
    units="K days",
    long_name="Integral of mean daily temperature above {thresh_tasmin}, with maximum value of "
    "{max_daily_degree_days}, multiplied by day-length coefficient and temperature range modifier based on {method} "
    "method for days between {start_date} and {end_date}",
    description="Heat-summation index for agroclimatic suitability estimation, developed specifically for viticulture. "
    "Computed with {method} formula (Summation of min((max((Tn + Tx)/2 - {thresh_tasmin}, 0) * k) + TR_adj, Dmax), "
    "where coefficient `k` is a latitude-based day-length for days between {start_date} and {end_date}), "
    "coefficient `TR_adj` is a modifier accounting for large temperature swings, and `Dmax` is the maximum possible"
    "amount of degree days that can be gained within a day ({max_daily_degree_days}).",
    abstract="Considers daily minimum and maximum temperature with a given base threshold between 1 April and 31 "
    "October, with a maximum daily value for cumulative degree days (typically 9°C), and integrates modification "
    "coefficients for latitudes between 40°N and 50°N as well as for swings in daily temperature range. "
    "Metric originally published in Gladstones (1992).",
    cell_methods="",
    var_name="bedd",
    compute=indices.biologically_effective_degree_days,
    parameters={
        "lat": {"kind": InputKind.VARIABLE},
        "method": {"default": "gladstones"},
        "start_date": {"default": "04-01"},
        "end_date": {"default": "11-01"},
    },
)


effective_growing_degree_days = Temp(
    title="Effective growing degree days",
    identifier="effective_growing_degree_days",
    units="K days",
    long_name="Integral of mean daily temperature above {thresh} for days between start and end dates "
    "dynamically determined using {method} method",
    description="Heat-summation index for agroclimatic suitability estimation."
    "Computed with {method} formula (Summation of max((Tn + Tx)/2 - {thresh}, 0) between dynamically-determined "
    "growing season start and end dates. The `bootsma` method uses a 10-day average temperature above {thresh} to "
    "identify a start date, while the `qian` method uses a weighted mean average above {thresh} over 5 days to "
    "determine the start date. The end date of the growing season is the date of first fall frost (Tn < 0°C) occurring"
    "after {after_date}.",
    abstract="Considers daily minimum and maximum temperature with a given base threshold between "
    "dynamically-determined growing season start and end dates. The `bootsma` method uses a 10-day mean temperature "
    "above a given threshold to identify a start date, while the `qian` method uses a weighted mean temperature above "
    "a given threshold over 5 days to determine the start date. The end date of the growing season is the date of "
    "first fall frost (Tn < 0°C) occurring after a given date (typically, July 1). "
    "Metric originally published in Bootsma et al. (2005).",
    cell_methods="",
    var_name="egdd",
    compute=indices.effective_growing_degree_days,
    parameters={
        "method": {"default": "bootsma"},
        "thresh": {"default": "5 degC"},
        "after_date": {"default": "07-01"},
    },
)


latitude_temperature_index = Temp(
    title="Latitude temperature index",
    identifier="latitude_temperature_index",
    units="",
    long_name="Mean temperature of warmest month multiplied by the difference of {lat_factor} minus latitude",
    description="A climate indice based on mean temperature of the warmest month and a latitude-based coefficient to "
    "account for longer day-length favouring growing conditions. Developed specifically for viticulture. "
    "Mean temperature of warmest month multiplied by the difference of {lat_factor} minus latitude.",
    abstract="A climate indice based on mean temperature of the warmest month and a latitude-based coefficient to "
    "account for longer day-length favouring growing conditions. Developed specifically for viticulture. "
    "Mean temperature of warmest month multiplied by the difference of latitude factor coefficient minus latitude. "
    "Metric originally published in Jackson, D. I., & Cherry, N. J. (1988).",
    cell_methods="",
    allowed_periods=["A"],
    var_name="lti",
    compute=indices.latitude_temperature_index,
    parameters={"lat": {"kind": InputKind.VARIABLE}, "lat_factor": 60},
)


late_frost_days = Temp(
    title="Late frost days",
    identifier="late_frost_days",
    units="days",
    standard_name="days_with_air_temperature_below_threshold",
    long_name="Number of days where the daily minimum temperature is below {thresh}",
    description="{freq} number of days where the daily minimum temperature is below {thresh}"
    "over the period {indexer}.",
    abstract="Number of days where the daily minimum temperature is below a given threshold between a given"
    "start date and a given end date.",
    cell_methods="time: sum over days",
    compute=indices.frost_days,
)


australian_hardiness_zones = Temp(
    title="Australian hardiness zones",
    identifier="australian_hardiness_zones",
    units="",
    long_name="Hardiness zones",
    description="A climate indice based on a {window}-year rolling average of the annual minimum temperature. "
    "Developed specifically to aid in determining plant suitability of geographic regions. The Australian National "
    "Botanical Gardens (ANBG) classification scheme divides categories into 5-degree Celsius zones, starting from -15 "
    "degrees Celsius and ending at 20 degrees Celsius.",
    abstract="A climate indice based on a multi-year rolling average of the annual minimum temperature. "
    "Developed specifically to aid in determining plant suitability of geographic regions. The Australian National "
    "Botanical Gardens (ANBG) classification scheme divides categories into 5-degree Celsius zones, starting from -15 "
    "degrees Celsius and ending at 20 degrees Celsius.",
    cell_methods="",
    allowed_periods=["A"],
    var_name="hz",
    compute=indices.hardiness_zones,
    parameters={"method": "anbg"},
)


usda_hardiness_zones = Temp(
    title="USDA hardiness zones",
    identifier="usda_hardiness_zones",
    units="",
    long_name="Hardiness zones",
    description="A climate indice based on a {window}-year rolling average of the annual minimum temperature. "
    "Developed specifically to aid in determining plant suitability of geographic regions. The USDA classification"
    "scheme divides categories into 10 degree Fahrenheit zones, with 5-degree Fahrenheit half-zones, "
    "starting from -65 degrees Fahrenheit and ending at 65 degrees Fahrenheit.",
    abstract="A climate indice based on a multi-year rolling average of the annual minimum temperature. "
    "Developed specifically to aid in determining plant suitability of geographic regions. The USDA classification"
    "scheme divides categories into 10 degree Fahrenheit zones, with 5-degree Fahrenheit half-zones, "
    "starting from -65 degrees Fahrenheit and ending at 65 degrees Fahrenheit.",
    cell_methods="",
    allowed_periods=["A"],
    var_name="hz",
    compute=indices.hardiness_zones,
    parameters={"method": "usda"},
)

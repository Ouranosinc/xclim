"""Temperature indicator definitions."""
from __future__ import annotations

from inspect import _empty  # noqa

from xclim import indices
from xclim.core import cfchecks
from xclim.core.indicator import Daily, Indicator, ResamplingIndicatorWithIndexing
from xclim.core.utils import InputKind

__all__ = [
    "tn_days_above",
    "tn_days_below",
    "tg_days_above",
    "tg_days_below",
    "tx_days_above",
    "tx_days_below",
    "tx_tn_days_above",
    "heat_wave_frequency",
    "heat_wave_max_length",
    "heat_wave_total_length",
    "heat_wave_index",
    "hot_spell_frequency",
    "hot_spell_max_length",
    "tg_max",
    "tg_mean",
    "tg_min",
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
    "cool_night_index",
    "daily_freezethaw_cycles",
    "freezethaw_spell_frequency",
    "freezethaw_spell_max_length",
    "freezethaw_spell_mean_length",
    "cooling_degree_days",
    "heating_degree_days",
    "growing_degree_days",
    "thawing_degree_days",
    "freezing_degree_days",
    "frost_season_length",
    "freshet_start",
    "frost_days",
    "last_spring_frost",
    "first_day_below",
    "first_day_above",
    "ice_days",
    "consecutive_frost_days",
    "maximum_consecutive_frost_free_days",
    "frost_free_season_start",
    "frost_free_season_end",
    "frost_free_season_length",
    "growing_season_start",
    "growing_season_end",
    "growing_season_length",
    "tropical_nights",
    "degree_days_exceedance_date",
    "warm_spell_duration_index",
    "maximum_consecutive_warm_days",
    "fire_season",
    "huglin_index",
    "biologically_effective_degree_days",
    "latitude_temperature_index",
]


# We need to declare the class here so that the `atmos` realm is automatically detected.
class Temp(Daily):
    """Indicators involving daily temperature."""


class TempWithIndexing(ResamplingIndicatorWithIndexing):
    """Indicators involving daily temperature and adding an indexing possibility."""

    src_freq = "D"


tn_days_above = TempWithIndexing(
    identifier="tn_days_above",
    units="days",
    standard_name="number_of_days_with_air_temperature_above_threshold",
    long_name="Number of days with Tmin > {thresh}",
    description="{freq} number of days where daily minimum temperature exceeds {thresh}.",
    cell_methods="time: sum over days",
    compute=indices.tn_days_above,
)

tn_days_below = TempWithIndexing(
    identifier="tn_days_below",
    units="days",
    standard_name="number_of_days_with_air_temperature_below_threshold",
    long_name="Number of days with Tmin < {thresh}",
    description="{freq} number of days where daily minimum temperature is below {thresh}.",
    cell_methods="time: sum over days",
    compute=indices.tn_days_below,
)

tg_days_above = TempWithIndexing(
    identifier="tg_days_above",
    units="days",
    standard_name="number_of_days_with_air_temperature_above_threshold",
    long_name="Number of days with Tavg > {thresh}",
    description="{freq} number of days where daily mean temperature exceeds {thresh}.",
    cell_methods="time: sum over days",
    compute=indices.tg_days_above,
)

tg_days_below = TempWithIndexing(
    identifier="tg_days_below",
    units="days",
    standard_name="number_of_days_with_air_temperature_below_threshold",
    long_name="Number of days with Tavg < {thresh}",
    description="{freq} number of days where daily mean temperature is below {thresh}.",
    cell_methods="time: sum over days",
    compute=indices.tg_days_below,
)

tx_days_above = TempWithIndexing(
    identifier="tx_days_above",
    units="days",
    standard_name="number_of_days_with_air_temperature_above_threshold",
    long_name="Number of days with Tmax > {thresh}",
    description="{freq} number of days where daily maximum temperature exceeds {thresh}.",
    cell_methods="time: sum over days",
    compute=indices.tx_days_above,
)

tx_days_below = TempWithIndexing(
    identifier="tx_days_below",
    units="days",
    standard_name="number_of_days_with_air_temperature_below_threshold",
    long_name="Number of days with Tmax < {thresh}",
    description="{freq} number of days where daily max temperature is below {thresh}.",
    cell_methods="time: sum over days",
    compute=indices.tx_days_below,
)

tx_tn_days_above = TempWithIndexing(
    identifier="tx_tn_days_above",
    units="days",
    standard_name="number_of_days_with_air_temperature_above_threshold",
    long_name="Number of days with Tmax > {thresh_tasmax} and Tmin > {thresh_tasmin}",
    description="{freq} number of days where daily maximum temperature exceeds "
    "{thresh_tasmax} and minimum temperature exceeds {thresh_tasmin}.",
    cell_methods="",
    compute=indices.tx_tn_days_above,
)


heat_wave_frequency = Temp(
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

heat_wave_max_length = Temp(
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

heat_wave_total_length = Temp(
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


heat_wave_index = Temp(
    identifier="heat_wave_index",
    units="days",
    standard_name="heat_wave_index",
    long_name="Number of days that are part of a heatwave",
    description="{freq} number of days that are part of a heatwave, "
    "defined as five or more consecutive days over {thresh}.",
    cell_methods="",
    compute=indices.heat_wave_index,
)


hot_spell_frequency = Temp(
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

hot_spell_max_length = Temp(
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

tg_mean = TempWithIndexing(
    identifier="tg_mean",
    units="K",
    standard_name="air_temperature",
    long_name="Mean daily mean temperature",
    description="{freq} mean of daily mean temperature.",
    cell_methods="time: mean over days",
    compute=indices.tg_mean,
)

tg_max = TempWithIndexing(
    identifier="tg_max",
    units="K",
    standard_name="air_temperature",
    long_name="Maximum daily mean temperature",
    description="{freq} maximum of daily mean temperature.",
    cell_methods="time: maximum over days",
    compute=indices.tg_max,
)

tg_min = TempWithIndexing(
    identifier="tg_min",
    units="K",
    standard_name="air_temperature",
    long_name="Minimum daily mean temperature",
    description="{freq} minimum of daily mean temperature.",
    cell_methods="time: minimum over days",
    compute=indices.tg_min,
)

tx_mean = TempWithIndexing(
    identifier="tx_mean",
    units="K",
    standard_name="air_temperature",
    long_name="Mean daily maximum temperature",
    description="{freq} mean of daily maximum temperature.",
    cell_methods="time: mean over days",
    compute=indices.tx_mean,
)

tx_max = TempWithIndexing(
    identifier="tx_max",
    units="K",
    standard_name="air_temperature",
    long_name="Maximum daily maximum temperature",
    description="{freq} maximum of daily maximum temperature.",
    cell_methods="time: maximum over days",
    compute=indices.tx_max,
)

tx_min = TempWithIndexing(
    identifier="tx_min",
    units="K",
    standard_name="air_temperature",
    long_name="Minimum daily maximum temperature",
    description="{freq} minimum of daily maximum temperature.",
    cell_methods="time: minimum over days",
    compute=indices.tx_min,
)

tn_mean = TempWithIndexing(
    identifier="tn_mean",
    units="K",
    standard_name="air_temperature",
    long_name="Mean daily minimum temperature",
    description="{freq} mean of daily minimum temperature.",
    cell_methods="time: mean over days",
    compute=indices.tn_mean,
)

tn_max = TempWithIndexing(
    identifier="tn_max",
    units="K",
    standard_name="air_temperature",
    long_name="Maximum daily minimum temperature",
    description="{freq} maximum of daily minimum temperature.",
    cell_methods="time: maximum over days",
    compute=indices.tn_max,
)

tn_min = TempWithIndexing(
    identifier="tn_min",
    units="K",
    standard_name="air_temperature",
    long_name="Minimum daily minimum temperature",
    description="{freq} minimum of daily minimum temperature.",
    cell_methods="time: minimum over days",
    compute=indices.tn_min,
)

daily_temperature_range = TempWithIndexing(
    title="Mean of daily temperature range.",
    identifier="dtr",
    units="K",
    standard_name="air_temperature",
    long_name="Mean Diurnal Temperature Range",
    description="{freq} mean diurnal temperature range.",
    cell_methods="time range within days time: mean over days",
    compute=indices.daily_temperature_range,
    parameters=dict(op="mean"),
)

max_daily_temperature_range = TempWithIndexing(
    title="Maximum of daily temperature range.",
    identifier="dtrmax",
    units="K",
    standard_name="air_temperature",
    long_name="Maximum Diurnal Temperature Range",
    description="{freq} maximum diurnal temperature range.",
    cell_methods="time range within days time: max over days",
    compute=indices.daily_temperature_range,
    parameters=dict(op="max"),
)

daily_temperature_range_variability = TempWithIndexing(
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

extreme_temperature_range = TempWithIndexing(
    identifier="etr",
    units="K",
    standard_name="air_temperature",
    long_name="Intra-period Extreme Temperature Range",
    description="{freq} range between the maximum of daily max temperature "
    "(tx_max) and the minimum of daily min temperature (tn_min)",
    compute=indices.extreme_temperature_range,
)

cold_spell_duration_index = Temp(
    identifier="cold_spell_duration_index",
    var_name="csdi_{window}",
    units="days",
    standard_name="cold_spell_duration_index",
    long_name="Number of days part of a percentile-defined cold spell",
    description="{freq} number of days with at least {window} consecutive days "
    "where the daily minimum temperature is below the {tasmin_per_thresh}th "
    "percentile(s). A {tasmin_per_window} day(s) window, centred on each calendar day in the "
    "{tasmin_per_period} period, is used to compute the {tasmin_per_thresh}th percentile(s).",
    cell_methods="",
    compute=indices.cold_spell_duration_index,
)

cold_spell_days = Temp(
    identifier="cold_spell_days",
    units="days",
    standard_name="cold_spell_days",
    long_name="Number of days part of a cold spell",
    description="{freq} number of days that are part of a cold spell, defined as {window} "
    "or more consecutive days with mean daily "
    "temperature below {thresh}.",
    cell_methods="",
    compute=indices.cold_spell_days,
)

cold_spell_frequency = Temp(
    identifier="cold_spell_frequency",
    units="",
    standard_name="cold_spell_frequency",
    long_name="Number of cold spell events",
    description="{freq} number cold spell events, defined as {window} "
    "or more consecutive days with mean daily "
    "temperature below {thresh}.",
    cell_methods="",
    compute=indices.cold_spell_frequency,
)

cool_night_index = Temp(
    identifier="cool_night_index",
    units="degC",
    long_name="cool night index",
    description="Mean minimum temperature for September (northern hemisphere) or March (southern hemisphere).",
    cell_methods="time: mean over days",
    abstract="A night coolness variable which takes into account the mean minimum night temperatures during the "
    "month when ripening usually occurs beyond the ripening period.",
    allowed_periods=["A"],
    compute=indices.cool_night_index,
)

daily_freezethaw_cycles = TempWithIndexing(
    identifier="dlyfrzthw",
    units="days",
    long_name="daily freezethaw cycles",
    description="{freq} number of days with a diurnal freeze-thaw cycle "
    ": Tmax > {thresh_tasmax} and Tmin <= {thresh_tasmin}.",
    cell_methods="",
    compute=indices.multiday_temperature_swing,
    parameters={
        "op": "sum",
        "window": 1,
        "thresh_tasmax": {"default": "0 degC"},
        "thresh_tasmin": {"default": "0 degC"},
    },
)


freezethaw_spell_frequency = Temp(
    identifier="freezethaw_spell_frequency",
    title="Frequency of freeze-thaw spells",
    units="days",
    long_name="{freq} number of freeze-thaw spells.",
    description="{freq} number of freeze-thaw spells"
    ": Tmax > {thresh_tasmax} and Tmin <= {thresh_tasmin} "
    "for at least {window} consecutive day(s).",
    cell_methods="",
    compute=indices.multiday_temperature_swing,
    parameters={
        "op": "count",
        "thresh_tasmax": {"default": "0 degC"},
        "thresh_tasmin": {"default": "0 degC"},
    },
)


freezethaw_spell_mean_length = Temp(
    identifier="freezethaw_spell_mean_length",
    title="Averge length of freeze-thaw spells.",
    units="days",
    long_name="{freq} average length of freeze-thaw spells.",
    description="{freq} average length of freeze-thaw spells"
    ": Tmax > {thresh_tasmax} and Tmin <= {thresh_tasmin} "
    "for at least {window} consecutive day(s).",
    cell_methods="",
    compute=indices.multiday_temperature_swing,
    parameters={
        "op": "mean",
        "thresh_tasmax": {"default": "0 degC"},
        "thresh_tasmin": {"default": "0 degC"},
    },
)


freezethaw_spell_max_length = Temp(
    identifier="freezethaw_spell_max_length",
    title="Maximal length of freeze-thaw spells.",
    units="days",
    long_name="{freq} maximal length of freeze-thaw spells.",
    description="{freq} maximal length of freeze-thaw spells"
    ": Tmax > {thresh_tasmax} and Tmin <= {thresh_tasmin} "
    "for at least {window} consecutive day(s).",
    cell_methods="",
    compute=indices.multiday_temperature_swing,
    parameters={
        "op": "max",
        "thresh_tasmax": {"default": "0 degC"},
        "thresh_tasmin": {"default": "0 degC"},
    },
)


cooling_degree_days = TempWithIndexing(
    identifier="cooling_degree_days",
    units="K days",
    standard_name="integral_of_air_temperature_excess_wrt_time",
    long_name="Cooling degree days (Tmean > {thresh})",
    description="{freq} cooling degree days above {thresh}.",
    cell_methods="time: sum over days",
    compute=indices.cooling_degree_days,
    parameters={"thresh": {"default": "18.0 degC"}},
)

heating_degree_days = TempWithIndexing(
    identifier="heating_degree_days",
    units="K days",
    standard_name="integral_of_air_temperature_deficit_wrt_time",
    long_name="Heating degree days (Tmean < {thresh})",
    description="{freq} heating degree days below {thresh}.",
    cell_methods="time: sum over days",
    compute=indices.heating_degree_days,
    parameters={"thresh": {"default": "17.0 degC"}},
)

growing_degree_days = TempWithIndexing(
    identifier="growing_degree_days",
    units="K days",
    standard_name="integral_of_air_temperature_excess_wrt_time",
    long_name="Growing degree days above {thresh}",
    description="{freq} growing degree days above {thresh}.",
    cell_methods="time: sum over days",
    compute=indices.growing_degree_days,
    parameters={"thresh": {"default": "4.0 degC"}},
)

freezing_degree_days = TempWithIndexing(
    identifier="freezing_degree_days",
    units="K days",
    standard_name="integral_of_air_temperature_deficit_wrt_time",
    long_name="Freezing degree days (Tmean < {thresh})",
    description="{freq} freezing degree days below {thresh}.",
    cell_methods="time: sum over days",
    compute=indices.heating_degree_days,
    parameters={"thresh": {"default": "0 degC"}},
)

thawing_degree_days = TempWithIndexing(
    identifier="thawing_degree_days",
    units="K days",
    standard_name="integral_of_air_temperature_excess_wrt_time",
    long_name="Thawing degree days (degree days above 0°C)",
    description="{freq} thawing degree days above 0°C.",
    cell_methods="time: sum over days",
    compute=indices.growing_degree_days,
    parameters={"thresh": {"default": "0 degC"}},
)

freshet_start = Temp(
    identifier="freshet_start",
    units="",
    standard_name="day_of_year",
    long_name="Day of year of spring freshet start",
    description="Day of year of spring freshet start, defined as the first day a temperature "
    "threshold of {thresh} is exceeded for at least {window} days.",
    compute=indices.freshet_start,
)

frost_days = TempWithIndexing(
    identifier="frost_days",
    units="days",
    standard_name="days_with_air_temperature_below_threshold",
    long_name="Number of frost days (Tmin < {thresh})",
    description="{freq} number of days with minimum daily temperature below {thresh}.",
    cell_methods="time: sum over days",
    compute=indices.frost_days,
)

frost_season_length = Temp(
    identifier="frost_season_length",
    units="days",
    standard_name="days_with_air_temperature_below_threshold",
    long_name="Length of the frost season",
    description="{freq} number of days between the first occurrence of at least "
    "{window} consecutive days with minimum daily temperature below freezing and "
    "the first occurrence of at least {window} consecutive days with "
    "minimuim daily temperature above freezing after {mid_date}.",
    cell_methods="time: sum over days",
    compute=indices.frost_season_length,
    parameters=dict(thresh="0 degC"),
)

last_spring_frost = Temp(
    identifier="last_spring_frost",
    units="",
    standard_name="day_of_year",
    long_name="Day of year of last spring frost",
    description="Day of year of last spring frost, defined as the last day a minimum temperature "
    "threshold of {thresh} is not exceeded before a given date.",
    compute=indices.last_spring_frost,
)

first_day_below = Temp(
    identifier="first_day_below",
    units="",
    standard_name="day_of_year",
    long_name="First day of year with temperature below {thresh}",
    description="First day of year with temperature below {thresh} for at least {window} days.",
    compute=indices.first_day_below,
)

first_day_above = Temp(
    identifier="first_day_above",
    units="",
    standard_name="day_of_year",
    long_name="First day of year with temperature above {thresh}",
    description="First day of year with temperature above {thresh} for at least {window} days.",
    compute=indices.first_day_above,
)


ice_days = TempWithIndexing(
    identifier="ice_days",
    standard_name="days_with_air_temperature_below_threshold",
    units="days",
    long_name="Number of ice days (Tmax < {thresh})",
    description="{freq} number of days with maximum daily temperature below {thresh}.",
    cell_methods="time: sum over days",
    compute=indices.ice_days,
)

consecutive_frost_days = Temp(
    identifier="consecutive_frost_days",
    units="days",
    standard_name="spell_length_of_days_with_air_temperature_below_threshold",
    long_name="Maximum number of consecutive days with Tmin < {thresh}",
    description="{freq} maximum number of consecutive days with "
    "minimum daily temperature below {thresh}.",
    cell_methods="time: maximum over days",
    compute=indices.maximum_consecutive_frost_days,
)

frost_free_season_length = Temp(
    identifier="frost_free_season_length",
    units="days",
    standard_name="days_with_air_temperature_above_threshold",
    long_name="Length of the frost free season",
    description="{freq} number of days between the first occurrence of at least "
    "{window} consecutive days with minimum daily temperature above or at the freezing point and "
    "the first occurrence of at least {window} consecutive days with "
    "minimum daily temperature below freezing after {mid_date}.",
    cell_methods="time: sum over days",
    compute=indices.frost_free_season_length,
    parameters={"thresh": {"default": "0 degC"}},
)

frost_free_season_start = Temp(
    identifier="frost_free_season_start",
    units="",
    standard_name="day_of_year",
    long_name="Day of year of frost free season start",
    description="Day of year of beginning of frost free season, defined as the first day a minimum temperature "
    "threshold of {thresh} is equal or exceeded for at least {window} days.",
    compute=indices.frost_free_season_start,
    parameters={"thresh": {"default": "0 degC"}},
)

frost_free_season_end = Temp(
    identifier="frost_free_season_end",
    units="",
    standard_name="day_of_year",
    long_name="Day of year of frost free season end",
    description="Day of year of end of frost free season, defined as the first day minimum temperatures below a  "
    "threshold of {thresh}, after a run of days above this threshold, for at least {window} days.",
    cell_methods="",
    compute=indices.frost_free_season_end,
    parameters={"thresh": {"default": "0 degC"}},
)

maximum_consecutive_frost_free_days = Temp(
    identifier="consecutive_frost_free_days",
    units="days",
    standard_name="spell_length_of_days_with_air_temperature_above_threshold",
    long_name="Maximum number of consecutive days with Tmin >= {thresh}",
    description="{freq} maximum number of consecutive days with "
    "minimum daily temperature above or equal to {thresh}.",
    cell_methods="time: maximum over days",
    compute=indices.maximum_consecutive_frost_free_days,
)

growing_season_start = Temp(
    identifier="growing_season_start",
    units="",
    standard_name="day_of_year",
    long_name="Day of year of growing season start",
    description="Day of year of start of growing season, defined as the first day of "
    "consistent superior or equal to threshold temperature of {thresh} after a run of "
    "{window} days inferior to threshold temperature.",
    cell_methods="",
    compute=indices.growing_season_start,
    parameters={"thresh": {"default": "5.0 degC"}},
)

growing_season_length = Temp(
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
    parameters={"thresh": {"default": "5.0 degC"}},
)

growing_season_end = Temp(
    identifier="growing_season_end",
    units="",
    standard_name="day_of_year",
    long_name="Day of year of growing season end",
    description="Day of year of end of growing season, defined as the first day of "
    "consistent inferior threshold temperature of {thresh} after a run of "
    "{window} days superior to threshold temperature.",
    cell_methods="",
    compute=indices.growing_season_end,
    parameters={"thresh": {"default": "5.0 degC"}},
)

tropical_nights = TempWithIndexing(
    identifier="tropical_nights",
    units="days",
    standard_name="number_of_days_with_air_temperature_above_threshold",
    long_name="Number of Tropical Nights (Tmin > {thresh})",
    description="{freq} number of Tropical Nights : defined as days with minimum daily temperature"
    " above {thresh}.",
    cell_methods="time: sum over days",
    compute=indices.tn_days_above,
    parameters={"thresh": {"default": "20.0 degC"}},
)

tg90p = TempWithIndexing(
    identifier="tg90p",
    units="days",
    standard_name="days_with_air_temperature_above_threshold",
    long_name="Number of days when Tmean > {tas_per_thresh}th percentile",
    description="{freq} number of days with mean daily temperature above the the {tas_per_thresh}th "
    "percentile(s). A {tas_per_window} day(s) window, centred on each calendar day in the "
    "{tas_per_period} period, is used to compute the {tas_per_thresh}th percentile(s).",
    cell_methods="time: sum over days",
    compute=indices.tg90p,
)

tg10p = TempWithIndexing(
    identifier="tg10p",
    units="days",
    standard_name="days_with_air_temperature_below_threshold",
    long_name="Number of days when Tmean < {tas_per_thresh}th percentile",
    description="{freq} number of days with mean daily temperature below the {tas_per_thresh}th "
    "percentile(s). A {tas_per_window} day(s) window, centred on each calendar day in the "
    "{tas_per_period} period, is used to compute the {tas_per_thresh}th percentile(s).",
    cell_methods="time: sum over days",
    compute=indices.tg10p,
)

tx90p = TempWithIndexing(
    identifier="tx90p",
    units="days",
    standard_name="days_with_air_temperature_above_threshold",
    long_name="Number of days when Tmax > {tasmax_per_thresh}th percentile",
    description="{freq} number of days with maximum daily temperature above the {tasmax_per_thresh}th "
    "percentile(s). A {tasmax_per_window} day(s) window, centred on each calendar day in the "
    "{tasmax_per_period} period, is used to compute the {tasmax_per_thresh}th percentile(s).",
    cell_methods="time: sum over days",
    compute=indices.tx90p,
)

tx10p = TempWithIndexing(
    identifier="tx10p",
    units="days",
    standard_name="days_with_air_temperature_below_threshold",
    long_name="Number of days when Tmax < {tasmax_per_thresh}th percentile",
    description="{freq} number of days with maximum daily temperature below the {tasmax_per_thresh}th "
    "percentile(s). A {tasmax_per_window} day(s) window, centred on each calendar day in the "
    "{tasmax_per_period} period, is used to compute the {tasmax_per_thresh}th percentile(s).",
    cell_methods="time: sum over days",
    compute=indices.tx10p,
)

tn90p = TempWithIndexing(
    identifier="tn90p",
    units="days",
    standard_name="days_with_air_temperature_above_threshold",
    long_name="Number of days when Tmin > {tasmin_per_thresh}th percentile",
    description="{freq} number of days with minimum daily temperature above the the {tasmin_per_thresh}th "
    "percentile(s). A {tasmin_per_window} day(s) window, centred on each calendar day in the "
    "{tasmin_per_period} period, is used to compute the {tasmin_per_thresh}th percentile(s).",
    cell_methods="time: sum over days",
    compute=indices.tn90p,
)

tn10p = TempWithIndexing(
    identifier="tn10p",
    units="days",
    standard_name="days_with_air_temperature_below_threshold",
    long_name="Number of days when Tmin < {tasmin_per_thresh}th percentile",
    description="{freq} number of days with minimum daily temperature below the the {tasmin_per_thresh}th "
    "percentile(s). A {tasmin_per_window} day(s) window, centred on each calendar day in the "
    "{tasmin_per_period} period, is used to compute the {tasmin_per_thresh}th percentile(s).",
    cell_methods="time: sum over days",
    compute=indices.tn10p,
)


degree_days_exceedance_date = Temp(
    identifier="degree_days_exceedance_date",
    units="",
    standard_name="day_of_year",
    long_name="Day of year when cumulative degree days exceed {sum_thresh}.",
    description="Day of year when the integral of degree days (tmean {op} {thresh})"
    " exceeds {sum_thresh}, the cumulative sum starts on {after_date}.",
    cell_methods="",
    compute=indices.degree_days_exceedance_date,
)


warm_spell_duration_index = Temp(
    identifier="warm_spell_duration_index",
    long_name="Number of days part of a percentile-defined warm spell",
    description="{freq} number of days with at least {window} consecutive days "
    "where the daily maximum temperature is above the {tasmax_per_thresh}th "
    "percentile(s). A {tasmax_per_window} day(s) window, centred on each calendar day in the "
    "{tasmax_per_period} period, is used to compute the {tasmax_per_thresh}th percentile(s).",
    units="days",
    standard_name="number_of_days_with_air_temperature_above_threshold",
    cell_methods="time: sum over days",
    compute=indices.warm_spell_duration_index,
)


maximum_consecutive_warm_days = Temp(
    identifier="maximum_consecutive_warm_days",
    description="{freq} longest spell of consecutive days with Tmax above {thresh}.",
    units="days",
    standard_name="spell_length_of_days_with_air_temperature_above_threshold",
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
    identifier="huglin_index",
    units="",
    long_name="Huglin heliothermal index (Summation of ((Tmin + Tmax)/2 - {thresh}) * Latitude-based day-length"
    "coefficient (`k`), for days between {start_date} and {end_date}).",
    description="Heat-summation index for agroclimatic suitability estimation, developed specifically for viticulture. "
    "Considers daily Tmin and Tmax with a base of {thresh}, typically between 1 April and 30 September. "
    "Integrates a day-length coefficient calculation for higher latitudes.",
    cell_methods="",
    comment="Metric originally published in Huglin (1978). Day-length coefficient based on Hall & Jones (2010)",
    var_name="hi",
    compute=indices.huglin_index,
    parameters=dict(method="jones"),
)


biologically_effective_degree_days = Temp(
    identifier="biologically_effective_degree_days",
    units="K days",
    long_name="Biologically effective degree days computed with {method} formula (Summation of min((max((Tmin + Tmax)/2"
    " - {thresh_tasmin}, 0) * k) + TR_adg, 9°C), for days between {start_date} and {end_date}).",
    description="Heat-summation index for agroclimatic suitability estimation, developed specifically for viticulture. "
    "Considers daily Tmin and Tmax with a base of {thresh_tasmin} between 1 April and 31 October, with a maximum daily "
    "value for degree days (typically 9°C). It also integrates a modification coefficient for latitudes "
    "between 40°N and 50°N as well as swings in daily temperature range.",
    cell_methods="",
    comment="Original formula published in Gladstones, 1992.",
    var_name="bedd",
    compute=indices.biologically_effective_degree_days,
    parameters={"method": "gladstones", "lat": {"kind": InputKind.VARIABLE}},
)


effective_growing_degree_days = Temp(
    identifier="effective_growing_degree_days",
    units="K days",
    long_name="Effective growing degree days computed with {method} formula (Summation of max((Tmin + Tmax)/2 "
    "- {thresh}, 0), for days between between dynamically-determined start and end dates).",
    description="Heat-summation index for agroclimatic suitability estimation."
    "Considers daily Tmin and Tmax with a base of {thresh} between dynamically-determined growing season start"
    "and end dates. The 'bootsma' method uses a 10-day average temperature above {thresh} to identify a start date, "
    "while the 'qian' method uses a weighted mean average above {thresh} over 5 days to determine start date. "
    "The end date of the growing season is the date of first fall frost (Tmin < 0°C).",
    cell_methods="",
    comment="Original formula published in Bootsma et al. 2005.",
    var_name="egdd",
    compute=indices.effective_growing_degree_days,
)


latitude_temperature_index = Temp(
    identifier="latitude_temperature_index",
    units="",
    long_name="Latitude-temperature index",
    description="A climate indice based on mean temperature of the warmest month and a latitude-based coefficient to "
    "account for longer day-length favouring growing conditions. Developed specifically for viticulture. "
    "Mean temperature of warmest month * ({lat_factor} - latitude).",
    cell_methods="",
    allowed_periods=["A"],
    comment="Indice originally published in Jackson, D. I., & Cherry, N. J. (1988)",
    var_name="lti",
    compute=indices.latitude_temperature_index,
    parameters={"lat_factor": 60, "lat": {"kind": InputKind.VARIABLE}},
)

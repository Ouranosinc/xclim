"""Precipitation indicator definitions."""
from __future__ import annotations

from inspect import _empty  # noqa

from xclim import indices
from xclim.core import cfchecks
from xclim.core.indicator import (
    Daily,
    Hourly,
    Indicator,
    ResamplingIndicator,
    ResamplingIndicatorWithIndexing,
)
from xclim.core.utils import InputKind

__all__ = [
    "rain_on_frozen_ground_days",
    "max_1day_precipitation_amount",
    "max_n_day_precipitation_amount",
    "wetdays",
    "wetdays_prop",
    "dry_days",
    "maximum_consecutive_dry_days",
    "maximum_consecutive_wet_days",
    "daily_pr_intensity",
    "max_pr_intensity",
    "precip_accumulation",
    "liquid_precip_accumulation",
    "solid_precip_accumulation",
    "standardized_precipitation_index",
    "standardized_precipitation_evapotranspiration_index",
    "drought_code",
    "fire_weather_indexes",
    "cffwis_indices",
    "keetch_byram_drought_index",
    "griffiths_drought_factor",
    "mcarthur_forest_fire_danger_index",
    "last_snowfall",
    "first_snowfall",
    "days_with_snow",
    "days_over_precip_thresh",
    "days_over_precip_doy_thresh",
    "high_precip_low_temp",
    "fraction_over_precip_thresh",
    "fraction_over_precip_doy_thresh",
    "liquid_precip_ratio",
    "dry_spell_frequency",
    "dry_spell_total_length",
    "wet_precip_accumulation",
    "rprctot",
    "cold_and_dry_days",
    "cold_and_wet_days",
    "warm_and_dry_days",
    "warm_and_wet_days",
]


class FireWeather(Indicator):
    """Non resampling - precipitation related indicators."""

    src_freq = "D"
    context = "hydro"


class Precip(Daily):
    """Indicator involving daily pr series."""

    context = "hydro"


class PrecipWithIndexing(ResamplingIndicatorWithIndexing):
    """Indicator involving daily pr series and allowing indexing."""

    src_freq = "D"
    context = "hydro"


class PrTasxWithIndexing(ResamplingIndicatorWithIndexing):
    """Indicator involving pr and one of tas, tasmin or tasmax, allowing indexing."""

    src_freq = "D"
    context = "hydro"

    def cfcheck(self, pr, tas):
        cfchecks.cfcheck_from_name("pr", pr)
        cfchecks.check_valid(tas, "standard_name", "air_temperature")


class StandardizedIndexes(ResamplingIndicator):
    """Resampling but flexible inputs indicators."""

    src_freq = ["D", "M"]
    context = "hydro"


class HrPrecip(Hourly):
    """Indicator involving hourly pr series."""

    context = "hydro"


rain_on_frozen_ground_days = PrTasxWithIndexing(
    title="Number of rain on frozen ground days",
    identifier="rain_frzgr",
    units="days",
    standard_name="number_of_days_with_lwe_thickness_of_precipitation_amount_above_threshold",
    long_name="Number of rain on frozen ground days (mean daily temperature > 0℃ and precipitation > {thresh})",
    description="{freq} number of days with rain above {thresh} after a series of seven days with average daily "
    "temperature below 0℃. Precipitation is assumed to be rain when the daily average temperature is above 0℃.",
    abstract="The number of days with rain above a given threshold after a series of seven days with average daily "
    "temperature below 0°C. Precipitation is assumed to be rain when the daily average temperature is above 0°C.",
    cell_methods="",
    compute=indices.rain_on_frozen_ground_days,
)

max_1day_precipitation_amount = PrecipWithIndexing(
    title="Maximum 1-day total precipitation",
    identifier="rx1day",
    units="mm/day",
    standard_name="lwe_thickness_of_precipitation_amount",
    long_name="Maximum 1-day total precipitation",
    description="{freq} maximum 1-day total precipitation",
    abstract="Maximum total daily precipitation for a given period.",
    cell_methods="time: maximum over days",
    compute=indices.max_1day_precipitation_amount,
)

max_n_day_precipitation_amount = Precip(
    title="maximum n-day total precipitation",
    identifier="max_n_day_precipitation_amount",
    var_name="rx{window}day",
    units="mm",
    standard_name="lwe_thickness_of_precipitation_amount",
    long_name="maximum {window}-day total precipitation amount",
    description="{freq} maximum {window}-day total precipitation amount.",
    abstract="Maximum of the moving sum of daily precipitation for a given period.",
    cell_methods="time: maximum over days",
    compute=indices.max_n_day_precipitation_amount,
)

wetdays = PrecipWithIndexing(
    title="Number of wet days",
    identifier="wetdays",
    units="days",
    standard_name="number_of_days_with_lwe_thickness_of_precipitation_amount_at_or_above_threshold",
    long_name="Number of days with daily precipitation at or above {thresh}",
    description="{freq} number of days with daily precipitation at or above {thresh}.",
    abstract="The number of days with daily precipitation at or above a given threshold.",
    cell_methods="time: sum over days",
    compute=indices.wetdays,
)

wetdays_prop = PrecipWithIndexing(
    title="Proportion of wet days",
    identifier="wetdays_prop",
    units="1",
    long_name="Proportion of days with precipitation at or above {thresh}",
    description="{freq} proportion of days with precipitation at or above {thresh}.",
    abstract="The proportion of days with daily precipitation at or above a given threshold.",
    cell_methods="time: sum over days",
    compute=indices.wetdays_prop,
)

dry_days = PrecipWithIndexing(
    title="Number of dry days",
    identifier="dry_days",
    units="days",
    standard_name="number_of_days_with_lwe_thickness_of_precipitation_amount_below_threshold",
    long_name="Number of dry days",
    description="{freq} number of days with daily precipitation under {thresh}.",
    abstract="The number of days with daily precipitation under a given threshold.",
    cell_methods="time: sum over days",
    compute=indices.dry_days,
)

maximum_consecutive_wet_days = Precip(
    title="Maximum consecutive wet days",
    identifier="cwd",
    units="days",
    standard_name="number_of_days_with_lwe_thickness_of_precipitation_amount_at_or_above_threshold",
    long_name="Maximum consecutive days with daily precipitation at or above {thresh}",
    description="{freq} maximum number of consecutive days with daily precipitation at or above {thresh}.",
    abstract="The longest number of consecutive days where daily precipitation is at or above a given threshold.",
    cell_methods="time: sum over days",
    compute=indices.maximum_consecutive_wet_days,
)

maximum_consecutive_dry_days = Precip(
    title="Maximum consecutive dry days",
    identifier="cdd",
    units="days",
    standard_name="number_of_days_with_lwe_thickness_of_precipitation_amount_below_threshold",
    long_name="Maximum consecutive days with daily precipitation below {thresh}",
    description="{freq} maximum number of consecutive days with daily precipitation below {thresh}.",
    abstract="The longest number of consecutive days where daily precipitation below a given threshold.",
    cell_methods="time: sum over days",
    compute=indices.maximum_consecutive_dry_days,
)

daily_pr_intensity = PrecipWithIndexing(
    title="Simple Daily Intensity Index",
    identifier="sdii",
    units="mm d-1",
    standard_name="lwe_thickness_of_precipitation_amount",
    long_name="Average precipitation during days with daily precipitation over {thresh} "
    "(Simple Daily Intensity Index: SDII)",
    description="{freq} Simple Daily Intensity Index (SDII) or "
    "{freq} average precipitation for days with daily precipitation over {thresh}.",
    abstract="Average precipitation for days with daily precipitation above a given threshold.",
    cell_methods="",
    compute=indices.daily_pr_intensity,
)

max_pr_intensity = HrPrecip(
    title="Maximum precipitation intensity over time window",
    identifier="max_pr_intensity",
    units="mm h-1",
    standard_name="precipitation",
    long_name="Maximum precipitation intensity over rolling {window}h time window",
    description="{freq} maximum precipitation intensity over rolling {window}h time window.",
    abstract="Maximum precipitation intensity over a given rolling time window.",
    cell_methods="time: max",
    compute=indices.max_pr_intensity,
    duration="{window}",
    keywords="IDF curves",
)

precip_accumulation = PrecipWithIndexing(
    title="Total accumulated precipitation (solid and liquid)",
    identifier="prcptot",
    units="mm",
    standard_name="lwe_thickness_of_precipitation_amount",
    long_name="Total accumulated precipitation",
    description="{freq} total precipitation.",
    abstract="Total accumulated precipitation. If the average daily temperature is given, the phase parameter can be "
    "used to restrict the calculation to precipitation of only one phase (liquid or solid). Precipitation is "
    "considered solid if the average daily temperature is below 0°C (and vice versa).",
    cell_methods="time: sum over days",
    compute=indices.precip_accumulation,
    parameters=dict(tas=None, phase=None),
)

wet_precip_accumulation = PrecipWithIndexing(
    title="Total accumulated precipitation (solid and liquid) during wet days",
    identifier="wet_prcptot",
    units="mm",
    standard_name="lwe_thickness_of_precipitation_amount",
    long_name="Total accumulated precipitation over days where precipitation exceeds {thresh}",
    description="{freq} total precipitation over wet days, defined as days where precipitation exceeds {thresh}.",
    abstract="Total accumulated precipitation on days with precipitation. "
    "A day is considered to have precipitation if the precipitation is greater than or equal to a given threshold.",
    cell_methods="time: sum over days",
    compute=indices.prcptot,
    parameters={"thresh": {"default": "1 mm/day"}},
)


liquid_precip_accumulation = PrTasxWithIndexing(
    title="Total accumulated liquid precipitation.",
    identifier="liquidprcptot",
    units="mm",
    standard_name="lwe_thickness_of_liquid_precipitation_amount",
    long_name="Total accumulated precipitation when temperature is above {thresh}",
    description="{freq} total {phase} precipitation, estimated as precipitation when temperature is above {thresh}.",
    abstract="Total accumulated liquid precipitation. "
    "Precipitation is considered liquid when the average daily temperature is above 0°C.",
    cell_methods="time: sum over days",
    compute=indices.precip_accumulation,
    parameters={"tas": {"kind": InputKind.VARIABLE}, "phase": "liquid"},
)

solid_precip_accumulation = PrTasxWithIndexing(
    title="Total accumulated solid precipitation.",
    identifier="solidprcptot",
    units="mm",
    standard_name="lwe_thickness_of_snowfall_amount",
    long_name="Total accumulated solid precipitation",
    description="{freq} total solid precipitation, estimated as precipitation when temperature at or below {thresh}.",
    abstract="Total accumulated solid precipitation. "
    "Precipitation is considered solid when the average daily temperature is at or below 0°C.",
    cell_methods="time: sum over days",
    compute=indices.precip_accumulation,
    parameters={"tas": {"kind": InputKind.VARIABLE}, "phase": "solid"},
)

standardized_precipitation_index = StandardizedIndexes(
    title="Standardized Precipitation Index (SPI)",
    identifier="spi",
    units="",
    standard_name="spi",
    long_name="Standardized Precipitation Index (SPI)",
    description="Precipitations over a moving {window}-X window, normalized such that SPI averages to 0 for "
    "calibration data. The window unit `X` is the minimal time period defined by resampling frequency {freq}.",
    abstract="Precipitation over a moving window, normalized such that SPI averages to 0 for the calibration data. "
    "The window unit `X` is the minimal time period defined by the resampling frequency.",
    cell_methods="",
    compute=indices.standardized_precipitation_index,
)

standardized_precipitation_evapotranspiration_index = StandardizedIndexes(
    title="Standardized Precipitation Evapotranspiration Index (SPEI)",
    identifier="spei",
    units="",
    standard_name="spei",
    long_name="Standardized precipitation evapotranspiration index (SPEI)",
    description="Water budget (precipitation minus evapotranspiration) over a moving {window}-X window, normalized "
    "such that SPEI averages to 0 for calibration data. The window unit `X` is the minimal time period defined by the "
    "resampling frequency {freq}.",
    abstract="Water budget (precipitation - evapotranspiration) over a moving window, normalized such that the "
    "SPEI averages to 0 for the calibration data. The window unit `X` is the minimal time period defined by the "
    "resampling frequency.",
    cell_methods="",
    compute=indices.standardized_precipitation_evapotranspiration_index,
)

drought_code = FireWeather(
    title="Daily drought code",
    identifier="dc",
    units="",
    standard_name="drought_code",
    long_name="Drought Code",
    description="Numerical code estimating the average moisture content of organic layers.",
    abstract="The Drought Index is part of the Canadian Forest-Weather Index system. "
    "It is a numerical code that estimates the average moisture content of organic layers.",
    compute=indices.drought_code,
    missing="skip",
)


fire_weather_indexes = FireWeather(
    title="Fire weather indexes",
    identifier="fwi",
    realm="atmos",
    var_name=["dc", "dmc", "ffmc", "isi", "bui", "fwi"],
    standard_name=[
        "drought_code",
        "duff_moisture_code",
        "fine_fuel_moisture_code",
        "initial_spread_index",
        "buildup_index",
        "fire_weather_index",
    ],
    long_name=[
        "Drought code",
        "Duff moisture code",
        "Fine fuel moisture code",
        "Initial spread index",
        "Buildup index",
        "Fire weather index",
    ],
    description=[
        "Numeric rating of the average moisture content of deep, compact organic layers.",
        "Numeric rating of the average moisture content of loosely compacted organic layers of moderate depth.",
        "Numeric rating of the average moisture content of litter and other cured fine fuels.",
        "Numeric rating of the expected rate of fire spread.",
        "Numeric rating of the total amount of fuel available for combustion.",
        "Numeric rating of fire intensity.",
    ],
    units="",
    compute=indices.cffwis_indices,
    missing="skip",
    _version_deprecated="0.38.0",
)


cffwis_indices = FireWeather(
    identifier="cffwis",
    realm="atmos",
    var_name=["dc", "dmc", "ffmc", "isi", "bui", "fwi"],
    standard_name=[
        "drought_code",
        "duff_moisture_code",
        "fine_fuel_moisture_code",
        "initial_spread_index",
        "buildup_index",
        "fire_weather_index",
    ],
    long_name=[
        "Drought Code",
        "Duff Moisture Code",
        "Fine Fuel Moisture Code",
        "Initial Spread Index",
        "Buildup Index",
        "Fire Weather Index",
    ],
    description=[
        "Numeric rating of the average moisture content of deep, compact organic layers.",
        "Numeric rating of the average moisture content of loosely compacted organic layers of moderate depth.",
        "Numeric rating of the average moisture content of litter and other cured fine fuels.",
        "Numeric rating of the expected rate of fire spread.",
        "Numeric rating of the total amount of fuel available for combustion.",
        "Numeric rating of fire intensity.",
    ],
    units="",
    compute=indices.cffwis_indices,
    missing="skip",
)


keetch_byram_drought_index = FireWeather(
    identifier="kbdi",
    units="mm/day",
    standard_name="keetch_byram_drought_index",
    long_name="Keetch-Byran Drought Index",
    description="Amount of water necessary to bring the soil moisture content back to field capacity",
    compute=indices.keetch_byram_drought_index,
    missing="skip",
)


griffiths_drought_factor = FireWeather(
    identifier="df",
    units="",
    standard_name="griffiths_drought_factor",
    long_name="Griffiths Drought Factor",
    description="Numeric indicator of the forest fire fuel availability in the deep litter bed",
    compute=indices.griffiths_drought_factor,
    missing="skip",
)


mcarthur_forest_fire_danger_index = FireWeather(
    identifier="ffdi",
    units="",
    standard_name="mcarthur_forest_fire_danger_index",
    long_name="McArthur Forest Fire Danger Index",
    description="Numeric rating of the potential danger of a forest fire",
    compute=indices.mcarthur_forest_fire_danger_index,
    missing="skip",
)


last_snowfall = PrecipWithIndexing(
    title="Last day where solid precipitation flux exceeded a given threshold",
    identifier="last_snowfall",
    standard_name="day_of_year",
    long_name="Date of last day where the solid precipitation flux exceeded {thresh}",
    description="{freq} last day where the solid precipitation flux exceeded {thresh}.",
    abstract="The last day where the solid precipitation flux exceeded a given threshold during a time period.",
    units="",
    compute=indices.last_snowfall,
)

first_snowfall = PrecipWithIndexing(
    title="First day where solid precipitation flux exceeded a given threshold",
    identifier="first_snowfall",
    standard_name="day_of_year",
    long_name="Date of first day where the solid precipitation flux exceeded {thresh}",
    description="{freq} first day where the solid precipitation flux exceeded {thresh}.",
    abstract="The first day where the solid precipitation flux exceeded a given threshold during a time period.",
    units="",
    compute=indices.first_snowfall,
)

days_with_snow = PrecipWithIndexing(
    title="Days with snowfall",
    identifier="days_with_snow",
    long_name="Number of days with solid precipitation flux between {low} and {high} thresholds",
    description="{freq} number of days with solid precipitation flux larger than {low} and smaller or equal to {high}.",
    abstract="Number of days with snow between a lower and upper limit.",
    units="days",
    compute=indices.days_with_snow,
)

# FIXME: Are days_over_precip_thresh and days_over_precip_doy_thresh the same thing?
days_over_precip_thresh = PrecipWithIndexing(
    title="Number of days with precipitation above a given percentile",
    identifier="days_over_precip_thresh",
    standard_name="number_of_days_with_lwe_thickness_of_precipitation_amount_above_threshold",
    long_name="Number of days with precipitation flux above the {pr_per_thresh}th percentile of {pr_per_period}",
    description="{freq} number of days with precipitation above the {pr_per_thresh}th percentile of {pr_per_period} "
    "period. Only days with at least {thresh} are counted.",
    abstract="Number of days in a period where precipitation is above a given percentile, "
    "calculated over a given period and a fixed threshold.",
    units="days",
    cell_methods="time: sum over days",
    compute=indices.days_over_precip_thresh,
)

# FIXME: Are days_over_precip_thresh and days_over_precip_doy_thresh the same thing?
days_over_precip_doy_thresh = PrecipWithIndexing(
    title="Number of days with precipitation above a given daily percentile",
    identifier="days_over_precip_doy_thresh",
    standard_name="number_of_days_with_lwe_thickness_of_precipitation_amount_above_daily_threshold",
    long_name="Number of days with daily precipitation flux above the {pr_per_thresh}th percentile of {pr_per_period}",
    description="{freq} number of days with precipitation above the {pr_per_thresh}th daily percentile. Only days with "
    "at least {thresh} are counted. A {pr_per_window} day(s) window, centered on each calendar day in the "
    "{pr_per_period} period, is used to compute the {pr_per_thresh}th percentile(s).",
    abstract="Number of days in a period where precipitation is above a given daily percentile and a fixed threshold.",
    units="days",
    cell_methods="time: sum over days",
    compute=indices.days_over_precip_thresh,
)

high_precip_low_temp = PrTasxWithIndexing(
    title="Days with precipitation and cold temperature",
    identifier="high_precip_low_temp",
    long_name="Days with precipitation at or above {pr_thresh} and temperature below {tas_thresh}",
    description="{freq} number of days with precipitation at or above {pr_thresh} and temperature below {tas_thresh}.",
    abstract="Number of days with precipitation above a given threshold and temperature below a given threshold.",
    units="days",
    cell_methods="time: sum over days",
    compute=indices.high_precip_low_temp,
)

# FIXME: Are fraction_over_precip_thresh and fraction_over_precip_doy_thresh the same thing?
# FIXME: Clarity needed in both French and English metadata fields
fraction_over_precip_doy_thresh = PrecipWithIndexing(
    title="",
    identifier="fraction_over_precip_doy_thresh",
    long_name="Fraction of precipitation due to days with daily precipitation above {pr_per_thresh}th daily percentile",
    description="{freq} fraction of total precipitation due to days with precipitation above {pr_per_thresh}th daily "
    "percentile. Only days with at least {thresh} are included in the total. A {pr_per_window} day(s) window, centered "
    "on each calendar day in the {pr_per_period} period, is used to compute the {pr_per_thresh}th percentile(s).",
    units="",
    cell_methods="",
    compute=indices.fraction_over_precip_thresh,
)

# FIXME: Are fraction_over_precip_thresh and fraction_over_precip_doy_thresh the same thing?
# FIXME: Clarity needed in both French and English metadata fields
fraction_over_precip_thresh = PrecipWithIndexing(
    identifier="fraction_over_precip_thresh",
    long_name="Fraction of precipitation due to days with precipitation above {pr_per_thresh}th daily percentile",
    description="{freq} fraction of total precipitation due to days with precipitation above {pr_per_thresh}th "
    "percentile of {pr_per_period} period. Only days with at least {thresh} are included in the total.",
    units="",
    cell_methods="",
    compute=indices.fraction_over_precip_thresh,
)

liquid_precip_ratio = PrTasxWithIndexing(
    title="Fraction of liquid to total precipitation",
    identifier="liquid_precip_ratio",
    long_name="Fraction of liquid to total precipitation (temperature above {thresh})",
    description="The {freq} ratio of rainfall to total precipitation. Rainfall is estimated as precipitation on days "
    "where temperature is above {thresh}.",
    abstract="The ratio of total liquid precipitation over the total precipitation. Liquid precipitation is "
    "approximated from total precipitation on days where temperature is above a given threshold.",
    units="",
    compute=indices.liquid_precip_ratio,
    parameters={"tas": {"kind": InputKind.VARIABLE}, "prsn": None},
)


dry_spell_frequency = Precip(
    title="Dry spell frequency",
    identifier="dry_spell_frequency",
    long_name="Number of dry periods of {window} day(s) or more, during which the {op} precipitation on a "
    "window of {window} day(s) is below {thresh}.",
    description="The {freq} number of dry periods of {window} day(s) or more, during which the {op} precipitation on a "
    "window of {window} day(s) is below {thresh}.",
    abstract="The frequency of dry periods of `N` days or more, during which the accumulated or maximum precipitation "
    "over a given time window of days is below a given threshold.",
    units="",
    cell_methods="",
    compute=indices.dry_spell_frequency,
)


dry_spell_total_length = Precip(
    title="Dry spell total length",
    identifier="dry_spell_total_length",
    long_name="Number of days in dry periods of {window} day(s) or more, during which the {op} "
    "precipitation within windows of {window} day(s) is under {thresh}.",
    description="The {freq} number of days in dry periods of {window} day(s) or more, during which the {op} "
    "precipitation within windows of {window} day(s) is under {thresh}.",
    abstract="The total length of dry periods of `N` days or more, during which the accumulated or maximum "
    "precipitation over a given time window of days is below a given threshold.",
    units="days",
    cell_methods="",
    compute=indices.dry_spell_total_length,
)

rprctot = PrecipWithIndexing(
    title="Proportion of accumulated precipitation arising from convective processes",
    identifier="rprctot",
    long_name="Proportion of accumulated precipitation arising from convective processes"
    "with precipitation of at least {thresh}",
    description="{freq} proportion of accumulated precipitation arising from convective processes "
    "with precipitation of at least {thresh}.",
    abstract="The proportion of total precipitation due to convective processes. "
    "Only days with surpassing a minimum precipitation flux are considered.",
    units="",
    cell_methods="time: sum",
    compute=indices.rprctot,
)


cold_and_dry_days = PrecipWithIndexing(
    title="Cold and dry days",
    identifier="cold_and_dry_days",
    units="days",
    long_name="Number of days where temperature is below {tas_per_thresh}th percentile and "
    "precipitation is below {pr_per_thresh}th percentile",
    description="{freq} number of days where temperature is below {tas_per_thresh}th percentile and "
    "precipitation is below {pr_per_thresh}th percentile.",
    abstract="Number of days with temperature below a given percentile and precipitation below a given percentile.",
    cell_methods="time: sum over days",
    compute=indices.cold_and_dry_days,
)

warm_and_dry_days = PrecipWithIndexing(
    title="Warm and dry days",
    identifier="warm_and_dry_days",
    units="days",
    long_name="Number of days where temperature is above {tas_per_thresh}th percentile and "
    "precipitation is below {pr_per_thresh}th percentile",
    description="{freq} number of days where temperature is above {tas_per_thresh}th percentile and "
    "precipitation is below {pr_per_thresh}th percentile.",
    abstract="Number of days with temperature above a given percentile and precipitation below a given percentile.",
    cell_methods="time: sum over days",
    compute=indices.warm_and_dry_days,
)

warm_and_wet_days = PrecipWithIndexing(
    title="Warm and wet days",
    identifier="warm_and_wet_days",
    units="days",
    long_name="Number of days where temperature above {tas_per_thresh}th percentile and "
    "precipitation above {pr_per_thresh}th percentile",
    description="{freq} number of days where temperature is above {tas_per_thresh}th percentile and "
    "precipitation is above {pr_per_thresh}th percentile.",
    abstract="Number of days with temperature above a given percentile and precipitation above a given percentile.",
    cell_methods="time: sum over days",
    compute=indices.warm_and_wet_days,
)

cold_and_wet_days = PrecipWithIndexing(
    title="Cold and wet days",
    identifier="cold_and_wet_days",
    units="days",
    long_name="Number of days where temperature is below {tas_per_thresh}th percentile and "
    "precipitation is above {pr_per_thresh}th percentile",
    description="{freq} number of days where temperature is below {tas_per_thresh}th percentile and "
    "precipitation is above {pr_per_thresh}th percentile.",
    abstract="Number of days with temperature below a given percentile and precipitation above a given percentile.",
    cell_methods="time: sum over days",
    compute=indices.cold_and_wet_days,
)

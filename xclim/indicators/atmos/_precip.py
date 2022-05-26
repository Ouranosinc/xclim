"""Precipitation indicator definitions."""
from __future__ import annotations

from inspect import _empty  # noqa

from xclim import indices
from xclim.core import cfchecks
from xclim.core.indicator import (
    Daily,
    Hourly,
    Indicator,
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
    "drought_code",
    "fire_weather_indexes",
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


class HrPrecip(Hourly):
    """Indicator involving hourly pr series."""

    context = "hydro"


rain_on_frozen_ground_days = PrTasxWithIndexing(
    identifier="rain_frzgr",
    units="days",
    standard_name="number_of_days_with_lwe_thickness_of_"
    "precipitation_amount_above_threshold",
    long_name="Number of rain on frozen ground days",
    description="{freq} number of days with rain above {thresh} "
    "after a series of seven days "
    "with average daily temperature below 0℃. "
    "Precipitation is assumed to be rain when the"
    "daily average temperature is above 0℃.",
    cell_methods="",
    compute=indices.rain_on_frozen_ground_days,
)

max_1day_precipitation_amount = PrecipWithIndexing(
    identifier="rx1day",
    units="mm/day",
    standard_name="lwe_thickness_of_precipitation_amount",
    long_name="maximum 1-day total precipitation",
    description="{freq} maximum 1-day total precipitation",
    cell_methods="time: maximum over days",
    compute=indices.max_1day_precipitation_amount,
)

max_n_day_precipitation_amount = Precip(
    identifier="max_n_day_precipitation_amount",
    var_name="rx{window}day",
    units="mm",
    standard_name="lwe_thickness_of_precipitation_amount",
    long_name="maximum {window}-day total precipitation",
    description="{freq} maximum {window}-day total precipitation.",
    cell_methods="time: maximum over days",
    compute=indices.max_n_day_precipitation_amount,
)

wetdays = PrecipWithIndexing(
    identifier="wetdays",
    units="days",
    standard_name="number_of_days_with_lwe_thickness_of_precipitation_amount_at_or_above_threshold",
    long_name="Number of wet days (precip >= {thresh})",
    description="{freq} number of days with daily precipitation over {thresh}.",
    cell_methods="time: sum over days",
    compute=indices.wetdays,
)

wetdays_prop = PrecipWithIndexing(
    identifier="wetdays_prop",
    units="1",
    long_name="Proportion of wet days (precip >= {thresh})",
    description="{freq} proportion of days with precipitation over {thresh}.",
    cell_methods="time: sum over days",
    compute=indices.wetdays_prop,
)

dry_days = PrecipWithIndexing(
    identifier="dry_days",
    units="days",
    standard_name="number_of_days_with_lwe_thickness_of_precipitation_amount_below_threshold",
    long_name="Number of dry days (precip < {thresh})",
    description="{freq} number of days with daily precipitation under {thresh}.",
    cell_methods="time: sum over days",
    compute=indices.dry_days,
)

maximum_consecutive_wet_days = Precip(
    identifier="cwd",
    units="days",
    standard_name="number_of_days_with_lwe_thickness_of_"
    "precipitation_amount_at_or_above_threshold",
    long_name="Maximum consecutive wet days (Precip >= {thresh})",
    description="{freq} maximum number of consecutive days with daily "
    "precipitation over {thresh}.",
    cell_methods="time: sum over days",
    compute=indices.maximum_consecutive_wet_days,
)

maximum_consecutive_dry_days = Precip(
    identifier="cdd",
    units="days",
    standard_name="number_of_days_with_lwe_thickness_of_"
    "precipitation_amount_below_threshold",
    long_name="Maximum consecutive dry days (Precip < {thresh})",
    description="{freq} maximum number of consecutive days with daily "
    "precipitation below {thresh}.",
    cell_methods="time: sum over days",
    compute=indices.maximum_consecutive_dry_days,
)

daily_pr_intensity = PrecipWithIndexing(
    identifier="sdii",
    units="mm/day",
    standard_name="lwe_thickness_of_precipitation_amount",
    long_name="Average precipitation during wet days (SDII)",
    description="{freq} Simple Daily Intensity Index (SDII) : {freq} average precipitation "
    "for days with daily precipitation over {thresh}. This indicator is also known as the 'Simple Daily "
    "Intensity Index' (SDII).",
    cell_methods="",
    compute=indices.daily_pr_intensity,
)

max_pr_intensity = HrPrecip(
    identifier="max_pr_intensity",
    units="mm/h",
    standard_name="precipitation",
    long_name="Maximum precipitation intensity over {window}h duration",
    description="{freq} maximum precipitation intensity over rolling {window}h window.",
    cell_methods="time: max",
    compute=indices.max_pr_intensity,
    duration="{window}",
    keywords="IDF curves",
)

precip_accumulation = PrecipWithIndexing(
    title="Accumulated total precipitation (solid and liquid)",
    identifier="prcptot",
    units="mm",
    standard_name="lwe_thickness_of_precipitation_amount",
    long_name="Total precipitation",
    description="{freq} total precipitation",
    cell_methods="time: sum over days",
    compute=indices.precip_accumulation,
    parameters=dict(tas=None, phase=None),
)

wet_precip_accumulation = PrecipWithIndexing(
    title="Accumulated total precipitation (solid and liquid) during wet days",
    identifier="wet_prcptot",
    units="mm",
    standard_name="lwe_thickness_of_precipitation_amount",
    long_name="Total precipitation",
    description="{freq} total precipitation over wet days, defined as days where precipitation exceeds {thresh}.",
    cell_methods="time: sum over days",
    compute=indices.prcptot,
    parameters={"thresh": {"default": "1 mm/day"}},
)

liquid_precip_accumulation = PrTasxWithIndexing(
    title="Accumulated liquid precipitation.",
    identifier="liquidprcptot",
    units="mm",
    standard_name="lwe_thickness_of_liquid_precipitation_amount",
    long_name="Total liquid precipitation",
    description="{freq} total {phase} precipitation, estimated as precipitation when temperature >= {thresh}",
    cell_methods="time: sum over days",
    compute=indices.precip_accumulation,
    parameters={"tas": {"kind": InputKind.VARIABLE}, "phase": "liquid"},
)

solid_precip_accumulation = PrTasxWithIndexing(
    title="Accumulated solid precipitation.",
    identifier="solidprcptot",
    units="mm",
    standard_name="lwe_thickness_of_snowfall_amount",
    long_name="Total solid precipitation",
    description="{freq} total solid precipitation, estimated as precipitation when temperature < {thresh}",
    cell_methods="time: sum over days",
    compute=indices.precip_accumulation,
    parameters={"tas": {"kind": InputKind.VARIABLE}, "phase": "solid"},
)

drought_code = FireWeather(
    identifier="dc",
    units="",
    standard_name="drought_code",
    long_name="Drought Code",
    description="Numeric rating of the average moisture content of organic layers.",
    compute=indices.drought_code,
    missing="skip",
)

fire_weather_indexes = FireWeather(
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
    compute=indices.fire_weather_indexes,
    missing="skip",
)


last_snowfall = PrecipWithIndexing(
    identifier="last_snowfall",
    standard_name="day_of_year",
    long_name="Date of last snowfall",
    description="{freq} last day where the solid precipitation flux exceeded {thresh}",
    units="",
    compute=indices.last_snowfall,
)

first_snowfall = PrecipWithIndexing(
    identifier="first_snowfall",
    standard_name="day_of_year",
    long_name="Date of first snowfall",
    description="{freq} first day where the solid precipitation flux exceeded {thresh}",
    units="",
    compute=indices.first_snowfall,
)

days_with_snow = PrecipWithIndexing(
    identifier="days_with_snow",
    title="Days with snowfall",
    long_name="Number of days with solid precipitation flux between low and high thresholds.",
    description="{freq} number of days with solid precipitation flux larger than {low} and smaller or equal to {high}.",
    units="days",
    compute=indices.days_with_snow,
)

days_over_precip_thresh = PrecipWithIndexing(
    identifier="days_over_precip_thresh",
    standard_name="number_of_days_with_lwe_thickness_of_precipitation_amount_above_threshold",
    description="{freq} number of days with precipitation above the {pr_per_thresh}th"
    " percentile of {pr_per_period} period."
    " Only days with at least {thresh} are counted.",
    units="days",
    cell_methods="time: sum over days",
    compute=indices.days_over_precip_thresh,
)

days_over_precip_doy_thresh = PrecipWithIndexing(
    identifier="days_over_precip_doy_thresh",
    standard_name="number_of_days_with_lwe_thickness_of_precipitation_amount_above_daily_threshold",
    description="{freq} number of days with precipitation above the {pr_per_thresh}th daily percentile."
    " Only days with at least {thresh} are counted."
    " A {pr_per_window} day(s) window, centred on each calendar day in the"
    " {pr_per_period} period, is used to compute the {pr_per_thresh}th percentile(s).",
    units="days",
    cell_methods="time: sum over days",
    compute=indices.days_over_precip_thresh,
)

high_precip_low_temp = PrTasxWithIndexing(
    identifier="high_precip_low_temp",
    description="{freq} number of days with precipitation above {pr_thresh} and temperature below {tas_thresh}.",
    units="days",
    cell_methods="time: sum over days",
    compute=indices.high_precip_low_temp,
)

fraction_over_precip_doy_thresh = PrecipWithIndexing(
    identifier="fraction_over_precip_doy_thresh",
    description="{freq} fraction of total precipitation due to days with precipitation"
    " above {pr_per_thresh}th daily percentile."
    " Only days with at least {thresh} are included in the total."
    " A {pr_per_window} day(s) window, centred on each calendar day in the"
    " {pr_per_period} period, is used to compute the {pr_per_thresh}th percentile(s).",
    units="",
    cell_methods="",
    compute=indices.fraction_over_precip_thresh,
)

fraction_over_precip_thresh = PrecipWithIndexing(
    identifier="fraction_over_precip_thresh",
    description="{freq} fraction of total precipitation due to days with precipitation"
    " above {pr_per_thresh}th percentile of {pr_per_period} period."
    " Only days with at least {thresh} are included in the total.",
    units="",
    cell_methods="",
    compute=indices.fraction_over_precip_thresh,
)

liquid_precip_ratio = PrTasxWithIndexing(
    identifier="liquid_precip_ratio",
    description="{freq} ratio of rainfall to total precipitation."
    " Rainfall is estimated as precipitation on days where temperature is above {thresh}.",
    abstract="The ratio of total liquid precipitation over the total precipitation. Liquid precipitation is"
    " approximated from total precipitation on days where temperature is above a threshold.",
    units="",
    compute=indices.liquid_precip_ratio,
    parameters={"tas": {"kind": InputKind.VARIABLE}, "prsn": None},
)


dry_spell_frequency = Precip(
    identifier="dry_spell_frequency",
    description="The {freq} number of dry periods of {window} days and more, during which the {op} precipitation "
    "on a window of {window} days is under {thresh}.",
    units="",
    cell_methods="",
    compute=indices.dry_spell_frequency,
)


dry_spell_total_length = Precip(
    identifier="dry_spell_total_length",
    description="The {freq} number of days in dry periods of {window} days and more, during which the {op}"
    "precipitation within windows of {window} days is under {thresh}.",
    units="days",
    cell_methods="",
    compute=indices.dry_spell_total_length,
)

rprctot = PrecipWithIndexing(
    identifier="rprctot",
    description="Proportion of accumulated precipitation arising from convective processes.",
    units="",
    cell_methods="time: sum",
    compute=indices.rprctot,
)


cold_and_dry_days = PrecipWithIndexing(
    identifier="cold_and_dry_days",
    units="days",
    long_name="Cold and dry days",
    title="Cold and dry days",
    description="{freq} number of days where tas < {tas_per_thresh}th percentile and pr < {pr_per_thresh}th percentile",
    cell_methods="time: sum over days",
    compute=indices.cold_and_dry_days,
)

warm_and_dry_days = PrecipWithIndexing(
    identifier="warm_and_dry_days",
    units="days",
    long_name="warm and dry days",
    title="warm and dry days",
    description="{freq} number of days where tas > {tas_per_thresh}th percentile and pr < {pr_per_thresh}th percentile",
    cell_methods="time: sum over days",
    compute=indices.warm_and_dry_days,
)

warm_and_wet_days = PrecipWithIndexing(
    identifier="warm_and_wet_days",
    units="days",
    long_name="warm and wet days",
    title="warm and wet days",
    description="{freq} number of days where tas > {tas_per_thresh}th percentile and pr > {pr_per_thresh}th percentile",
    cell_methods="time: sum over days",
    compute=indices.warm_and_wet_days,
)

cold_and_wet_days = PrecipWithIndexing(
    identifier="cold_and_wet_days",
    units="days",
    long_name="cold and wet days",
    title="cold and wet days",
    description="{freq} number of days where tas < {tas_per_thresh}th percentile and pr > {pr_per_thresh}th percentile",
    cell_methods="time: sum over days",
    compute=indices.cold_and_wet_days,
)

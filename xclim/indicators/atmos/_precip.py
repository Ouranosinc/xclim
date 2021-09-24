# -*- coding: utf-8 -*-
"""Precipitation indicator definitions."""
from inspect import _empty  # noqa

from xclim import indices
from xclim.core import cfchecks
from xclim.core.indicator import Daily, Hourly
from xclim.core.utils import wrapped_partial

__all__ = [
    "rain_on_frozen_ground_days",
    "max_1day_precipitation_amount",
    "max_n_day_precipitation_amount",
    "wetdays",
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
    "high_precip_low_temp",
    "fraction_over_precip_thresh",
    "liquid_precip_ratio",
    "dry_spell_frequency",
    "dry_spell_total_length",
    "wet_precip_accumulation",
]


class Precip(Daily):
    """Indicator involving daily pr series."""

    context = "hydro"


class PrTasx(Daily):
    """Indicator involving pr and one of tas, tasmin or tasmax."""

    context = "hydro"

    @staticmethod
    def cfcheck(pr, tas):
        cfchecks.cfcheck_from_name("pr", pr)
        cfchecks.check_valid(tas, "standard_name", "air_temperature")


class HrPrecip(Hourly):
    """Indicator involving hourly pr series,"""

    context = "hydro"


rain_on_frozen_ground_days = PrTasx(
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

max_1day_precipitation_amount = Precip(
    identifier="rx1day",
    units="mm/day",
    standard_name="lwe_thickness_of_precipitation_amount",
    long_name="maximum 1-day total precipitation",
    description="{freq} maximum 1-day total precipitation",
    cell_methods="time: sum within days time: maximum over days",
    compute=indices.max_1day_precipitation_amount,
)

max_n_day_precipitation_amount = Precip(
    identifier="max_n_day_precipitation_amount",
    var_name="rx{window}day",
    units="mm",
    standard_name="lwe_thickness_of_precipitation_amount",
    long_name="maximum {window}-day total precipitation",
    description="{freq} maximum {window}-day total precipitation.",
    cell_methods="time: sum within days time: maximum over days",
    compute=indices.max_n_day_precipitation_amount,
)

wetdays = Precip(
    identifier="wetdays",
    units="days",
    standard_name="number_of_days_with_lwe_thickness_of_precipitation_amount_at_or_above_threshold",
    long_name="Number of wet days (precip >= {thresh})",
    description="{freq} number of days with daily precipitation over {thresh}.",
    cell_methods="time: sum within days time: sum over days",
    compute=indices.wetdays,
)

dry_days = Precip(
    identifier="dry_days",
    units="days",
    standard_name="number_of_days_with_lwe_thickness_of_precipitation_amount_below_threshold",
    long_name="Number of dry days (precip < {thresh})",
    description="{freq} number of days with daily precipitation under {thresh}.",
    cell_methods="time: sum within days time: sum over days",
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
    cell_methods="time: sum within days time: sum over days",
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
    cell_methods="time: sum within days time: sum over days",
    compute=indices.maximum_consecutive_dry_days,
)

daily_pr_intensity = Precip(
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

precip_accumulation = Precip(
    title="Accumulated total precipitation (solid and liquid)",
    identifier="prcptot",
    units="mm",
    standard_name="lwe_thickness_of_precipitation_amount",
    long_name="Total precipitation",
    description="{freq} total precipitation",
    cell_methods="time: sum within days time: sum over days",
    compute=wrapped_partial(indices.precip_accumulation, tas=None, phase=None),
)

wet_precip_accumulation = Precip(
    title="Accumulated total precipitation (solid and liquid) during wet days",
    identifier="wet_prcptot",
    units="mm",
    standard_name="lwe_thickness_of_precipitation_amount",
    long_name="Total precipitation",
    description="{freq} total precipitation over wet days, defined as days where precipitation exceeds {thresh}.",
    cell_methods="time: sum within days time: sum over days",
    compute=wrapped_partial(indices.prcptot, suggested={"thresh": "1 mm/day"}),
)

liquid_precip_accumulation = PrTasx(
    title="Accumulated liquid precipitation.",
    identifier="liquidprcptot",
    units="mm",
    standard_name="lwe_thickness_of_liquid_precipitation_amount",
    long_name="Total liquid precipitation",
    description="{freq} total {phase} precipitation, estimated as precipitation when temperature >= {thresh}",
    cell_methods="time: sum within days time: sum over days",
    compute=wrapped_partial(
        indices.precip_accumulation, suggested={"tas": _empty}, phase="liquid"
    ),  # _empty is added to un-optionalize the argument.
)

solid_precip_accumulation = PrTasx(
    title="Accumulated solid precipitation.",
    identifier="solidprcptot",
    units="mm",
    standard_name="lwe_thickness_of_snowfall_amount",
    long_name="Total solid precipitation",
    description="{freq} total solid precipitation, estimated as precipitation when temperature < {thresh}",
    cell_methods="time: sum within days time: sum over days",
    compute=wrapped_partial(
        indices.precip_accumulation, suggested={"tas": _empty}, phase="solid"
    ),
)

drought_code = Precip(
    identifier="dc",
    units="",
    standard_name="drought_code",
    long_name="Drought Code",
    description="Numeric rating of the average moisture content of organic layers.",
    compute=indices.drought_code,
    missing="skip",
)

fire_weather_indexes = Precip(
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


last_snowfall = Precip(
    identifier="last_snowfall",
    standard_name="day_of_year",
    long_name="Date of last snowfall",
    description="{freq} last day where the solid precipitation flux exceeded {thresh}",
    units="",
    compute=indices.last_snowfall,
)

first_snowfall = Precip(
    identifier="first_snowfall",
    standard_name="day_of_year",
    long_name="Date of first snowfall",
    description="{freq} first day where the solid precipitation flux exceeded {thresh}",
    units="",
    compute=indices.first_snowfall,
)

days_with_snow = Precip(
    identifier="days_with_snow",
    title="Days with snowfall",
    long_name="Number of days with solid precipitation flux between low and high thresholds.",
    description="{freq} number of days with solid precipitation flux larger than {low} and smaller or equal to {high}.",
    units="days",
    compute=indices.days_with_snow,
)

days_over_precip_thresh = Precip(
    identifier="days_over_precip_thresh",
    standard_name="number_of_days_with_lwe_thickness_of_precipitation_amount_above_threshold",
    description="{freq} number of days with precipitation above a daily percentile."
    " Only days with at least {thresh} are counted.",
    units="days",
    cell_methods="time: sum over days",
    compute=indices.days_over_precip_thresh,
)


high_precip_low_temp = PrTasx(
    identifier="high_precip_low_temp",
    description="{freq} number of days with precipitation above {pr_thresh} and temperature below {tas_thresh}.",
    units="days",
    cell_methods="time: sum over days",
    compute=indices.high_precip_low_temp,
)


fraction_over_precip_thresh = Precip(
    identifier="fraction_over_precip_thresh",
    description="{freq} fraction of total precipitation due to days with precipitation above a daily percentile."
    " Only days with at least {thresh} are included in the total.",
    units="",
    cell_methods="",
    compute=indices.fraction_over_precip_thresh,
)


liquid_precip_ratio = PrTasx(
    identifier="liquid_precip_ratio",
    description="{freq} ratio of rainfall to total precipitation."
    " Rainfall is estimated as precipitation on days where temperature is above {thresh}.",
    abstract="The ratio of total liquid precipitation over the total precipitation. Liquid precipitation is"
    " approximated from total precipitation on days where temperature is above a threshold.",
    units="",
    compute=wrapped_partial(
        indices.liquid_precip_ratio, suggested={"tas": _empty}, prsn=None
    ),
)


dry_spell_frequency = Precip(
    identifier="dry_spell_frequency",
    description="The {freq} number of dry periods of {window} days and more, during which the accumulated "
    "precipitation on a window of {window} days is under {thresh}.",
    units="",
    cell_methods="",
    compute=indices.dry_spell_frequency,
)


dry_spell_total_length = Precip(
    identifier="dry_spell_total_length",
    description="The {freq} number of days in dry periods of {window} days and more, during which the accumulated "
    "precipitation on a window of {window} days is under {thresh}.",
    units="d",
    cell_methods="",
    compute=indices.dry_spell_total_length,
)

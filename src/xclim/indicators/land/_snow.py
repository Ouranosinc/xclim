"""Snow indicator definitions."""

from __future__ import annotations

import warnings

from xclim import indices as xci
from xclim.core.indicator import Daily, ResamplingIndicatorWithIndexing
from xclim.indicators.convert import snd_to_snw as _snd_to_snw
from xclim.indicators.convert import snw_to_snd as _snw_to_snd

__all__ = [
    "blowing_snow",
    "holiday_snow_and_snowfall_days",
    "holiday_snow_days",
    "snd_days_above",
    "snd_max_doy",
    "snd_season_end",
    "snd_season_length",
    "snd_season_start",
    "snd_storm_days",
    "snd_to_snw",
    "snow_depth",
    "snow_melt_we_max",
    "snw_days_above",
    "snw_max",
    "snw_max_doy",
    "snw_season_end",
    "snw_season_length",
    "snw_season_start",
    "snw_storm_days",
    "snw_to_snd",
]


def snd_to_snw(*args, **kwargs):  # numpydoc ignore=GL08
    warnings.warn(
        "The `snd_to_snw` indicator has been moved to `xclim.indicators.convert`. "
        "This alias will be removed in a future version.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _snd_to_snw(*args, **kwargs)


def snw_to_snd(*args, **kwargs):  # numpydoc ignore=GL08
    warnings.warn(
        "The `snw_to_snd` indicator has been moved to `xclim.indicators.convert`. "
        "This alias will be removed in a future version.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _snw_to_snd(*args, **kwargs)


# We need to declare the base class here so the `land` module is detected automatically.
class Snow(Daily):
    """Indicators dealing with snow variables."""

    keywords = "snow"


class SnowWithIndexing(ResamplingIndicatorWithIndexing):
    """Indicators dealing with snow variables, allowing indexing."""

    src_freq = "D"
    keywords = "snow"


snd_season_length = SnowWithIndexing(
    identifier="snd_season_length",
    units="days",
    long_name="Snow cover duration",
    description=(
        "The duration of the snow season, starting with at least {window} days with snow depth above {thresh} "
        "and ending with at least {window} days with snow depth under {thresh}."
    ),
    compute=xci.snd_season_length,
)

snw_season_length = SnowWithIndexing(
    identifier="snw_season_length",
    units="days",
    long_name="Snow cover duration",
    description=(
        "The duration of the snow season, starting with at least {window} days with snow amount above {thresh} "
        "and ending with at least {window} days with snow amount under {thresh}."
    ),
    compute=xci.snw_season_length,
)

snd_season_start = Snow(
    identifier="snd_season_start",
    standard_name="day_of_year",
    long_name="Start date of continuous snow depth cover",
    description="Day of year when snow depth is above or equal to {thresh} for {window} consecutive days.",
    abstract="The first date on which snow depth is greater than or equal to a given threshold "
    "for a given number of consecutive days.",
    units="",
    compute=xci.snd_season_start,
)

snw_season_start = Snow(
    identifier="snw_season_start",
    standard_name="day_of_year",
    long_name="Start date of continuous snow amount cover",
    description="Day of year when snow amount is above or equal to {thresh} for {window} consecutive days.",
    abstract="The first date on which snow amount is greater than or equal to a given threshold "
    "for a given number of consecutive days.",
    units="",
    compute=xci.snw_season_start,
)

snd_season_end = Snow(
    identifier="snd_season_end",
    standard_name="day_of_year",
    long_name="End date of continuous snow depth cover",
    description="Day of year when snow depth is below {thresh} for {window} consecutive days.",
    abstract="The first date on which snow depth is below a given threshold for a given number of consecutive days.",
    units="",
    compute=xci.snd_season_end,
)

snw_season_end = Snow(
    identifier="snw_season_end",
    standard_name="day_of_year",
    long_name="End date of continuous snow amount cover",
    description="Day of year when snow amount is below {thresh} for {window} consecutive days.",
    abstract="The first date on which snow amount is below a given threshold for a given number of consecutive days.",
    units="",
    compute=xci.snw_season_end,
)

snd_max_doy = SnowWithIndexing(
    title="Day of year of maximum snow depth",
    identifier="snd_max_doy",
    standard_name="day_of_year",
    var_name="{freq}_snd_max_doy",
    long_name="Day of the year when snow depth reaches its maximum value",
    description="The {freq} day of the year when snow depth reaches its maximum value.",
    abstract="Day of the year when snow depth reaches its maximum value.",
    units="",
    _partial=True,
    compute=xci.snd_max_doy,
)

snow_melt_we_max = Snow(
    title="Maximum snow melt",
    identifier="snow_melt_we_max",
    standard_name="change_over_time_in_surface_snow_amount",
    var_name="{freq}_snow_melt_we_max",
    long_name="Maximum snow melt",
    description="The {freq} maximum negative change in melt amount over {window} days.",
    abstract="The water equivalent of the maximum snow melt.",
    units="kg m-2",
    compute=xci.snow_melt_we_max,
)

snw_max = SnowWithIndexing(
    title="Maximum snow amount",
    identifier="snw_max",
    standard_name="surface_snow_amount",
    var_name="{freq}_snw_max",
    long_name="Maximum snow amount equivalent",
    description="The {freq} maximum snow amount equivalent on the surface.",
    abstract="The maximum snow amount equivalent on the surface.",
    units="kg m-2",
    compute=xci.snw_max,
)

snw_max_doy = SnowWithIndexing(
    title="Day of year of maximum snow amount",
    identifier="snw_max_doy",
    standard_name="day_of_year",
    var_name="{freq}_snw_max_doy",
    long_name="Day of year of maximum daily snow amount equivalent",
    description="The {freq} day of year when snow amount equivalent on the surface reaches its maximum.",
    abstract="The day of year when snow amount equivalent on the surface reaches its maximum.",
    units="",
    compute=xci.snw_max_doy,
)

melt_and_precip_max = Snow(
    title="Water equivalent maximum from precipitation and snow melt",
    identifier="melt_and_precip_max",
    var_name="{freq}_melt_and_precip_max",
    long_name="Water equivalent maximum from precipitation and snow melt",
    description="The {freq} maximum precipitation flux and negative change in snow amount over {window} days.",
    abstract="Maximum water input from precipitation flux and snow melt over a given window of days.",
    units="kg m-2",
    compute=xci.melt_and_precip_max,
)


snd_storm_days = SnowWithIndexing(
    title="Winter storm days",
    identifier="snd_storm_days",
    var_name="{freq}_snd_storm_days",
    long_name="Days with snowfall depth at or above a given threshold",
    description="The {freq} number of days with snowfall depth accumulation above {thresh}.",
    units="days",
    compute=xci.snd_storm_days,
)

snw_storm_days = SnowWithIndexing(
    title="Winter storm days",
    identifier="snw_storm_days",
    var_name="{freq}_snw_storm_days",
    long_name="Days with snowfall amount at or above a given threshold",
    description="The {freq} number of days with snowfall amount accumulation above {thresh}.",
    units="days",
    compute=xci.snw_storm_days,
)


blowing_snow = Snow(
    title="Blowing snow days",
    identifier="blowing_snow",
    var_name="{freq}_blowing_snow",
    long_name="Days with snowfall and wind speed at or above given thresholds",
    description="The {freq} number of days with snowfall over last {window} days above {snd_thresh} and wind speed "
    "above {sfcWind_thresh}.",
    abstract="The number of days with snowfall, snow depth, and windspeed over given thresholds for a period of days.",
    units="days",
    compute=xci.blowing_snow,
)

snow_depth = SnowWithIndexing(
    title="Mean snow depth",
    identifier="snow_depth",
    units="cm",
    standard_name="surface_snow_thickness",
    long_name="Mean of daily snow depth",
    description="The {freq} mean of daily mean snow depth.",
    abstract="Mean of daily snow depth.",
    cell_methods="time: mean over days",
    compute=xci.snow_depth,
)


snd_days_above = SnowWithIndexing(
    title="Days with snow (depth)",
    identifier="snd_days_above",
    units="days",
    long_name="Number of days with snow",
    description="The {freq} number of days with snow depth greater than or equal to {thresh}.",
    abstract="Number of days when the snow depth is greater than or equal to a given threshold.",
    compute=xci.snd_days_above,
)

snw_days_above = SnowWithIndexing(
    title="Days with snow (amount)",
    identifier="snw_days_above",
    units="days",
    long_name="Number of days with snow",
    description="The {freq} number of days with snow amount greater than or equal to {thresh}.",
    abstract="Number of days when the snow amount is greater than or equal to a given threshold.",
    compute=xci.snw_days_above,
)

holiday_snow_days = Snow(
    title="Christmas snow days",
    identifier="holiday_snow_days",
    units="days",
    long_name="Number of holiday days with snow",
    description="The total number of days where snow on the ground was greater than or equal to {snd_thresh} "
    "occurring on {date_start} and ending on {date_end}.",
    abstract="The total number of days where there is a significant amount of snow on the ground on December 25th.",
    compute=xci.holiday_snow_days,
)

holiday_snow_and_snowfall_days = Snow(
    title="Perfect Christmas snow days",
    identifier="holiday_snow_and_snowfall_days",
    units="days",
    long_name="Number of holiday days with snow and snowfall",
    description="The total number of days where snow on the ground was greater than or equal to {snd_thresh} "
    "and snowfall was greater than or equal to {prsn_thresh} occurring on {date_start} and ending on {date_end}.",
    abstract="The total number of days where there is a significant amount of snow on the ground "
    "and a measurable snowfall occurring on December 25th.",
    compute=xci.holiday_snow_and_snowfall_days,
)

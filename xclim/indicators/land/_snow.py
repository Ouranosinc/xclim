from xclim import indices as xci
from xclim.core.cfchecks import generate_cfcheck
from xclim.core.indicator import Daily, Daily2D

__all__ = [
    "blowing_snow",
    "snow_cover_duration",
    "continuous_snow_cover_start",
    "continuous_snow_cover_end",
    "snd_max_doy",
    "snow_melt_we_max",
    "winter_storm",
]


class SnowDepth(Daily):
    cfcheck = staticmethod(generate_cfcheck("snd"))


class SnowCover(Daily):
    cfcheck = staticmethod(generate_cfcheck("snc"))


class SnowAmount(Daily):
    cfcheck = staticmethod(generate_cfcheck("snw"))


class SnwPr(Daily2D):
    cfcheck = staticmethod(generate_cfcheck("snw", "pr"))


class SndWs(Daily2D):
    cfcheck = staticmethod(generate_cfcheck("snd", "sfcWind"))


snow_cover_duration = SnowDepth(
    identifier="snow_cover_duration",
    units="days",
    long_name="Number of days with snow depth above threshold",
    description="{freq} number of days with snow depth greater or equal to {thresh}",
    compute=xci.snow_cover_duration,
)

continuous_snow_cover_start = SnowDepth(
    identifier="continuous_snow_cover_start",
    standard_name="day_of_year",
    long_name="Start date of continuous snow cover",
    description="Day of year when snow depth is above {thresh} for {window} consecutive days.",
    units="",
    compute=xci.continuous_snow_cover_start,
)

continuous_snow_cover_end = SnowDepth(
    identifier="continuous_snow_cover_end",
    standard_name="day_of_year",
    long_name="Start date of continuous snow cover",
    description="Day of year when snow depth is above {thresh} for {window} consecutive days.",
    units="",
    compute=xci.continuous_snow_cover_end,
)

snd_max_doy = SnowDepth(
    identifier="snd_max_doy",
    var_name="{freq}_snd_max_doy",
    long_name="Date when snow depth reaches its maximum value.",
    description="{freq} day of year when snow depth reaches its maximum value.",
    units="",
    _partial=True,
    compute=xci.snd_max_doy,
)

snow_melt_we_max = SnowAmount(
    identifier="snow_melt_we_max",
    standard_name="change_over_time_in_surface_snow_amount",
    var_name="{freq}_snow_melt_we_max",
    description="{freq} maximum negative change in melt amount over {window} days.",
    units="kg m-2",
    compute=xci.snow_melt_we_max,
)


melt_and_precip_max = SnwPr(
    identifier="melt_and_precip_max",
    var_name="{freq}_melt_and_precip_max",
    description="{freq} maximum precipitation flux and negative change in snow amount over {window} days.",
    units="kg m-2",
    compute=xci.melt_and_precip_max,
)


winter_storm = SnowDepth(
    identifier="winter_storm",
    var_name="{freq}_winter_storm",
    description="{freq} number of days with snowfall accumulation above {thresh}.",
    units="days",
    compute=xci.winter_storm,
)


blowing_snow = SndWs(
    identifier="blowing_snow",
    var_name="{freq}_blowing_snow",
    description="{freq} number of days with snowfall over last {window} days above {snd_thresh} and wind speed above "
    "{sfcWind_thresh}.",
    units="days",
    compute=xci.blowing_snow,
)

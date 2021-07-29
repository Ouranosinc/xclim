from xclim import indices as xci
from xclim.core.indicator import Daily

__all__ = [
    "blowing_snow",
    "snow_cover_duration",
    "continuous_snow_cover_start",
    "continuous_snow_cover_end",
    "snd_max_doy",
    "snow_melt_we_max",
    "winter_storm",
    "snow_depth",
]


# We need to declare the base class here so the `land` module is detected automatically.
class Snow(Daily):
    """Indicators dealing with snow variables."""


snow_cover_duration = Snow(
    identifier="snow_cover_duration",
    units="days",
    long_name="Number of days with snow depth above threshold",
    description="{freq} number of days with snow depth greater or equal to {thresh}",
    compute=xci.snow_cover_duration,
)

continuous_snow_cover_start = Snow(
    identifier="continuous_snow_cover_start",
    standard_name="day_of_year",
    long_name="Start date of continuous snow cover",
    description="Day of year when snow depth is above {thresh} for {window} consecutive days.",
    units="",
    compute=xci.continuous_snow_cover_start,
)

continuous_snow_cover_end = Snow(
    identifier="continuous_snow_cover_end",
    standard_name="day_of_year",
    long_name="Start date of continuous snow cover",
    description="Day of year when snow depth is above {thresh} for {window} consecutive days.",
    units="",
    compute=xci.continuous_snow_cover_end,
)

snd_max_doy = Snow(
    identifier="snd_max_doy",
    var_name="{freq}_snd_max_doy",
    long_name="Date when snow depth reaches its maximum value.",
    description="{freq} day of year when snow depth reaches its maximum value.",
    units="",
    _partial=True,
    compute=xci.snd_max_doy,
)

snow_melt_we_max = Snow(
    identifier="snow_melt_we_max",
    standard_name="change_over_time_in_surface_snow_amount",
    var_name="{freq}_snow_melt_we_max",
    description="{freq} maximum negative change in melt amount over {window} days.",
    units="kg m-2",
    compute=xci.snow_melt_we_max,
)


melt_and_precip_max = Snow(
    identifier="melt_and_precip_max",
    var_name="{freq}_melt_and_precip_max",
    description="{freq} maximum precipitation flux and negative change in snow amount over {window} days.",
    units="kg m-2",
    compute=xci.melt_and_precip_max,
)


winter_storm = Snow(
    identifier="winter_storm",
    var_name="{freq}_winter_storm",
    description="{freq} number of days with snowfall accumulation above {thresh}.",
    units="days",
    compute=xci.winter_storm,
)


blowing_snow = Snow(
    identifier="blowing_snow",
    var_name="{freq}_blowing_snow",
    description="{freq} number of days with snowfall over last {window} days above {snd_thresh} and wind speed above "
    "{sfcWind_thresh}.",
    units="days",
    compute=xci.blowing_snow,
)

snow_depth = Snow(
    identifier="snow_depth",
    units="cm",
    standard_name="surface_snow_thickness",
    long_name="Mean of daily snow depth",
    description="{freq} mean of daily mean snow depth.",
    cell_methods="time: mean within days time: mean over days",
    compute=xci.snow_depth,
)

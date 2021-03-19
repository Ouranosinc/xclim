from xclim import indices as xci
from xclim.core.cfchecks import check_valid
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
    @staticmethod
    def cfcheck(snd):
        check_valid(snd, "standard_name", "surface_snow_thickness")


class SnowCover(Daily):
    @staticmethod
    def cfcheck(snc):
        check_valid(snc, "standard_name", "surface_snow_area_fraction")


class SnowWaterEq(Daily):
    @staticmethod
    def cfcheck(swe):
        check_valid(
            swe,
            "standard_name",
            [
                "liquid_water_content_of_surface_snow",
                "liquid_water_content_of_snow_layer",
            ],
        )


class SWEPr(Daily2D):
    @staticmethod
    def cfcheck(swe, pr):
        check_valid(
            swe,
            "standard_name",
            [
                "liquid_water_content_of_surface_snow",
                "liquid_water_content_of_snow_layer",
            ],
        )
        check_valid(
            pr, "standard_name", ["precipitation_flux", "lwe_precipitation_rate"]
        )


class SndWs(Daily2D):
    @staticmethod
    def cfcheck(snd, sfcWind):
        check_valid(snd, "standard_name", "surface_snow_thickness")
        check_valid(sfcWind, "standard_name", "wind_speed")


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

snow_melt_we_max = SnowWaterEq(
    identifier="snow_melt_we_max",
    standard_name="change_over_time_in_surface_snow_amount",
    var_name="{freq}_snow_melt_we_max",
    description="{freq} maximum negative change in melt water equivalent over {window} days.",
    units="kg m-2",
    compute=xci.snow_melt_we_max,
)


melt_and_precip_max = SWEPr(
    identifier="melt_and_precip_max",
    var_name="{freq}_melt_and_precip_max",
    description="{freq} maximum precipitation flux and negative change in snow water equivalent over {window} days.",
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

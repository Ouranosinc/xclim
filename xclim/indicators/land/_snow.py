from xclim.core.cfchecks import check_valid
from xclim.core.indicator import Daily
from xclim.indices import (
    continuous_snow_cover_end,
    continuous_snow_cover_start,
    snow_cover_duration,
)


class SnowDepth(Daily):
    @staticmethod
    def cfcheck(snd):
        check_valid(snd, "standard_name", "surface_snow_thickness")


class SnowCover(Daily):
    @staticmethod
    def cfcheck(snc):
        check_valid(snc, "standard_name", "surface_snow_area_fraction ")


snow_cover_duration = SnowDepth(
    identifier="snow_cover_duration",
    units="days",
    long_name="Number of days with snow depth above threshold",
    description="{freq} number of days with snow depth greater or equal to {thresh}",
    compute=snow_cover_duration,
)

continuous_snow_cover_start = SnowDepth(
    identifier="continuous_snow_cover_start",
    standard_name="day_of_year",
    long_name="Start date of continuous snow cover",
    description="Day of year when snow depth is above {thresh} for {window} consecutive days.",
    units="",
    compute=continuous_snow_cover_start,
)

continuous_snow_cover_end = SnowDepth(
    identifier="continuous_snow_cover_end",
    standard_name="day_of_year",
    long_name="Start date of continuous snow cover",
    description="Day of year when snow depth is above {thresh} for {window} consecutive days.",
    units="",
    compute=continuous_snow_cover_end,
)

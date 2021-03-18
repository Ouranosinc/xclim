from xclim import indices as xci
from xclim.core.cfchecks import check_valid
from xclim.core.indicator import Daily
from xclim.core.units import declare_units
from xclim.core.utils import wrapped_partial

__all__ = [
    "snow_cover_duration",
    "continuous_snow_cover_start",
    "continuous_snow_cover_end",
    "snd_max_doy",
    "snow_melt_we_max",
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


class SWEPr(Daily):
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
    compute=declare_units(da="[length]")(
        wrapped_partial(
            xci.generic.select_resample_op,
            op=xci.generic.doymax,
            suggested=dict(freq="AS-JUL"),
        )
    ),
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

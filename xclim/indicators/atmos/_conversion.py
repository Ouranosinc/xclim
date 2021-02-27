"""Atmospheric conversion definitions."""

from xclim import indices
from xclim.core.indicator import Indicator
from xclim.core.utils import wrapped_partial

__all__ = [
    "tg",
    "wind_speed_from_vector",
    "wind_vector_from_speed",
    "saturation_vapor_pressure",
    "relative_humidity_from_dewpoint",
    "relative_humidity",
    "specific_humidity",
    "snowfall_approximation",
    "rain_approximation",
]


class Converter(Indicator):
    """Class for indicators doing variable conversion (dimension-independent 1-to-1 computation)."""

    missing = "skip"


tg = Converter(
    identifier="tg",
    nvar=2,
    units="K",
    standard_name="air_temperature",
    long_name="Daily mean temperature",
    description="Estimated mean temperature from maximum and minimum temperatures",
    cell_methods="time: mean within days",
    compute=indices.tas,
)


wind_speed_from_vector = Converter(
    identifier="wind_speed_from_vector",
    nvar=2,
    var_name=["sfcWind", "sfcWindfromdir"],
    units=["m s-1", "degree"],
    standard_name=["wind_speed", "wind_from_direction"],
    description=[
        "Wind speed computed as the magnitude of the (uas, vas) vector.",
        "Wind direction computed as the angle of the (uas, vas) vector. A direction of 0° is attributed to winds with a speed under {calm_wind_thresh}.",
    ],
    long_name=["Near-Surface Wind Speed", "Near-Surface Wind from Direction"],
    cell_methods="",
    compute=indices.uas_vas_2_sfcwind,
)


wind_vector_from_speed = Converter(
    identifier="wind_vector_from_speed",
    nvar=2,
    var_name=["uas", "vas"],
    units=["m s-1", "m s-1"],
    standard_name=["eastward_wind", "northward_wind"],
    long_name=["Near-Surface Eastward Wind", "Near-Surface Northward Wind"],
    description=[
        "Eastward wind speed computed from its speed and direction of origin.",
        "Northward wind speed computed from its speed and direction of origin.",
    ],
    cell_methods="",
    compute=indices.sfcwind_2_uas_vas,
)


saturation_vapor_pressure = Converter(
    identifier="e_sat",
    nvar=1,
    units="Pa",
    long_name="Saturation vapor pressure",
    description=lambda **kws: (
        "The saturation vapor pressure was calculated from a temperature "
        "according to the {method} method."
    )
    + (
        " The computation was done in reference to ice for temperatures below {ice_thresh}."
        if kws["ice_thresh"] is not None
        else ""
    ),
    compute=indices.saturation_vapor_pressure,
)


relative_humidity_from_dewpoint = Converter(
    identifier="rh_fromdewpoint",
    nvar=2,
    units="%",
    long_name="Relative Humidity",
    standard_name="relative_humidity",
    title="Relative humidity from temperature and dewpoint temperature.",
    description=lambda **kws: (
        "Computed from temperature, and dew point temperature through the "
        "saturation vapor pressures, which were calculated "
        "according to the {method} method."
    )
    + (
        " The computation was done in reference to ice for temperatures below {ice_thresh}."
        if kws["ice_thresh"] is not None
        else ""
    ),
    compute=wrapped_partial(
        indices.relative_humidity,
        huss=None,
        ps=None,
        invalid_values="mask",
    ),
)


relative_humidity = Converter(
    identifier="rh",
    nvar=3,
    units="%",
    long_name="Relative Humidity",
    standard_name="relative_humidity",
    title="Relative humidity from temperature, pressure and specific humidity.",
    description=lambda **kws: (
        "Computed from temperature, specific humidity and pressure through the "
        "saturation vapor pressure, which was calculated from temperature "
        "according to the {method} method."
    )
    + (
        " The computation was done in reference to ice for temperatures below {ice_thresh}."
        if kws["ice_thresh"] is not None
        else ""
    ),
    compute=wrapped_partial(
        indices.relative_humidity, dtas=None, invalid_values="mask"
    ),
)


specific_humidity = Converter(
    identifier="huss",
    nvar=3,
    units="",
    long_name="Specific Humidity",
    standard_name="specific_humidity",
    description=lambda **kws: (
        "Computed from temperature, relative humidity and pressure through the "
        "saturation vapor pressure, which was calculated from temperature "
        "according to the {method} method."
    )
    + (
        " The computation was done in reference to ice for temperatures below {ice_thresh}."
        if kws["ice_thresh"] is not None
        else ""
    ),
    compute=wrapped_partial(indices.specific_humidity, invalid_values="mask"),
)


snowfall_approximation = Converter(
    identifier="prsn",
    nvar=2,
    units="kg m-2 s-1",
    standard_name="solid_precipitation_flux",
    long_name="Solid precipitation",
    description=(
        "Solid precipitation estimated from total precipitation and temperature"
        " with method {method} and threshold temperature {thresh}."
    ),
    compute=indices.snowfall_approximation,
)

rain_approximation = Converter(
    identifier="prlp",
    nvar=2,
    units="kg m-2 s-1",
    standard_name="precipitation_flux",
    long_name="Liquid precipitation",
    description=(
        "Liquid precipitation estimated from total precipitation and temperature"
        " with method {method} and threshold temperature {thresh}."
    ),
    compute=indices.rain_approximation,
)

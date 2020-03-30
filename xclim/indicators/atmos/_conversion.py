from xclim import indices
from xclim.core.indicator import Indicator
from xclim.core.utils import wrapped_partial


__all__ = [
    "tg",
    "wind_speed_from_vector",
    "saturation_vapor_pressure",
    "relative_humidity_from_dewpoint",
    "relative_humidity",
    "specific_humidity",
]


class Converter(Indicator):
    """Class for indicators doing variable conversion (dimension-independent 1-to-1 computation)"""

    def validate(self, da):
        """Input validation."""


tg = Converter(
    identifier="tg",
    _nvar=2,
    units="K",
    standard_name="air_temperature",
    long_name="Daily mean temperature",
    description="Estimated mean temperature from maximum and minimum temperatures",
    cell_methods="",
    compute=indices.tas,
)


wind_speed_from_vector = Converter(
    identifier="sfcWind",
    _nvar=2,
    units="m s-1",
    standard_name="wind_speed",
    description="Wind speed computed as the magnitude of the (uas, vas) vector.",
    long_name="Near-Surface Wind Speed",
    cell_methods="",
    compute=wrapped_partial(indices.uas_vas_2_sfcwind, return_direction=False),
)


saturation_vapor_pressure = Converter(
    identifier="e_sat",
    _nvar=1,
    units="Pa",
    long_name="Saturation vapor pressure",
    description=lambda **kws: (
        "The saturation vapor pressure was calculated from the temperature "
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
    identifier="rh",
    _nvar=2,
    units="%",
    long_name="Relative Humidity",
    standard_name="relative_humidity",
    description="Computed from temperature and dew point temperature.",
    compute=wrapped_partial(
        indices.relative_humidity,
        huss=None,
        ps=None,
        method="dewpoint",
        invalid_values="mask",
    ),
)


relative_humidity = Converter(
    identifier="rh",
    _nvar=3,
    units="%",
    long_name="Relative Humidity",
    standard_name="relative_humidity",
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
    _nvar=3,
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

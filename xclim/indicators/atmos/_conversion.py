"""Atmospheric conversion definitions."""
from __future__ import annotations

from inspect import _empty  # noqa

from xclim import indices
from xclim.core.indicator import Indicator
from xclim.core.utils import InputKind

__all__ = [
    "humidex",
    "heat_index",
    "tg",
    "wind_speed_from_vector",
    "wind_vector_from_speed",
    "saturation_vapor_pressure",
    "relative_humidity_from_dewpoint",
    "relative_humidity",
    "specific_humidity",
    "specific_humidity_from_dewpoint",
    "snowfall_approximation",
    "rain_approximation",
    "wind_chill_index",
    "potential_evapotranspiration",
    "water_budget_from_tas",
    "water_budget",
    "corn_heat_units",
    "universal_thermal_climate_index",
    "mean_radiant_temperature",
]


class Converter(Indicator):
    """Class for indicators doing variable conversion (dimension-independent 1-to-1 computation)."""


humidex = Converter(
    identifier="humidex",
    units="C",
    standard_name="air_temperature",
    long_name="Humidex index",
    description="Humidex index describing the temperature felt by the average person in response to relative humidity.",
    cell_methods="",
    abstract="The humidex describes the temperature felt by a person when relative humidity is taken into account. "
    "It can be interpreted as the equivalent temperature felt when the air is dry.",
    compute=indices.humidex,
)

heat_index = Converter(
    identifier="heat_index",
    units="C",
    standard_name="air_temperature",
    long_name="Heat index",
    description="Perceived temperature after relative humidity is taken into account.",
    cell_methods="",
    abstract="The heat index is an estimate of the temperature felt by a person in the shade "
    "when relative humidity is taken into account.",
    compute=indices.heat_index,
)

tg = Converter(
    identifier="tg",
    units="K",
    standard_name="air_temperature",
    long_name="Daily mean temperature",
    description="Estimated mean temperature from maximum and minimum temperatures",
    cell_methods="time: mean within days",
    abstract="The average daily temperature assuming a symmetrical temperature distribution (Tg = (Tx + Tn) / 2).",
    compute=indices.tas,
)


wind_speed_from_vector = Converter(
    identifier="wind_speed_from_vector",
    var_name=["sfcWind", "sfcWindfromdir"],
    units=["m s-1", "degree"],
    standard_name=["wind_speed", "wind_from_direction"],
    description=[
        "Wind speed computed as the magnitude of the (uas, vas) vector.",
        "Wind direction computed as the angle of the (uas, vas) vector."
        " A direction of 0° is attributed to winds with a speed under {calm_wind_thresh}.",
    ],
    long_name=["Near-surface wind speed", "Near-surface wind from direction"],
    cell_methods="",
    abstract="Calculation of the magnitude and direction of the wind speed "
    "from the two components west-east and south-north.",
    compute=indices.uas_vas_2_sfcwind,
)


wind_vector_from_speed = Converter(
    identifier="wind_vector_from_speed",
    var_name=["uas", "vas"],
    units=["m s-1", "m s-1"],
    standard_name=["eastward_wind", "northward_wind"],
    long_name=["Near-surface eastward wind", "Near-surface northward wind"],
    description=[
        "Eastward wind speed computed from the magnitude of its speed and direction of origin.",
        "Northward wind speed computed from magnitude of its speed and direction of origin.",
    ],
    cell_methods="",
    abstract="Calculation of the two components (west-east and north-south) of the wind "
    "from the magnitude of its speed and direction of origin.",
    compute=indices.sfcwind_2_uas_vas,
)


saturation_vapor_pressure = Converter(
    identifier="e_sat",
    units="Pa",
    long_name="Saturation vapour pressure",
    description=lambda **kws: (
        "The saturation vapour pressure was calculated from a temperature according to the {method} method."
    )
    + (
        " The computation was done in reference to ice for temperatures below {ice_thresh}."
        if kws["ice_thresh"] is not None
        else ""
    ),
    title="Saturation vapour pressure (e_sat)",
    abstract="Calculation of the saturation vapour pressure from the temperature, according to a given method. "
    "If ice_thresh is given, the calculation is done with reference to ice for temperatures below this threshold.",
    compute=indices.saturation_vapor_pressure,
)


relative_humidity_from_dewpoint = Converter(
    identifier="hurs_fromdewpoint",
    units="%",
    var_name="hurs",
    long_name="Relative humidity",
    standard_name="relative_humidity",
    title="Relative humidity from temperature and dewpoint temperature",
    description=lambda **kws: (
        "Computed from temperature, and dew point temperature through the "
        "saturation vapour pressures, which were calculated "
        "according to the {method} method."
    )
    + (
        " The computation was done in reference to ice for temperatures below {ice_thresh}."
        if kws["ice_thresh"] is not None
        else ""
    ),
    abstract="Calculation of relative humidity from temperature and dew point using the saturation vapour pressure.",
    compute=indices.relative_humidity,
    parameters={
        "tdps": {"kind": InputKind.VARIABLE},
        "huss": None,
        "ps": None,
        "invalid_values": "mask",
    },
)


relative_humidity = Converter(
    identifier="hurs",
    units="%",
    long_name="Relative humidity",
    standard_name="relative_humidity",
    title="Relative humidity from temperature, specific humidity, and pressure",
    description=lambda **kws: (
        "Computed from temperature, specific humidity and pressure through the saturation vapour pressure, "
        "which was calculated from temperature according to the {method} method."
    )
    + (
        " The computation was done in reference to ice for temperatures below {ice_thresh}."
        if kws["ice_thresh"] is not None
        else ""
    ),
    abstract="Calculation of relative humidity from temperature, "
    "specific humidity, and pressure using the saturation vapour pressure.",
    compute=indices.relative_humidity,
    parameters={
        "tdps": None,
        "huss": {"kind": InputKind.VARIABLE},
        "ps": {"kind": InputKind.VARIABLE},
        "invalid_values": "mask",
    },
)


specific_humidity = Converter(
    identifier="huss",
    units="",
    long_name="Specific humidity",
    standard_name="specific_humidity",
    title="Specific humidity from temperature, relative humidity, and pressure",
    description=lambda **kws: (
        "Computed from temperature, relative humidity and pressure through the saturation vapour pressure, "
        "which was calculated from temperature according to the {method} method."
    )
    + (
        " The computation was done in reference to ice for temperatures below {ice_thresh}."
        if kws["ice_thresh"] is not None
        else ""
    ),
    abstract="Calculation of specific humidity from temperature, "
    "relative humidity, and pressure using the saturation vapour pressure.",
    compute=indices.specific_humidity,
    parameters={"invalid_values": "mask"},
)

specific_humidity_from_dewpoint = Converter(
    identifier="huss_fromdewpoint",
    units="",
    long_name="Specific humidity",
    standard_name="specific_humidity",
    title="Specific humidity from dew point temperature and pressure",
    description=(
        "Computed from dewpoint temperature and pressure through the saturation "
        "vapor pressure, which was calculated according to the {method} method."
    ),
    abstract="Calculation of the specific humidity from dew point temperature "
    "and pressure using the saturation vapour pressure.",
    compute=indices.specific_humidity_from_dewpoint,
)

snowfall_approximation = Converter(
    identifier="prsn",
    units="kg m-2 s-1",
    standard_name="solid_precipitation_flux",
    long_name="Solid precipitation",
    title="Snowfall approximation",
    description=(
        "Solid precipitation estimated from total precipitation and temperature"
        " with method {method} and threshold temperature {thresh}."
    ),
    abstract="Solid precipitation estimated from total precipitation and temperature "
    "with a given method and temperature threshold.",
    compute=indices.snowfall_approximation,
)


rain_approximation = Converter(
    identifier="prlp",
    units="kg m-2 s-1",
    standard_name="precipitation_flux",
    long_name="Liquid precipitation",
    title="Rainfall approximation",
    description=(
        "Liquid precipitation estimated from total precipitation and temperature"
        " with method {method} and threshold temperature {thresh}."
    ),
    abstract="Liquid precipitation estimated from total precipitation and temperature "
    "with a given method and temperature threshold.",
    compute=indices.rain_approximation,
)


wind_chill_index = Converter(
    identifier="wind_chill",
    units="degC",
    long_name="Wind chill factor",
    title="Wind chill",
    description=lambda **kws: (
        "Wind chill index describing the temperature felt by the average person in response to cold wind."
    )
    + (
        "A slow-wind version of the wind chill index was used for wind speeds under 5 km/h and invalid "
        "temperatures were masked (T > 0°C)."
        if kws["method"] == "CAN"
        else "Invalid temperatures (T > 50°F) and winds (V < 3 mph) where masked."
    ),
    abstract="Wind chill factor is an index that equates to how cold an average person feels. "
    "It is calculated from the temperature and the wind speed at 10 m. "
    "As defined by Environment and Climate Change Canada, a second formula is used for light winds. "
    "The standard formula is otherwise the same as used in the United States.",
    compute=indices.wind_chill_index,
    parameters={"mask_invalid": True},
)


potential_evapotranspiration = Converter(
    identifier="potential_evapotranspiration",
    var_name="evspsblpot",
    units="kg m-2 s-1",
    standard_name="water_potential_evapotranspiration_flux",
    long_name="Potential evapotranspiration",
    title="Potential evapotranspiration",
    description=(
        "The potential for water evaporation from soil and transpiration by plants if the water "
        "supply is sufficient, calculated with the {method} method."
    ),
    abstract=(
        "The potential for water evaporation from soil and transpiration by plants if the water "
        "supply is sufficient, calculated with a given method."
    ),
    compute=indices.potential_evapotranspiration,
)

water_budget_from_tas = Converter(
    identifier="water_budget_from_tas",
    units="kg m-2 s-1",
    long_name="Water budget",
    title="Water budget",
    description=(
        "Precipitation minus potential evapotranspiration as a measure of an approximated surface water budget, "
        "where the potential evapotranspiration, calculated with the {method} method."
    ),
    abstract=(
        "Precipitation minus potential evapotranspiration as a measure of an approximated surface water budget, "
        "where the potential evapotranspiration, calculated with a given method."
    ),
    compute=indices.water_budget,
)

water_budget = Converter(
    identifier="water_budget",
    units="kg m-2 s-1",
    long_name="Water budget",
    title="Water budget",
    description=(
        "Precipitation minus potential evapotranspiration as a measure of an approximated surface water budget."
    ),
    abstract=(
        "Precipitation minus potential evapotranspiration as a measure of an approximated surface water budget."
    ),
    compute=indices.water_budget,
    parameters={"method": "dummy"},
)


corn_heat_units = Converter(
    identifier="corn_heat_units",
    units="",
    long_name="Corn heat units",
    title=" Corn heat units",
    description="Temperature-based index used to estimate the development of corn crops. "
    "Corn growth occurs when the minimum and maximum daily temperature both exceed "
    " {thresh_tasmin} and {thresh_tasmax}, respectively.",
    abstract="A temperature-based index used to estimate the development of corn crops. "
    "Corn growth occurs when the daily minimum and maximum temperatures exceed given thresholds.",
    var_name="chu",
    cell_methods="",
    missing="skip",
    compute=indices.corn_heat_units,
)

universal_thermal_climate_index = Converter(
    identifier="utci",
    units="K",
    long_name="Universal thermal climate index",
    title="Universal thermal climate index",
    description="UTCI is the equivalent temperature for the environment derived from a reference environment "
    "and is used to evaluate heat stress in outdoor spaces.",
    abstract="UTCI is the equivalent temperature for the environment derived from a reference environment "
    "and is used to evaluate heat stress in outdoor spaces.",
    cell_methods="",
    var_name="utci",
    compute=indices.universal_thermal_climate_index,
)

mean_radiant_temperature = Converter(
    identifier="mean_radiant_temperature",
    units="K",
    long_name="Mean radiant temperature",
    title="Mean radiant temperature",
    description="The incidence of radiation on the body from all directions.",
    abstract="The average temperature of solar and thermal radiation incident on the body's exterior.",
    cell_methods="",
    var_name="mrt",
    compute=indices.mean_radiant_temperature,
)

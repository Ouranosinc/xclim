"""Atmospheric conversion definitions."""
from __future__ import annotations

from inspect import _empty  # noqa

from xclim import indices
from xclim.core.cfchecks import cfcheck_from_name
from xclim.core.indicator import Indicator
from xclim.core.utils import InputKind

__all__ = [
    "corn_heat_units",
    "heat_index",
    "humidex",
    "longwave_upwelling_radiation_from_net_downwelling",
    "mean_radiant_temperature",
    "potential_evapotranspiration",
    "rain_approximation",
    "relative_humidity",
    "relative_humidity_from_dewpoint",
    "saturation_vapor_pressure",
    "shortwave_upwelling_radiation_from_net_downwelling",
    "snowfall_approximation",
    "specific_humidity",
    "specific_humidity_from_dewpoint",
    "tg",
    "universal_thermal_climate_index",
    "water_budget",
    "water_budget_from_tas",
    "wind_chill_index",
    "wind_speed_from_vector",
    "wind_vector_from_speed",
]


class Converter(Indicator):
    """Class for indicators doing variable conversion (dimension-independent 1-to-1 computation)."""

    def cfcheck(self, **das):
        for varname, vardata in das.items():
            try:
                # Only check standard_name, and not cell_methods which depends on the variable's frequency.
                cfcheck_from_name(varname, vardata, attrs=["standard_name"])
            except KeyError:
                # Silently ignore unknown variables.
                pass


humidex = Converter(
    title="Humidex",
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
    title="Heat index",
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
    title="Mean temperature",
    identifier="tg",
    units="K",
    standard_name="air_temperature",
    long_name="Daily mean temperature",
    description="Estimated mean temperature from maximum and minimum temperatures.",
    cell_methods="time: mean within days",
    abstract="The average daily temperature assuming a symmetrical temperature distribution (Tg = (Tx + Tn) / 2).",
    compute=indices.tas,
)


wind_speed_from_vector = Converter(
    title="Wind speed and direction from vector",
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
    title="Wind vector from speed and direction",
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
    title="Saturation vapour pressure (e_sat)",
    identifier="e_sat",
    units="Pa",
    long_name='Saturation vapour pressure ("{method}" method)',
    description=lambda **kws: (
        "The saturation vapour pressure was calculated from a temperature according to the {method} method."
    )
    + (
        " The computation was done in reference to ice for temperatures below {ice_thresh}."
        if kws["ice_thresh"] is not None
        else ""
    ),
    abstract="Calculation of the saturation vapour pressure from the temperature, according to a given method. "
    "If ice_thresh is given, the calculation is done with reference to ice for temperatures below this threshold.",
    compute=indices.saturation_vapor_pressure,
)


relative_humidity_from_dewpoint = Converter(
    title="Relative humidity from temperature and dewpoint temperature",
    identifier="hurs_fromdewpoint",
    units="%",
    var_name="hurs",
    long_name='Relative humidity ("{method}" method)',
    standard_name="relative_humidity",
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
    title="Relative humidity from temperature, specific humidity, and pressure",
    identifier="hurs",
    units="%",
    var_name="hurs",
    long_name='Relative Humidity ("{method}" method)',
    standard_name="relative_humidity",
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
    title="Specific humidity from temperature, relative humidity, and pressure",
    identifier="huss",
    units="",
    var_name="huss",
    long_name='Specific Humidity ("{method}" method)',
    standard_name="specific_humidity",
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
    title="Specific humidity from dew point temperature and pressure",
    identifier="huss_fromdewpoint",
    units="",
    long_name='Specific humidity ("{method}" method)',
    standard_name="specific_humidity",
    description=(
        "Computed from dewpoint temperature and pressure through the saturation "
        "vapor pressure, which was calculated according to the {method} method."
    ),
    abstract="Calculation of the specific humidity from dew point temperature "
    "and pressure using the saturation vapour pressure.",
    compute=indices.specific_humidity_from_dewpoint,
)

snowfall_approximation = Converter(
    title="Snowfall approximation",
    identifier="prsn",
    units="kg m-2 s-1",
    standard_name="solid_precipitation_flux",
    long_name='Solid precipitation ("{method}" method with temperature at or below {thresh})',
    description=(
        "Solid precipitation estimated from total precipitation and temperature"
        " with method {method} and threshold temperature {thresh}."
    ),
    abstract="Solid precipitation estimated from total precipitation and temperature "
    "with a given method and temperature threshold.",
    compute=indices.snowfall_approximation,
    context="hydro",
)


rain_approximation = Converter(
    title="Rainfall approximation",
    identifier="prlp",
    units="kg m-2 s-1",
    standard_name="precipitation_flux",
    long_name='Liquid precipitation ("{method}" method with temperature at or above {thresh})',
    description=(
        "Liquid precipitation estimated from total precipitation and temperature"
        " with method {method} and threshold temperature {thresh}."
    ),
    abstract="Liquid precipitation estimated from total precipitation and temperature "
    "with a given method and temperature threshold.",
    compute=indices.rain_approximation,
    context="hydro",
)


wind_chill_index = Converter(
    title="Wind chill",
    identifier="wind_chill",
    units="degC",
    long_name="Wind chill factor",
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
    title="Potential evapotranspiration",
    identifier="potential_evapotranspiration",
    var_name="evspsblpot",
    units="kg m-2 s-1",
    standard_name="water_potential_evapotranspiration_flux",
    long_name='Potential evapotranspiration ("{method}" method)',
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
    title="Water budget",
    identifier="water_budget_from_tas",
    units="kg m-2 s-1",
    long_name='Water budget ("{method}" method)',
    description=(
        "Precipitation minus potential evapotranspiration as a measure of an approximated surface water budget, "
        "where the potential evapotranspiration is calculated with the {method} method."
    ),
    abstract=(
        "Precipitation minus potential evapotranspiration as a measure of an approximated surface water budget, "
        "where the potential evapotranspiration is calculated with a given method."
    ),
    compute=indices.water_budget,
)

water_budget = Converter(
    title="Water budget",
    identifier="water_budget",
    units="kg m-2 s-1",
    long_name="Water budget",
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
    title=" Corn heat units",
    identifier="corn_heat_units",
    units="",
    long_name="Corn heat units (Tmin > {thresh_tasmin} and Tmax > {thresh_tasmax})",
    description="Temperature-based index used to estimate the development of corn crops. "
    "Corn growth occurs when the minimum and maximum daily temperatures both exceed "
    "{thresh_tasmin} and {thresh_tasmax}, respectively.",
    abstract="A temperature-based index used to estimate the development of corn crops. "
    "Corn growth occurs when the daily minimum and maximum temperatures exceed given thresholds.",
    var_name="chu",
    cell_methods="",
    missing="skip",
    compute=indices.corn_heat_units,
)

universal_thermal_climate_index = Converter(
    title="Universal Thermal Climate Index (UTCI)",
    identifier="utci",
    units="K",
    long_name="Universal Thermal Climate Index (UTCI)",
    description="UTCI is the equivalent temperature for the environment derived from a reference environment "
    "and is used to evaluate heat stress in outdoor spaces.",
    abstract="UTCI is the equivalent temperature for the environment derived from a reference environment "
    "and is used to evaluate heat stress in outdoor spaces.",
    cell_methods="",
    var_name="utci",
    compute=indices.universal_thermal_climate_index,
)

mean_radiant_temperature = Converter(
    title="Mean radiant temperature",
    identifier="mean_radiant_temperature",
    units="K",
    long_name="Mean radiant temperature",
    description="The incidence of radiation on the body from all directions.",
    abstract="The average temperature of solar and thermal radiation incident on the body's exterior.",
    cell_methods="",
    var_name="mrt",
    compute=indices.mean_radiant_temperature,
)


shortwave_upwelling_radiation_from_net_downwelling = Converter(
    title="Upwelling shortwave radiation",
    identifier="shortwave_upwelling_radiation_from_net_downwelling",
    units="W m-2",
    standard_name="surface_upwelling_shortwave_flux",
    long_name="Upwelling shortwave flux",
    description="The calculation of upwelling shortwave radiative flux from net surface shortwave "
    "and downwelling surface shortwave fluxes.",
    var_name="rsus",
    compute=indices.shortwave_upwelling_radiation_from_net_downwelling,
)

longwave_upwelling_radiation_from_net_downwelling = Converter(
    title="Upwelling longwave radiation",
    identifier="longwave_upwelling_radiation_from_net_downwelling",
    units="W m-2",
    standard_name="surface_upwelling_longwave_flux",
    long_name="Upwelling longwave flux",
    description="The calculation of upwelling longwave radiative flux from net surface longwave "
    "and downwelling surface longwave fluxes.",
    var_name="rlus",
    compute=indices.longwave_upwelling_radiation_from_net_downwelling,
)

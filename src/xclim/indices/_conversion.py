"""Conversion and approximation functions."""

import warnings

from xclim.indices.converters import *

warnings.warn(
    "The `xclim.indices._conversion` submodule is deprecated and will be removed in a future release. "
    "Functions available here have been migrated to `xclim.indices.converters`.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "clausius_clapeyron_scaled_precipitation",
    "clearness_index",
    "dewpoint_from_specific_humidity",
    "fao_allen98",
    "heat_index",
    "humidex",
    "longwave_upwelling_radiation_from_net_downwelling",
    "mean_radiant_temperature",
    "potential_evapotranspiration",
    "prsn_to_prsnd",
    "prsnd_to_prsn",
    "rain_approximation",
    "relative_humidity",
    "saturation_vapor_pressure",
    "sfcwind_to_uas_vas",
    "shortwave_downwelling_radiation_from_clearness_index",
    "shortwave_upwelling_radiation_from_net_downwelling",
    "snd_to_snw",
    "snowfall_approximation",
    "snw_to_snd",
    "specific_humidity",
    "specific_humidity_from_dewpoint",
    "tas",
    "uas_vas_to_sfcwind",
    "universal_thermal_climate_index",
    "vapor_pressure",
    "vapor_pressure_deficit",
    "wind_chill_index",
    "wind_power_potential",
    "wind_profile",
]

# noqa: D205,D400
"""
Sea ice indicators
------------------
"""
from xclim import indices
from xclim.core.indicator import Indicator

__all__ = ["sea_ice_area", "sea_ice_extent"]


class SiconcAreacello(Indicator):
    """Class for indicators having sea ice concentration and grid cell area inputs."""

    missing = "skip"


sea_ice_extent = SiconcAreacello(
    identifier="sea_ice_extent",
    units="m2",
    standard_name="sea_ice_extent",
    long_name="Sea ice extent",
    description="The sum of ocean areas where sea ice concentration is at least {thresh}.",
    cell_methods="lon: sum lat: sum",
    compute=indices.sea_ice_extent,
)


sea_ice_area = SiconcAreacello(
    identifier="sea_ice_area",
    units="m2",
    standard_name="sea_ice_area",
    long_name="Sea ice area",
    description="The sum of ice-covered areas where sea ice concentration is at least {thresh}.",
    cell_methods="lon: sum lat: sum",
    compute=indices.sea_ice_area,
)

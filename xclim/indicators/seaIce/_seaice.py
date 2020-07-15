# noqa: D205,D400
"""
Sea ice indicators
------------------
"""
import abc

from xclim import indices
from xclim.core import cfchecks
from xclim.core.indicator import Indicator2D

__all__ = ["sea_ice_area", "sea_ice_extent"]


class SicArea(Indicator2D):
    """Class for indicators having sea ice concentration and grid cell area inputs."""

    missing = "skip"

    @staticmethod
    def cfcheck(sic, area):
        cfchecks.check_valid(sic, "standard_name", "sea_ice_area_fraction")
        cfchecks.check_valid(area, "standard_name", "cell_area")


sea_ice_extent = SicArea(
    identifier="sea_ice_extent",
    units="m2",
    standard_name="sea_ice_extent",
    long_name="Sea ice extent",
    description="The sum of ocean areas where sea ice concentration is at least {thresh}.",
    cell_methods="lon: sum lat: sum",
    compute=indices.sea_ice_extent,
)


sea_ice_area = SicArea(
    identifier="sea_ice_area",
    units="m2",
    standard_name="sea_ice_area",
    long_name="Sea ice area",
    description="The sum of ice-covered areas where sea ice concentration is at least {thresh}.",
    cell_methods="lon: sum lat: sum",
    compute=indices.sea_ice_area,
)

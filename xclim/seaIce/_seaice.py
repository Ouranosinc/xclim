"""
Sea ice indicators
------------------
"""
import abc

from xclim import checks
from xclim import indices
from xclim.utils import Indicator2D

__all__ = ["sea_ice_area", "sea_ice_extent"]


class SicArea(Indicator2D):
    """Class for indicators having sea ice concentration and grid cell area inputs."""

    def cfprobe(self, sic, area):
        checks.check_valid(sic, "standard_name", "sea_ice_area_fraction")

    @abc.abstractmethod
    def compute(*args, **kwds):
        """The function computing the indicator."""

    def validate(self, da):
        """Input validation."""

    def missing(self, *args, **kwds):
        """Return whether an output is considered missing or not."""
        return False


sea_ice_extent = SicArea(
    identifier="sea_ice_extent",
    units="m^2",
    standard_name="sea_ice_extent",
    long_name="Sea ice extent",
    description="The sum of ocean areas where sea ice concentration is at least {thresh}.",
    cell_methods="lon: sum lat: sum",
    compute=indices.sea_ice_extent,
)


sea_ice_area = SicArea(
    identifier="sea_ice_area",
    units="m^2",
    standard_name="sea_ice_area",
    long_name="Sea ice area",
    description="The sum of ice-covered areas where sea ice concentration is at least {thresh}.",
    cell_methods="lon: sum lat: sum",
    compute=indices.sea_ice_area,
)

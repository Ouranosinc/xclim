"""
Sea ice indicators
------------------
"""
from __future__ import annotations

from xclim import indices
from xclim.core.indicator import Indicator

__all__ = ["sea_ice_area", "sea_ice_extent"]


class SiconcAreacello(Indicator):
    """Class for indicators having sea ice concentration and grid cell area inputs."""

    missing = "skip"


sea_ice_extent = SiconcAreacello(
    title="Sea ice extent",
    identifier="sea_ice_extent",
    units="m2",
    standard_name="sea_ice_extent",
    long_name="Sum of ocean areas where sea ice concentration exceeds {thresh}",
    description="The sum of ocean areas where sea ice concentration exceeds {thresh}.",
    abstract="A measure of the extent of all areas where sea ice concentration exceeds a threshold.",
    cell_methods="lon: sum lat: sum",
    compute=indices.sea_ice_extent,
)


sea_ice_area = SiconcAreacello(
    title="Sea ice area",
    identifier="sea_ice_area",
    units="m2",
    standard_name="sea_ice_area",
    long_name="Sum of ice-covered areas where sea ice concentration exceeds {thresh}",
    description="The sum of ice-covered areas where sea ice concentration exceeds {thresh}.",
    abstract="A measure of total ocean surface covered by sea ice.",
    cell_methods="lon: sum lat: sum",
    compute=indices.sea_ice_area,
)

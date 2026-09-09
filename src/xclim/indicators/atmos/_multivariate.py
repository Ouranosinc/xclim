"""Complex multivariate indicator definitions."""

from __future__ import annotations

from xclim import compute
from xclim.core.indicator import Indicator

__all__ = ["canadian_hardiness_zones"]


class MultivariateIndicator(Indicator):
    """Indicators involving daily temperature, precipitation, and other variables."""


canadian_hardiness_zones = MultivariateIndicator(
    keywords="temperature precipitation speed",
    title="Canadian hardiness zones",
    identifier="canadian_hardiness_zones",
    standard_name="",
    units="",
    long_name="Canadian hardiness zones",
    description="A climate index based on a {freq} climatology of the annual average of maximum and minimum "
    "monthly temperatures, seasonal precipitation, average annual maximum snow depth, and maximum wind gust "
    "experienced over the entire period. Developed specifically to aid in determining plant suitability of Canadian "
    "geographic regions.",
    abstract="A climate index based on a multi-year climatology of the annual average of maximum and minimum "
    "monthly temperatures, seasonal precipitation, average annual maximum snow depth, and maximum wind gust "
    "experienced over the entire period. Developed specifically to aid in determining plant suitability of Canadian "
    "geographic regions.",
    cell_methods="",
    var_name="chz",
    compute=compute.canadian_hardiness_zones,
    parameters={
        "freq": {"default": "30YS"},
    },
)

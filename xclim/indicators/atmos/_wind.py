from __future__ import annotations

from xclim import indices
from xclim.core.indicator import ResamplingIndicatorWithIndexing

__all__ = ["calm_days", "windy_days"]


class Wind(ResamplingIndicatorWithIndexing):
    """Indicator involving daily sfcWind series."""

    src_freq = "D"


calm_days = Wind(
    title="Calm days",
    identifier="calm_days",
    units="days",
    standard_name="number_of_days_with_sfcWind_below_threshold",
    long_name="Number of days with surface wind speed below {thresh}",
    description="{freq} number of days with surface wind speed below {thresh}.",
    abstract="Number of days with surface wind speed below threshold.",
    cell_methods="time: sum over days",
    compute=indices.calm_days,
)

windy_days = Wind(
    title="Windy days",
    identifier="windy_days",
    units="days",
    standard_name="number_of_days_with_sfcWind_above_threshold",
    long_name="Number of days with surface wind speed at or above {thresh}",
    description="{freq} number of days with surface wind speed at or above {thresh}.",
    abstract="Number of days with surface wind speed at or above threshold.",
    cell_methods="time: sum over days",
    compute=indices.windy_days,
)

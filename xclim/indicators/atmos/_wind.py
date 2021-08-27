from xclim import indices
from xclim.core.indicator import Daily

__all__ = ["calm_days", "windy_days"]


class Wind(Daily):
    """Indicator involving daily sfcWind series."""


calm_days = Wind(
    identifier="calm_days",
    units="days",
    standard_name="number_of_days_with_sfcWind_below_threshold",
    long_name="Number of days with surface wind speed below threshold",
    description="{freq} number of days with surface wind speed < {thresh}",
    cell_methods="time: mean within days time: sum over days",
    compute=indices.calm_days,
)

windy_days = Wind(
    identifier="windy_days",
    units="days",
    standard_name="number_of_days_with_sfcWind_above_threshold",
    long_name="Number of days with surface wind speed above threshold",
    description="{freq} number of days with surface wind speed >= {thresh}",
    cell_methods="time: mean within days time: sum over days",
    compute=indices.windy_days,
)

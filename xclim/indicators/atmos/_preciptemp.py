# -*- coding: utf-8 -*-
"""Precipitation and temperature indicator definitions."""

from xclim import indices
from xclim.core.indicator import Daily

__all__ = [
    "cold_and_dry_days",
    "warm_and_dry_days",
    "warm_and_wet_days",
    "cold_and_wet_days",
]


class PrecipTemp(Daily):
    """Indicators involving temperature."""


cold_and_dry_days = PrecipTemp(
    identifier="cold_and_dry_days",
    units="days",
    long_name="Cold and dry days",
    title="Cold and dry days",
    description="{freq} number of days where tas < 25th percentile and pr < 25th percentile",
    cell_methods="time: mean within days time: sum over days",
    compute=indices.cold_and_dry_days,
)

warm_and_dry_days = PrecipTemp(
    identifier="warm_and_dry_days",
    units="days",
    long_name="warm and dry days",
    title="warm and dry days",
    description="{freq} number of days where tas > 75th percentile and pr < 25th percentile",
    cell_methods="time: mean within days time: sum over days",
    compute=indices.warm_and_dry_days,
)

warm_and_wet_days = PrecipTemp(
    identifier="warm_and_wet_days",
    units="days",
    long_name="warm and wet days",
    title="warm and wet days",
    description="{freq} number of days where tas > 75th percentile and pr > 75th percentile",
    cell_methods="time: mean within days time: sum over days",
    compute=indices.warm_and_wet_days,
)

cold_and_wet_days = PrecipTemp(
    identifier="cold_and_wet_days",
    units="days",
    long_name="cold and wet days",
    title="cold and wet days",
    description="{freq} number of days where tas < 25th percentile and pr > 75th percentile",
    cell_methods="time: mean within days time: sum over days",
    compute=indices.cold_and_wet_days,
)

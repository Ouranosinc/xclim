# -*- coding: utf-8 -*-
"""Synoptic indicator definitions."""

from xclim import indices
from xclim.core.indicator import Daily

__all__ = ["jetstream_metric_woolings"]


class JetStream(Daily):
    """Indicator involving daily u- and/or v-component wind series."""


jetstream_metric_woolings = JetStream(
    identifier="jetstream_metric_woolings",
    units="days",
    standard_name="daily_latitude_and_strength_of_jetstream",
    long_name="Daily latitude and strength of maximum smoothed zonal wind speed",
    description="{freq} daily latitude and strength of maximum smoothed zonal wind speed",
    cell_methods="",
    compute=indices.jetstream_metric_woolings,
)

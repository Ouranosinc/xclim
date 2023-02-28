"""Synoptic indicator definitions."""
from __future__ import annotations

from xclim import indices
from xclim.core.indicator import Indicator

__all__ = ["jetstream_metric_woollings"]


class JetStream(Indicator):
    """Indicator involving daily u- and/or v-component wind series."""

    src_freq = "D"


jetstream_metric_woollings = JetStream(
    title="Strength and latitude of jetstream",
    identifier="jetstream_metric_woollings",
    var_name=["jetlat", "jetstr"],
    units=["degrees_North", "m s-1"],
    long_name=[
        "Latitude of maximum smoothed zonal wind speed",
        "Maximum strength of smoothed zonal wind speed",
    ],
    description=[
        "Daily latitude of maximum Lanczos smoothed zonal wind speed.",
        "Daily maximum strength of Lanczos smoothed zonal wind speed.",
    ],
    compute=indices.jetstream_metric_woollings,
)

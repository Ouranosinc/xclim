"""Streamflow indicator definitions."""
from __future__ import annotations

from xclim.core.cfchecks import check_valid
from xclim.core.indicator import Indicator, ResamplingIndicator
from xclim.core.units import declare_units
from xclim.indices import base_flow_index, generic, rb_flashiness_index

__all__ = [
    "base_flow_index",
    "doy_qmax",
    "doy_qmin",
    "rb_flashiness_index",
]


class Streamflow(ResamplingIndicator):
    context = "hydro"
    src_freq = "D"

    @staticmethod
    def cfcheck(q):
        check_valid(q, "standard_name", "water_volume_transport_in_river_channel")


base_flow_index = Streamflow(
    title="Base flow index",
    identifier="base_flow_index",
    units="",
    long_name="Base flow index",
    description="Minimum of the 7-day moving average flow divided by the mean flow.",
    asbtract="Minimum of the 7-day moving average flow divided by the mean flow.",
    compute=base_flow_index,
)


rb_flashiness_index = Streamflow(
    title="Richards-Baker Flashiness Index",
    identifier="rb_flashiness_index",
    units="",
    var_name="rbi",
    long_name="Richards-Baker Flashiness Index",
    description="{freq} of Richards-Baker Index, an index measuring the flashiness of flow.",
    abstract="Measurement of flow oscillations relative to average flow, "
    "quantifying the frequency and speed of flow changes.",
    compute=rb_flashiness_index,
)


doy_qmax = Streamflow(
    title="Day of year of the maximum streamflow",
    identifier="doy_qmax",
    var_name="q{indexer}_doy_qmax",
    long_name="Day of the year of the maximum streamflow over {indexer}",
    description="Day of the year of the maximum streamflow over {indexer}.",
    units="",
    compute=declare_units(da="[discharge]")(generic.select_resample_op),
    parameters=dict(op=generic.doymax),
)


doy_qmin = Streamflow(
    title="Day of year of the minimum streamflow",
    identifier="doy_qmin",
    var_name="q{indexer}_doy_qmin",
    long_name="Day of the year of the minimum streamflow over {indexer}",
    description="Day of the year of the minimum streamflow over {indexer}.",
    units="",
    compute=declare_units(da="[discharge]")(generic.select_resample_op),
    parameters=dict(op=generic.doymin),
)

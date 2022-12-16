"""Streamflow indicator definitions."""
from __future__ import annotations

from abc import ABC

from xclim.core.cfchecks import check_valid
from xclim.core.indicator import Indicator, ResamplingIndicator
from xclim.core.units import declare_units
from xclim.indices import base_flow_index, generic, rb_flashiness_index
from xclim.indices.stats import fit as _fit
from xclim.indices.stats import frequency_analysis

__all__ = [
    "base_flow_index",
    "rb_flashiness_index",
    "freq_analysis",
    "stats",
    "fit",
    "doy_qmax",
    "doy_qmin",
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

freq_analysis = Streamflow(
    title="Return level",
    identifier="freq_analysis",
    var_name="q{window}{mode:r}{indexer}",
    long_name="N-year return level discharge",
    description="Streamflow frequency analysis for the {mode} {indexer} {window}-day flow estimated using the {dist} "
    "distribution.",
    abstract="Streamflow frequency analysis on the basis of a given mode and distribution.",
    units="m^3 s-1",
    compute=frequency_analysis,
    missing="skip",
    input={"da": "discharge"},
    _version_deprecated="0.40",
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


stats = Streamflow(
    title="Statistic of the daily flow for a given period.",
    identifier="discharge_stats",
    var_name="q{indexer}{op:r}",
    long_name="Daily flow statistics",
    description="{freq} {op} of daily flow ({indexer}).",
    units="m^3 s-1",
    compute=generic.select_resample_op,
    missing="any",
    input={"da": "discharge"},
    _version_deprecated="0.40",
)


fit = Indicator(
    title="Distribution parameters fitted over the time dimension.",
    identifier="discharge_distribution_fit",
    var_name="params",
    units="",
    standard_name="{dist} parameters",
    long_name="{dist} distribution parameters",
    description="Parameters of the {dist} distribution.",
    cell_methods="time: fit",
    src_freq=None,
    compute=_fit,
    input={"da": "discharge"},
    realm="land",
    _version_deprecated="0.40",
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

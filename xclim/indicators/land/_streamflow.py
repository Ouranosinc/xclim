"""Streamflow indicator definitions."""

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


class Stats(Streamflow):
    missing = "any"


class FA(Streamflow):
    """Frequency analysis.

    Notes
    -----
    FA performs three steps:
     1. Compute stats over time series (min, max)
     2. Fit statistical distribution parameters
     3. Compute parametric quantiles for given return periods

    Missing value functionality cannot be meaningfully applied here, because indicators apply missing value
    operations on input and apply the mask on output. The `freq` of the input could be "YS", but this same
    `freq` would then be used to compute the mask, which makes no sense.
    """

    missing = "skip"


# Disable the daily checks because the inputs are period extremas.
class Fit(Indicator):
    src_freq = None

    def cfcheck(self, **das):
        pass


base_flow_index = Streamflow(
    identifier="base_flow_index",
    units="",
    long_name="Base flow index",
    description="Minimum 7-day average flow divided by the mean flow.",
    compute=base_flow_index,
)

freq_analysis = FA(
    identifier="freq_analysis",
    var_name="q{window}{mode:r}{indexer}",
    long_name="N-year return period {mode} {indexer} {window}-day flow",
    description="Streamflow frequency analysis for the {mode} {indexer} {window}-day flow "
    "estimated using the {dist} distribution.",
    units="m^3 s-1",
    title="Flow values for given return periods.",
    compute=declare_units(da=None)(frequency_analysis),
)

rb_flashiness_index = Streamflow(
    identifier="rb_flashiness_index",
    units="",
    var_name="rbi",
    long_name="Richards-Baker flashiness index",
    description="{freq} R-B Index, an index measuring the flashiness of flow.",
    compute=rb_flashiness_index,
)

stats = Stats(
    identifier="stats",
    var_name="q{indexer}{op:r}",
    long_name="{freq} {op} of {indexer} daily flow ",
    description="{freq} {op} of {indexer} daily flow",
    title="Statistic of the daily flow on a given period.",
    units="m^3 s-1",
    compute=declare_units(da=None)(generic.select_resample_op),
)

fit = Fit(
    identifier="fit",
    var_name="params",
    units="",
    standard_name="{dist} parameters",
    long_name="{dist} distribution parameters",
    description="Parameters of the {dist} distribution",
    title="Distribution parameters fitted over the time dimension.",
    cell_methods="time: fit",
    compute=declare_units(da=None)(_fit),
)


doy_qmax = Streamflow(
    identifier="doy_qmax",
    var_name="q{indexer}_doy_qmax",
    long_name="Day of the year of the maximum over {indexer}",
    description="Day of the year of the maximum over {indexer}",
    title="Day of year of the maximum.",
    units="",
    compute=declare_units(da=None)(generic.select_resample_op),
    parameters=dict(op=generic.doymax),
)


doy_qmin = Streamflow(
    identifier="doy_qmin",
    var_name="q{indexer}_doy_qmin",
    long_name="Day of the year of the minimum over {indexer}",
    description="Day of the year of the minimum over {indexer}",
    title="Day of year of the minimum.",
    units="",
    compute=declare_units(da=None)(generic.select_resample_op),
    parameters=dict(op=generic.doymin),
)

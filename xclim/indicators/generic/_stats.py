from __future__ import annotations

from xclim.core.indicator import Indicator, ResamplingIndicator
from xclim.indices.generic import select_resample_op
from xclim.indices.stats import fit as _fit
from xclim.indices.stats import frequency_analysis

__all__ = ["fit", "return_level", "stats"]


fit = Indicator(
    title="Distribution parameters fitted over the time dimension.",
    identifier="fit",
    var_name="params",
    units="",
    standard_name="{dist} parameters",
    long_name="{dist} distribution parameters",
    description="Parameters of the {dist} distribution.",
    cell_methods="time: fit",
    compute=_fit,
    realm="generic",
    src_freq=None,
)


return_level = ResamplingIndicator(
    title="Return level from frequency analysis",
    identifier="return_level",
    var_name="fa_{window}{mode:r}{indexer}",
    long_name="N-year return level",
    description="Frequency analysis for the {mode} {indexer} {window}-day value estimated using the {dist} "
    "distribution.",
    abstract="Frequency analysis on the basis of a given mode and distribution.",
    compute=frequency_analysis,
    realm="generic",
    src_freq="D",
    missing="skip",
)


stats = ResamplingIndicator(
    title="Statistic of the daily values for a given period.",
    identifier="stats",
    var_name="stat_{indexer}{op:r}",
    long_name="Daily statistics",
    description="{freq} {op} of daily values ({indexer}).",
    compute=select_resample_op,
    missing="any",
    src_freq="D",
    realm="generic",
)

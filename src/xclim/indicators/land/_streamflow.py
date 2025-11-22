"""Streamflow indicator definitions."""

from __future__ import annotations

from xarray import DataArray

from xclim.core.cfchecks import check_valid
from xclim.core.indicator import (
    ReducingIndicator,
    ResamplingIndicator,
    StandardizedIndexes,
)
from xclim.core.units import declare_units
from xclim.indices import (
    base_flow_index,
    base_flow_index_seasonal_ratio,
    flow_index,
    generic,
    high_flow_frequency,
    lag_snowpack_flow_peaks,
    low_flow_frequency,
    rb_flashiness_index,
    runoff_ratio,
    sen_slope,
    standardized_groundwater_index,
    standardized_streamflow_index,
)

__all__ = [
    "base_flow_index",
    "base_flow_index_seasonal_ratio",
    "doy_qmax",
    "doy_qmin",
    "flow_index",
    "high_flow_frequency",
    "lag_snowpack_flow_peaks",
    "low_flow_frequency",
    "rb_flashiness_index",
    "runoff_ratio",
    "sen_slope",
    "standardized_groundwater_index",
    "standardized_streamflow_index",
]


class Streamflow(ResamplingIndicator):
    """Streamflow class."""

    context = "hydro"
    src_freq = "D"
    keywords = "streamflow hydrology"

    @staticmethod
    def cfcheck(q: DataArray):
        r"""
        Verify the CF-compliance of the input data.

        Parameters
        ----------
        q : xarray.DataArray
            The input data array.
        """
        check_valid(q, "standard_name", "water_volume_transport_in_river_channel")


base_flow_index = Streamflow(
    title="Base flow index",
    identifier="base_flow_index",
    units="",
    long_name="Base flow index",
    description="Minimum of the 7-day moving average flow divided by the mean flow.",
    abstract="Minimum of the 7-day moving average flow divided by the mean flow.",
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
    parameters={"op": generic.doymax, "out_units": None},
)


doy_qmin = Streamflow(
    title="Day of year of the minimum streamflow",
    identifier="doy_qmin",
    var_name="q{indexer}_doy_qmin",
    long_name="Day of the year of the minimum streamflow over {indexer}",
    description="Day of the year of the minimum streamflow over {indexer}.",
    units="",
    compute=declare_units(da="[discharge]")(generic.select_resample_op),
    parameters={"op": generic.doymin, "out_units": None},
)


flow_index = ReducingIndicator(
    realm="land",
    context="hydro",
    title="Flow index",
    identifier="flow_index",
    var_name="q_flow_index",
    long_name="Flow index",
    description="{p}th percentile normalized by the median flow.",
    units="1",
    compute=flow_index,
)


high_flow_frequency = Streamflow(
    title="High flow frequency",
    identifier="high_flow_frequency",
    var_name="q_high_flow_frequency",
    long_name="High flow frequency",
    description="{freq} frequency of flows greater than {threshold_factor} times the median flow.",
    units="days",
    compute=high_flow_frequency,
)


low_flow_frequency = Streamflow(
    title="Low flow frequency",
    identifier="low_flow_frequency",
    var_name="q_low_flow_frequency",
    long_name="Low flow frequency",
    description="{freq} frequency of flows smaller than a fraction ({threshold_factor}) of the mean flow.",
    units="days",
    compute=low_flow_frequency,
)

standardized_streamflow_index = StandardizedIndexes(
    realm="land",
    title="Standardized Streamflow Index (SSI)",
    identifier="ssi",
    units="",
    standard_name="ssi",
    long_name="Standardized Streamflow Index (SSI)",
    description="Streamflow over a moving {window}-X window, normalized such that SSI averages to 0 for "
    "calibration data. The window unit `X` is the minimal time period defined by resampling frequency {freq}.",
    abstract="Streamflow over a moving window, normalized such that SSI averages to 0 for the calibration data. "
    "The window unit `X` is the minimal time period defined by the resampling frequency.",
    cell_methods="",
    keywords="streamflow",
    compute=standardized_streamflow_index,
)

standardized_groundwater_index = StandardizedIndexes(
    realm="land",
    title="Standardized Groundwater Index (SGI)",
    identifier="sgi",
    units="",
    standard_name="sgi",
    long_name="Standardized Groundwater Index (SGI)",
    description="Groundwater over a moving {window}-X window, normalized such that SGI averages to 0 for "
    "calibration data. The window unit `X` is the minimal time period defined by resampling frequency {freq}.",
    abstract="Groundwater over a moving window, normalized such that SGI averages to 0 for the calibration data. "
    "The window unit `X` is the minimal time period defined by the resampling frequency.",
    cell_methods="",
    keywords="groundwater",
    compute=standardized_groundwater_index,
)

base_flow_index_seasonal_ratio = Streamflow(
    title="Seasonal Base flow index (bfi) and ratio of winter to summer base flow index",
    identifier="base_flow_index_seasonal_ratio",
    units="",
    long_name="Seasonal Base flow index and ratio of winter to summer base flow index",
    description="Yearly base flow index per season, defined as the minimum 7-day average flow divided by the mean flow"
    "as well as yearly winter to summer bfi ratio.",
    abstract="Yearly base flow index per season, defined as the minimum 7-day average flow divided by the mean flow"
    "as well as yearly winter to summer bfi ratio.",
    cell_methods="",
    keywords="streamflow, seasonal",
    compute=base_flow_index_seasonal_ratio,
)


lag_snowpack_flow_peaks = Streamflow(
    title="Time lag between maximum snowpack and river high flows",
    identifier="lag_snowpack_flow_peaks",
    units="days",
    long_name="Time lag between maximum snowpack and river high flows",
    description="Number of days between the annual maximum snowpack, measured by the snow water"
    "equivalent, and the mean date when river flow exceeds a percentile threshold"
    "during a given year.",
    cell_methods="",
    keywords="streamflow, SWE",
    compute=lag_snowpack_flow_peaks,
)

runoff_ratio = Streamflow(
    title="Runoff ratio",
    identifier="runoff_ratio",
    units="",
    long_name="Runoff ratio",
    description="Ratio of runoff volume measured at the stream to the total precipitation volume "
    "over the watershed. Temporal analysis: Yearly values computed from seasonal daily data "
    "and yearly data, depending on chosen frequency.",
    cell_methods="",
    keywords="streamflow",
    compute=runoff_ratio,
)

sen_slope = Streamflow(
    title="Sen Slope : Temporal robustness analysis of streamflow.",
    identifier="sen_slope",
    units="",
    long_name="Sen Slope",
    description="Compute annual and seasonal Theil–Sen slope estimators and perform the"
    "Mann–Kendall test for trend evaluation.",
    cell_methods="",
    keywords="streamflow",
    compute=sen_slope,
)

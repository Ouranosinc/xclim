from __future__ import annotations
import xarray as xr
import numpy as np
from xclim.core.units import declare_units
from xclim.indices.generic import compare, threshold_count

__all__ = [
    "flow_index",
    "high_flow_frequency",
    "low_flow_frequency",
]

@declare_units(q="[discharge]")
def flow_index(q: xr.DataArray, p: float = 0.95) -> xr.DataArray:
    """
    Calculate the Qp  (pth percentile of daily streamflow) normalized by the mean flow.

    Parameters
    ----------
    q : xarray.DataArray
        Daily streamflow data.
    p : float
        Percentile for calculating the flow index, between 0 and 1. Default of 0.95 is for high flows.

    Returns
    -------
    xarray.DataArray
    Normalized Qp, which is the p th percentile of daily streamflow normalized by the median flow.

    Reference:
    1. Addor, Nans & Nearing, Grey & Prieto, Cristina & Newman, A. & Le Vine, Nataliya & Clark, Martyn. (2018). A Ranking of Hydrological Signatures Based on Their Predictability in Space. Water Resources Research. 10.1029/2018WR022606.
    2. Clausen, B., & Biggs, B. J. F. (2000). Flow variables for ecological studies in temperate streams: Groupings based on covariance. Journal of Hydrology, 237(3–4), 184–197. https://doi.org/10.1016/S0022-1694(00)00306-1

    """
    qp = q.quantile(p, dim="time")
    q_median = q.median(dim="time")
    out = qp / q_median
    out.attrs["units"] = " "
    return out


@declare_units(q="[discharge]")
def high_flow_frequency(
    q: xr.DataArray,
    threshold_factor: int = 9,
    freq: str = "A-SEP",
) -> xr.DataArray:
    """
    Calculate the mean number of days in a given period with flows greater than a specified threshold. By default, the period is the water year starting on 1st October and ending on 30th September, as commonly defined in North America.

    Parameters
    ----------
    q : xarray.DataArray
        Daily streamflow data.
    threshold_factor : int
        Factor by which the median flow is multiplied to set the high flow threshold, default is 9.
    freq : str, optional
        Resampling frequency, default is 'A-SEP' for water year ending in September.
    op : {">", "<", "gt", "lt"}, optional
        Comparison operation. Default: "<".

    Returns
    -------
    xarray.DataArray
    Calculated mean of high flow days per water year

    References
    ----------
    1. Addor, Nans & Nearing, Grey & Prieto, Cristina & Newman, A. & Le Vine, Nataliya & Clark, Martyn. (2018). A Ranking of Hydrological Signatures Based on Their Predictability in Space. Water Resources Research. 10.1029/2018WR022606.
    2. Clausen, B., & Biggs, B. J. F. (2000). Flow variables for ecological studies in temperate streams: Groupings based on covariance. Journal of Hydrology, 237(3–4), 184–197. https://doi.org/10.1016/S0022-1694(00)00306-1
    """

    median_flow = q.median(dim="time")
    with xr.set_options(keep_attrs=True):
        threshold = threshold_factor * median_flow
    high_flow_days = compare(q, op=">", right=threshold).resample(time=freq).sum(dim="time")
    out = high_flow_days.mean(dim="time")
    out.attrs["units"] = "days/year"
    return out


@declare_units(q="[discharge]")
def low_flow_frequency(
    q: xr.DataArray,
    threshold_factor: float = 0.2,
    freq: str = "A-SEP",
) -> xr.DataArray:
    """
    Calculate the mean number of days in a given period with flows lower than a specified threshold. By default, the period is the water year starting on 1st October and ending on 30th September, as commonly defined in North America.

    Parameters
    ----------
    q : xarray.DataArray
        Daily streamflow data.
    threshold_factor : float
        Factor by which the mean flow is multiplied to set the low flow threshold, default is 0.2.
    freq : str, optional
        Resampling frequency, default is 'A-SEP' for water year ending in September.
    op : {">", "<", "gt", "lt"}, optional
        Comparison operation. Default: "<".

    Returns
    -------
    xarray.DataArray
    Calculated mean of low flow days per water year

    References
    ----------
    Olden, J. D., & Poff, N. L. (2003). Redundancy and the choice of hydrologic indices for characterizing streamflow regimes. River Research and
    Applications, 19(2), 101–121. https://doi.org/10.1002/rra.700
    """

    mean_flow = q.mean(dim="time")
    with xr.set_options(keep_attrs=True):
        threshold = threshold_factor * mean_flow
    low_flow_days = compare(q, op="<", right=threshold).resample(time=freq).sum(dim="time")
    out = low_flow_days.mean(dim="time")
    out.attrs["units"] = "days"
    return out


# noqa: D100
from __future__ import annotations

import numpy as np
import xarray as xr

from xclim.core.calendar import get_calendar
from xclim.core.missing import at_least_n_valid
from xclim.core.units import declare_units, rate2amount
from xclim.indices.generic import compare

from . import generic

__all__ = [
    "base_flow_index",
    "flow_index",
    "high_flow_frequency",
    "low_flow_frequency",
    "melt_and_precip_max",
    "rb_flashiness_index",
    "snd_max",
    "snd_max_doy",
    "snow_melt_we_max",
    "snw_max",
    "snw_max_doy",
]


@declare_units(q="[discharge]")
def base_flow_index(q: xarray.DataArray, freq: str = "YS") -> xarray.DataArray:
    r"""Base flow index.

    Return the base flow index, defined as the minimum 7-day average flow divided by the mean flow.

    Parameters
    ----------
    q : xarray.DataArray
        Rate of river discharge.
    freq : str
        Resampling frequency.

    Returns
    -------
    xarray.DataArray, [dimensionless]
        Base flow index.

    Notes
    -----
    Let :math:`\mathbf{q}=q_0, q_1, \ldots, q_n` be the sequence of daily discharge and :math:`\overline{\mathbf{q}}`
    the mean flow over the period. The base flow index is given by:

    .. math::

       \frac{\min(\mathrm{CMA}_7(\mathbf{q}))}{\overline{\mathbf{q}}}


    where :math:`\mathrm{CMA}_7` is the seven days moving average of the daily flow:

    .. math::

       \mathrm{CMA}_7(q_i) = \frac{\sum_{j=i-3}^{i+3} q_j}{7}

    """
    m7 = q.rolling(time=7, center=True).mean(skipna=False).resample(time=freq)
    mq = q.resample(time=freq)

    m7m = m7.min(dim="time")
    out = m7m / mq.mean(dim="time")
    out.attrs["units"] = ""
    return out


@declare_units(q="[discharge]")
def rb_flashiness_index(q: xarray.DataArray, freq: str = "YS") -> xarray.DataArray:
    r"""Richards-Baker flashiness index.

    Measures oscillations in flow relative to total flow, quantifying the frequency and rapidity of short term changes
    in flow, based on :cite:t:`baker_new_2004`.

    Parameters
    ----------
    q : xarray.DataArray
        Rate of river discharge.
    freq : str
        Resampling frequency.

    Returns
    -------
    xarray.DataArray, [dimensionless]
        R-B Index.

    Notes
    -----
    Let :math:`\mathbf{q}=q_0, q_1, \ldots, q_n` be the sequence of daily discharge, the R-B Index is given by:

    .. math::

       \frac{\sum_{i=1}^n |q_i - q_{i-1}|}{\sum_{i=1}^n q_i}

    References
    ----------
    :cite:cts:`baker_new_2004`
    """
    d = np.abs(q.diff(dim="time")).resample(time=freq)
    mq = q.resample(time=freq)
    out = d.sum(dim="time") / mq.sum(dim="time")
    out.attrs["units"] = ""
    return out


@declare_units(snd="[length]")
def snd_max(snd: xarray.DataArray, freq: str = "YS-JUL") -> xarray.DataArray:
    """Maximum snow depth.

    The maximum daily snow depth.

    Parameters
    ----------
    snw : xarray.DataArray
        Snow depth (mass per area).
    freq : str
        Resampling frequency.

    Returns
    -------
    xarray.DataArray
        The maximum snow depth over a given number of days for each period. [length].
    """
    return generic.select_resample_op(snd, op="max", freq=freq)


@declare_units(snd="[length]")
def snd_max_doy(snd: xarray.DataArray, freq: str = "YS-JUL") -> xarray.DataArray:
    """Maximum snow depth day of year.

    Day of year when surface snow reaches its peak value. If snow depth is 0 over entire period, return NaN.

    Parameters
    ----------
    snd : xarray.DataArray
        Surface snow depth.
    freq : str
         Resampling frequency.

    Returns
    -------
    xarray.DataArray
        The day of year at which snow depth reaches its maximum value.
    """
    # Identify periods where there is at least one non-null value for snow depth
    valid = at_least_n_valid(snd.where(snd > 0), n=1, freq=freq)

    # Compute doymax. Will return first time step if all snow depths are 0.
    out = generic.select_resample_op(
        snd.where(snd > 0, 0), op=generic.doymax, freq=freq
    )
    out.attrs.update(units="", is_dayofyear=np.int32(1), calendar=get_calendar(snd))

    # Mask arrays that miss at least one non-null snd.
    return out.where(~valid)


@declare_units(snw="[mass]/[area]")
def snw_max(snw: xarray.DataArray, freq: str = "YS-JUL") -> xarray.DataArray:
    """Maximum snow amount.

    The maximum daily snow amount.

    Parameters
    ----------
    snw : xarray.DataArray
        Snow amount (mass per area).
    freq: str
        Resampling frequency.

    Returns
    -------
    xarray.DataArray
        The maximum snow amount over a given number of days for each period. [mass/area].
    """
    return generic.select_resample_op(snw, op="max", freq=freq)


@declare_units(snw="[mass]/[area]")
def snw_max_doy(snw: xarray.DataArray, freq: str = "YS-JUL") -> xarray.DataArray:
    """Maximum snow amount day of year.

    Day of year when surface snow amount reaches its peak value. If snow amount is 0 over entire period, return NaN.

    Parameters
    ----------
    snw : xarray.DataArray
        Surface snow amount.
    freq : str
        Resampling frequency.

    Returns
    -------
    xarray.DataArray
        The day of year at which snow amount reaches its maximum value.
    """
    # Identify periods where there is at least one non-null value for snow depth
    valid = at_least_n_valid(snw.where(snw > 0), n=1, freq=freq)

    # Compute doymax. Will return first time step if all snow depths are 0.
    out = generic.select_resample_op(
        snw.where(snw > 0, 0), op=generic.doymax, freq=freq
    )
    out.attrs.update(units="", is_dayofyear=np.int32(1), calendar=get_calendar(snw))

    # Mask arrays that miss at least one non-null snd.
    return out.where(~valid)


@declare_units(snw="[mass]/[area]")
def snow_melt_we_max(
    snw: xarray.DataArray, window: int = 3, freq: str = "YS-JUL"
) -> xarray.DataArray:
    """Maximum snow melt.

    The maximum snow melt over a given number of days expressed in snow water equivalent.

    Parameters
    ----------
    snw : xarray.DataArray
        Snow amount (mass per area).
    window : int
        Number of days during which the melt is accumulated.
    freq : str
        Resampling frequency.

    Returns
    -------
    xarray.DataArray
        The maximum snow melt over a given number of days for each period. [mass/area].
    """
    # Compute change in SWE. Set melt as a positive change.
    dsnw = snw.diff(dim="time") * -1

    # Sum over window
    agg = dsnw.rolling(time=window).sum()

    # Max over period
    out = agg.resample(time=freq).max(dim="time")
    out.attrs["units"] = snw.units
    return out


@declare_units(snw="[mass]/[area]", pr="[precipitation]")
def melt_and_precip_max(
    snw: xarray.DataArray, pr: xarray.DataArray, window: int = 3, freq: str = "YS-JUL"
) -> xarray.DataArray:
    """Maximum snow melt and precipitation.

    The maximum snow melt plus precipitation over a given number of days expressed in snow water equivalent.

    Parameters
    ----------
    snw : xarray.DataArray
        Snow amount (mass per area).
    pr : xarray.DataArray
        Daily precipitation flux.
    window : int
        Number of days during which the water input is accumulated.
    freq: str
        Resampling frequency.

    Returns
    -------
    xarray.DataArray
        The maximum snow melt plus precipitation over a given number of days for each period. [mass/area].
    """
    # Compute change in SWE. Set melt as a positive change.
    dsnw = snw.diff(dim="time") * -1

    # Add precipitation total
    total = rate2amount(pr) + dsnw

    # Sum over window
    agg = total.rolling(time=window).sum()

    # Max over period
    out = agg.resample(time=freq).max(dim="time")
    out.attrs["units"] = snw.units
    return out


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
    high_flow_days = (
        compare(q, op=">", right=threshold).resample(time=freq).sum(dim="time")
    )
    out = high_flow_days.mean(dim="time")
    out.attrs["units"] = "days"
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
    low_flow_days = (
        compare(q, op="<", right=threshold).resample(time=freq).sum(dim="time")
    )
    out = low_flow_days.mean(dim="time")
    out.attrs["units"] = "days"
    return out

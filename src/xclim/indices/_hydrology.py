"""Hydrological indice definitions."""

from __future__ import annotations

from functools import partial

import numpy as np
import pandas as pd
import xarray
from scipy.stats import circmean, rv_continuous
from xarray import DataArray

from xclim.core._types import DateStr, Quantified
from xclim.core.calendar import get_calendar
from xclim.core.missing import at_least_n_valid
from xclim.core.units import convert_units_to, declare_units, rate2amount, to_agg_units
from xclim.indices.generic import threshold_count
from xclim.indices.stats import standardized_index

from . import generic

try:
    import pymannkendall as mk
except ImportError:
    mk = None

__all__ = [
    "antecedent_precipitation_index",
    "aridity_index",
    "base_flow_index",
    "base_flow_index_seasonal_ratio",
    "days_with_snowpack",
    "flow_index",
    "high_flow_frequency",
    "lag_snowpack_flow_peaks",
    "low_flow_frequency",
    "melt_and_precip_max",
    "rb_flashiness_index",
    "runoff_ratio",
    "sen_slope",
    "snd_max",
    "snd_max_doy",
    "snow_melt_we_max",
    "snw_max",
    "snw_max_doy",
    "standardized_groundwater_index",
    "standardized_streamflow_index",
]


@declare_units(q="[discharge]")
def base_flow_index(q: xarray.DataArray, freq: str = "YS") -> xarray.DataArray:
    r"""
    Base flow index.

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
    r"""
    Richards-Baker flashiness index.

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


@declare_units(
    q="[discharge]",
    params="[]",
)
def standardized_streamflow_index(
    q: xarray.DataArray,
    freq: str | None = "MS",
    window: int = 1,
    dist: str | rv_continuous = "genextreme",
    method: str = "ML",
    fitkwargs: dict | None = None,
    cal_start: DateStr | None = None,
    cal_end: DateStr | None = None,
    params: Quantified | None = None,
    **indexer,
) -> xarray.DataArray:
    r"""
    Standardized Streamflow Index (SSI).

    Parameters
    ----------
    q : xarray.DataArray
        Rate of river discharge.
    freq : str, optional
        Resampling frequency. A monthly or daily frequency is expected. Option `None` assumes
        that the desired resampling has already been applied input dataset and will skip the resampling step.
    window : int
        Averaging window length relative to the resampling frequency. For example, if `freq="MS"`,
        i.e. a monthly resampling, the window is an integer number of months.
    dist : {"genextreme", "fisk"} or `rv_continuous` function
        Name of the univariate distribution, or a callable `rv_continuous` (see :py:mod:`scipy.stats`).
    method : {"APP", "ML", "PWM"}
        Name of the fitting method, such as `ML` (maximum likelihood), `APP` (approximate). The approximate method
        uses a deterministic function that does not involve any optimization.
        `PWM` should be used with a `lmoments3` distribution.
    fitkwargs : dict, optional
        Kwargs passed to ``xclim.indices.stats.fit`` used to impose values of certain parameters (`floc`, `fscale`).
    cal_start : DateStr, optional
        Start date of the calibration period. A `DateStr` is expected, that is a `str` in format `"YYYY-MM-DD"`.
        Default option `None` means that the calibration period begins at the start of the input dataset.
    cal_end : DateStr, optional
        End date of the calibration period. A `DateStr` is expected, that is a `str` in format `"YYYY-MM-DD"`.
        Default option `None` means that the calibration period finishes at the end of the input dataset.
    params : xarray.DataArray, optional
        Fit parameters.
        The `params` can be computed using ``xclim.indices.stats.standardized_index_fit_params`` in advance.
        The output can be given here as input, and it overrides other options.
    **indexer : Indexer
        Indexing parameters to compute the indicator on a temporal subset of the data.
        It accepts the same arguments as :py:func:`xclim.indices.generic.select_time`.

    Returns
    -------
    xarray.DataArray, [unitless]
        Standardized Streamflow Index.

    See Also
    --------
    xclim.indices._agro.standardized_precipitation_index : Standardized Precipitation Index.
    xclim.indices.stats.standardized_index : Standardized Index.
    xclim.indices.stats.standardized_index_fit_params : Standardized Index Fit Params.

    Notes
    -----
    * N-month SSI / N-day SSI is determined by choosing the `window = N` and the appropriate frequency `freq`.
    * Supported statistical distributions are: ["genextreme", "fisk"], where "fisk" is scipy's implementation of
       a log-logistic distribution.
    * If `params` is provided, it overrides the `cal_start`, `cal_end`, `freq`, `window`, `dist`, and `method` options.
    * "APP" method only supports two-parameter distributions. Parameter `loc` needs to be fixed to use method "APP".
    * The standardized index is bounded by ±8.21. 8.21 is the largest standardized index as constrained by the
      float64 precision in the inversion to the normal distribution.

    References
    ----------
    :cite:cts:`vicente-serrano_2012`

    Examples
    --------
    >>> from datetime import datetime
    >>> from xclim.indices import standardized_streamflow_index
    >>> ds = xr.open_dataset(path_to_q_file)
    >>> q = ds.q_sim
    >>> cal_start, cal_end = "2006-05-01", "2008-06-01"
    >>> ssi_3 = standardized_streamflow_index(
    ...     q,
    ...     freq="MS",
    ...     window=3,
    ...     dist="genextreme",
    ...     method="ML",
    ...     cal_start=cal_start,
    ...     cal_end=cal_end,
    ... )  # Computing SSI-3 months using a GEV distribution for the fit
    >>> # Fitting parameters can also be obtained first, then reused as input.
    >>> from xclim.indices.stats import standardized_index_fit_params
    >>> params = standardized_index_fit_params(
    ...     q.sel(time=slice(cal_start, cal_end)),
    ...     freq="MS",
    ...     window=3,
    ...     dist="genextreme",
    ...     method="ML",
    ... )  # First getting params
    >>> ssi_3 = standardized_streamflow_index(q, params=params)
    """
    fitkwargs = fitkwargs or {}
    dist_methods = {"genextreme": ["ML", "APP"], "fisk": ["ML", "APP"]}
    if isinstance(dist, str):
        if dist in dist_methods:
            if method not in dist_methods[dist]:
                raise NotImplementedError(f"{method} method is not implemented for {dist} distribution")
        else:
            raise NotImplementedError(f"{dist} distribution is not yet implemented.")

    zero_inflated = False
    ssi = standardized_index(
        q,
        freq=freq,
        window=window,
        dist=dist,
        method=method,
        zero_inflated=zero_inflated,
        fitkwargs=fitkwargs,
        cal_start=cal_start,
        cal_end=cal_end,
        params=params,
        **indexer,
    )

    return ssi


@declare_units(snd="[length]")
def snd_max(snd: xarray.DataArray, freq: str = "YS-JUL") -> xarray.DataArray:
    """
    Maximum snow depth.

    The maximum daily snow depth.

    Parameters
    ----------
    snd : xarray.DataArray
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
    """
    Day of year of maximum snow depth.

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
    out = generic.select_resample_op(snd.where(snd > 0, 0), op=generic.doymax, freq=freq)
    out.attrs.update(units="", is_dayofyear=np.int32(1), calendar=get_calendar(snd))

    # Mask arrays that miss at least one non-null snd.
    return out.where(~valid)


@declare_units(snw="[mass]/[area]")
def snw_max(snw: xarray.DataArray, freq: str = "YS-JUL") -> xarray.DataArray:
    """
    Maximum snow amount.

    The maximum daily snow amount.

    Parameters
    ----------
    snw : xarray.DataArray
        Snow amount (mass per area).
    freq : str
        Resampling frequency.

    Returns
    -------
    xarray.DataArray
        The maximum snow amount over a given number of days for each period. [mass/area].
    """
    return generic.select_resample_op(snw, op="max", freq=freq)


@declare_units(snw="[mass]/[area]")
def snw_max_doy(snw: xarray.DataArray, freq: str = "YS-JUL") -> xarray.DataArray:
    """
    Day of year of maximum snow amount.

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
    out = generic.select_resample_op(snw.where(snw > 0, 0), op=generic.doymax, freq=freq)
    out.attrs.update(units="", is_dayofyear=np.int32(1), calendar=get_calendar(snw))

    # Mask arrays that miss at least one non-null snd.
    return out.where(~valid)


@declare_units(snw="[mass]/[area]")
def snow_melt_we_max(snw: xarray.DataArray, window: int = 3, freq: str = "YS-JUL") -> xarray.DataArray:
    """
    Maximum snow melt.

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
    """
    Maximum snow melt and precipitation.

    The maximum snow melt plus precipitation over a given number of days expressed in snow water equivalent.

    Parameters
    ----------
    snw : xarray.DataArray
        Snow amount (mass per area).
    pr : xarray.DataArray
        Daily precipitation flux.
    window : int
        Number of days during which the water input is accumulated.
    freq : str
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


@declare_units(
    gwl="[length]",
    params="[]",
)
def standardized_groundwater_index(
    gwl: xarray.DataArray,
    freq: str | None = "MS",
    window: int = 1,
    dist: str | rv_continuous = "genextreme",
    method: str = "ML",
    fitkwargs: dict | None = None,
    cal_start: DateStr | None = None,
    cal_end: DateStr | None = None,
    params: Quantified | None = None,
    **indexer,
) -> xarray.DataArray:
    r"""
    Standardized Groundwater Index (SGI).

    Parameters
    ----------
    gwl : xarray.DataArray
        Groundwater head level.
    freq : str, optional
        Resampling frequency. A monthly or daily frequency is expected. Option `None` assumes
        that the desired resampling has already been applied input dataset and will skip the resampling step.
    window : int
        Averaging window length relative to the resampling frequency. For example, if `freq="MS"`,
        i.e. a monthly resampling, the window is an integer number of months.
    dist : {"gamma", "genextreme", "lognorm"} or `rv_continuous`
        Name of the univariate distribution, or a callable `rv_continuous` (see :py:mod:`scipy.stats`).
    method : {"APP", "ML", "PWM"}
        Name of the fitting method, such as `ML` (maximum likelihood), `APP` (approximate).
        The approximate method uses a deterministic function that does not involve any optimization.
        `PWM` should be used with a `lmoments3` distribution.
    fitkwargs : dict, optional
        Kwargs passed to ``xclim.indices.stats.fit`` used to impose values of certain parameters (`floc`, `fscale`).
    cal_start : DateStr, optional
        Start date of the calibration period. A `DateStr` is expected, that is a `str` in format `"YYYY-MM-DD"`.
        Default option `None` means that the calibration period begins at the start of the input dataset.
    cal_end : DateStr, optional
        End date of the calibration period. A `DateStr` is expected, that is a `str` in format `"YYYY-MM-DD"`.
        Default option `None` means that the calibration period finishes at the end of the input dataset.
    params : xarray.DataArray, optional
        Fit parameters.
        The `params` can be computed using ``xclim.indices.stats.standardized_index_fit_params`` in advance.
        The output can be given here as input, and it overrides other options.
    **indexer : Indexer
        Indexing parameters to compute the indicator on a temporal subset of the data.
        It accepts the same arguments as :py:func:`xclim.indices.generic.select_time`.

    Returns
    -------
    xarray.DataArray, [unitless]
        Standardized Groundwater Index.

    See Also
    --------
    xclim.indices._agro.standardized_precipitation_index : Standardized Precipitation Index.
    xclim.indices.stats.standardized_index : Standardized Index.
    xclim.indices.stats.standardized_index_fit_params : Standardized Index Fit Params.

    Notes
    -----
    * N-month SGI / N-day SGI is determined by choosing the `window = N` and the appropriate frequency `freq`.
    * Supported statistical distributions are: ["gamma", "genextreme", "lognorm"].
    * If `params` is provided, it overrides the `cal_start`, `cal_end`, `freq`, `window`, `dist`, `method` options.
    * "APP" method only supports two-parameter distributions. Parameter `loc` needs to be fixed to use method "APP".

    References
    ----------
    :cite:cts:`bloomfield_2013`

    Examples
    --------
    >>> from datetime import datetime
    >>> from xclim.indices import standardized_groundwater_index
    >>> ds = xr.open_dataset(path_to_gwl_file)
    >>> gwl = ds.gwl
    >>> cal_start, cal_end = "1980-05-01", "1982-06-01"
    >>> sgi_3 = standardized_groundwater_index(
    ...     gwl,
    ...     freq="MS",
    ...     window=3,
    ...     dist="gamma",
    ...     method="ML",
    ...     cal_start=cal_start,
    ...     cal_end=cal_end,
    ... )  # Computing SGI-3 months using a Gamma distribution for the fit
    >>> # Fitting parameters can also be obtained first, then reused as input.
    >>> from xclim.indices.stats import standardized_index_fit_params
    >>> params = standardized_index_fit_params(
    ...     gwl.sel(time=slice(cal_start, cal_end)),
    ...     freq="MS",
    ...     window=3,
    ...     dist="gamma",
    ...     method="ML",
    ... )  # First getting params
    >>> sgi_3 = standardized_groundwater_index(gwl, params=params)
    """
    fitkwargs = fitkwargs or {}

    dist_methods = {
        "gamma": ["ML", "APP"],
        "genextreme": ["ML", "APP"],
        "lognorm": ["ML", "APP"],
    }
    if isinstance(dist, str):
        if dist in dist_methods:
            if method not in dist_methods[dist]:
                raise NotImplementedError(f"{method} method is not implemented for {dist} distribution")
        else:
            raise NotImplementedError(f"{dist} distribution is not yet implemented.")

    zero_inflated = False
    sgi = standardized_index(
        gwl,
        freq=freq,
        window=window,
        dist=dist,
        method=method,
        zero_inflated=zero_inflated,
        fitkwargs=fitkwargs,
        cal_start=cal_start,
        cal_end=cal_end,
        params=params,
        **indexer,
    )

    return sgi


@declare_units(q="[discharge]")
def flow_index(q: xarray.DataArray, p: float = 0.95) -> xarray.DataArray:
    """
    Flow index.

    Calculate the pth percentile of daily streamflow normalized by the median flow.

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

    References
    ----------
    :cite:cts:`Clausen2000`
    """
    qp = q.quantile(p, dim="time")
    q_median = q.median(dim="time")
    out = qp / q_median
    out.attrs["units"] = "1"
    return out


@declare_units(q="[discharge]")
def high_flow_frequency(q: xarray.DataArray, threshold_factor: int = 9, freq: str = "YS-OCT") -> xarray.DataArray:
    """
    High flow frequency.

    Calculate the number of days in a given period with flows greater than a specified threshold, given as a
    multiple of the median flow. By default, the period is the water year starting on 1st October and ending on
    30th September, as commonly defined in North America.

    Parameters
    ----------
    q : xarray.DataArray
        Daily streamflow data.
    threshold_factor : int
        Factor by which the median flow is multiplied to set the high flow threshold, default is 9.
    freq : str
        Resampling frequency, default is 'YS-OCT' for water year starting in October and ending in September.

    Returns
    -------
    xarray.DataArray
        Number of high flow days.

    References
    ----------
    :cite:cts:`addor2018,Clausen2000`
    """
    median_flow = q.median(dim="time")
    threshold = threshold_factor * median_flow
    out = threshold_count(q, ">", threshold, freq=freq)
    return to_agg_units(out, q, "count", deffreq="D")


@declare_units(q="[discharge]")
def low_flow_frequency(q: xarray.DataArray, threshold_factor: float = 0.2, freq: str = "YS-OCT") -> xarray.DataArray:
    """
    Low flow frequency.

    Calculate the number of days in a given period with flows lower than a specified threshold, given by a fraction
    of the mean flow. By default, the period is the water year starting on 1st October and ending on 30th September,
    as commonly defined in North America.

    Parameters
    ----------
    q : xarray.DataArray
        Daily streamflow data.
    threshold_factor : float
        Factor by which the mean flow is multiplied to set the low flow threshold, default is 0.2.
    freq : str
        Resampling frequency, default is 'YS-OCT' for water year starting in October and ending in September.

    Returns
    -------
    xarray.DataArray
        Number of low flow days.

    References
    ----------
    :cite:cts:`Olden2003`
    """
    mean_flow = q.mean(dim="time")
    threshold = threshold_factor * mean_flow
    out = threshold_count(q, "<", threshold, freq=freq)
    return to_agg_units(out, q, "count", deffreq="D")


@declare_units(pr="[precipitation]")
def antecedent_precipitation_index(pr: xarray.DataArray, window: int = 7, p_exp: float = 0.935) -> xarray.DataArray:
    """
    Antecedent Precipitation Index.

    Calculate the running weighted sum of daily precipitation values given a window and weighting exponent.
    This index serves as an indicator for soil moisture.

    Parameters
    ----------
    pr : xarray.DataArray
        Daily precipitation data.
    window : int
        Window for the days of precipitation data to be weighted and summed, default is 7.
    p_exp : float
        Weighting exponent, default is 0.935.

    Returns
    -------
    xarray.DataArray
        Antecedent Precipitation Index.

    References
    ----------
        :cite:cts:`schroter2015,li2021`
    """
    pr = rate2amount(pr)
    pr = convert_units_to(pr, "mm", context="hydro")
    weights = xarray.DataArray(
        list(reversed([p_exp ** (idx - 1) for idx in range(1, window + 1)])),
        dims="window_dim",
    )
    out = pr.rolling(time=window).construct("window_dim").dot(weights)
    out.attrs["units"] = "mm"
    return out


@declare_units(q="[discharge]", a="[area]", pr="[precipitation]")
def runoff_ratio(
    q: xarray.DataArray,
    a: xarray.DataArray,
    pr: xarray.DataArray,
    freq: str = "YS",
) -> xarray.DataArray:
    """
    Runoff ratio.

    Runoff ratio: Ratio of runoff volume measured at the stream to the total
    precipitation volume over the watershed.
    Temporal analysis: Yearly values computed from seasonal daily data and yearly data, depending on chosen frequency.
    (e.g., 'YS' for yearly starting Jan, or 'QS-DEC' for seasons,
    '30YS' to compute the value over slices of 30 years from the start of the time series).

    Parameters
    ----------
    q : xarray.DataArray
        Streamflow in discharge units. Will be converted to [m³/s].
    a : xarray.DataArray
        Watershed area in area units. Will be converted to [km²].
    pr : xarray.DataArray
        Mean daily precipitation in precipitation units. Will be converted to [mm/hr].
    freq : str
        Resampling frequency.

    Returns
    -------
    xarray.DataArray
        Out: xarray.DataArray
            Runoff ratio (dimensionless).

    Notes
    -----
    - Runoff ratio values are comparable to runoff coefficients.
    - Values near 0 mean most precipitation infiltrates watershed soil
      or is lost to evapotranspiration.
    - Values near 1 mean most precipitation leaves the watershed as runoff.
      Possible causes are impervious surfaces from urban sprawl, thin soils,
      steep slopes, etc.
    - Annual runoff ratios are typically ≤ 1.
    - Annual runoff ratios are typically higher than summer runoff ratios due to
      higher levels of evapotranspiration in summer months.
    - For snow-driven watersheds, spring runoff ratios are typically higher than
      annual runoff ratios, as snowmelt generates concentrated runoff events.

    References
    ----------
    :cite:cts:'knoben_2024'
    """
    q = convert_units_to(q, "m3/s")
    a = convert_units_to(a, "km2")
    pr = convert_units_to(pr, "mm/hr")

    runoff = q * 3.6 / a  # unit conversion for runoff in mm/hr : 3.6[s/hr *km2/m2]
    runoff_freq = runoff.resample(time=freq).sum()
    pr_freq = pr.resample(time=freq).sum()

    out = runoff_freq / pr_freq
    out.attrs["units"] = ""
    return out


@declare_units(swe="[length]", thresh="[length]")
def days_with_snowpack(
    swe: xarray.DataArray,
    thresh: str = "10 mm",
    freq: str = "YS-OCT",
) -> xarray.DataArray:
    """
    Days with snowpack.

    Number of days with snow water equivalent (SWE) above a given threshold.

    Parameters
    ----------
    swe : xarray.DataArray
        Daily surface snow amount as snow water equivalent.
    thresh : float
        Minimum snow quantity to consider a given day snow-covered. Default is 10 mm.
    freq : str
        Resampling frequency. Typically the water year starting on the 1st of October
        in the Northern Hemisphere.

    Returns
    -------
    xarray.DataArray
        Number of days with snowpack above the threshold.

    Notes
    -----
    Years with larger snowpacks tend to produce bigger spring floods.
    Additional spring flood analysis can be carried out using the
    ``annual_maxima`` and ``lag_snowpack_flow_peaks`` functions.

    References
    ----------
    :cite:cts:`alonso_gonzalez_2022`
    """
    frz = convert_units_to(thresh, swe)
    out = threshold_count(swe, ">", frz, freq)
    return to_agg_units(out, swe, "count", deffreq="D")


@declare_units(pr="[precipitation]", pet="[precipitation]")
def aridity_index(pr: xarray.DataArray, pet: xarray.DataArray, freq: str = "YS") -> xarray.DataArray:
    """
    Aridity index.

    The ratio of total precipitation over potential evapotranspiration.
    Classification based on the Aridity Index (AI).
    +----------------+----------------+-----------------+
    | Classification | Aridity Index  | Global land area|
    +----------------+----------------+-----------------+
    | Hyperarid      | AI < 0.05      | 7.5%            |
    | Arid           | 0.05 ≤ AI < 0.20 | 12.1%         |
    | Semi-arid      | 0.20 ≤ AI < 0.50 | 17.7%         |
    | Dry subhumid   | 0.50 ≤ AI < 0.65 | 9.9%          |
    +----------------+----------------+-----------------+

    Parameters
    ----------
    pr : array_like
        Precipitation.
    pet : array_like
        Potential evapotranspiration.
    freq : str
        Resampling frequency. A monthly or yearly frequency is expected.

    Returns
    -------
    float
        Aridity index per time step (Unitless).

    Notes
    -----
    - An aridity index below 0.65 indicates an arid environment,
      while values above this threshold correspond to more humid environments.
    - In North America, higher aridity index values can be associated with colder climates
      due to lower evapotranspiration, even when precipitation is limited or occurring as snow.

    References
    ----------
    :cite:cts:'zomer_2022'
    """
    pr = pr.resample(time=freq).sum()
    pet = pet.resample(time=freq).sum()
    ai = pr / pet
    ai.attrs["units"] = ""

    return ai


@declare_units(swe="[length]", q="[discharge]")
def lag_snowpack_flow_peaks(
    swe: xarray.DataArray,
    q: xarray.DataArray,
    freq: str = "YS-OCT",
    percentile: int = 90,
) -> xarray.DataArray:
    """
    Time lag between maximum snowpack and river high flows.

    Number of days between the annual maximum snowpack, measured by the snow water
    equivalent, and the mean date when river flow exceeds a percentile threshold
    during a given year.
    If the time lag between maximum snowpack and river high flows is ≤ 50 days,
    the watershed is likely in a nival regime.

    Parameters
    ----------
    swe : xarray.DataArray
        Surface snow amount as snow water equivalent.
    q : xarray.DataArray
        Streamflow.
    freq : str
        Resampling frequency. Defaults to the water year starting on the 1st of October.
    percentile : float
        Percentile threshold identifying high flows. Defaults to the 90th percentile.

    Returns
    -------
    xarray.DataArray
        Number of days between maximum snowpack and the circular mean date of high flow days.

    See Also
    --------
    xclim.indices.rb_flashiness_index: Richards-Baker flashiness index.

    Notes
    -----
    - The default ``freq`` is the water year used in the Northern Hemisphere, from October to September.
    - It is recommended to have at least 70% of valid data per water year in order to compute significant values.
    - Nival regime is characterized by a hydrological response dominated by snowmelt,
      where maximum flows occur shortly after peak snow cover (Burn et al., 2010).
    - The 50-day threshold is approximate and depends on the specific responsiveness of each watershed.
    - A negative value means the high flows occur before the peak snow cover.

    References
    ----------
    :cite:cts:`burn_2010`
    """
    # Find time of max SWE per year
    t_swe_max = swe.resample(time=freq).map(lambda x: x.idxmax())  # if x.max() > 0 else np.nan)
    t_swe_max = t_swe_max.where(swe.resample(time=freq).max() > 0)
    doy_swe_max = t_swe_max.dt.dayofyear

    # Compute percentile threshold per water year using resample
    thresh = q.resample(time="YS-OCT").reduce(
        np.nanpercentile, q=percentile, dim="time"
    )  # the second q, equal to percentile, is a keyword in np.nanpercentile, not the flow variable.
    threshold_for_each_time = thresh.reindex_like(q, method="ffill")
    q_high = q.where(q >= threshold_for_each_time).dropna(dim="time", how="all")

    # Day of year for high flow peaks
    doy = q_high.time.dt.dayofyear

    t_q_max = doy.resample(time=freq).reduce(partial(circmean, high=366, low=1), dim="time")

    # Compute lag
    lag = t_q_max - doy_swe_max
    lag.attrs["units"] = "days"
    return lag


@declare_units(q="[discharge]")
def sen_slope(
    q: xarray.DataArray,
    qsim: xarray.DataArray = None,
    freq: str = "YS",
    month: str = "DEC",
) -> xarray.Dataset:
    """
    Temporal robustness analysis of streamflow.

    Computes annual and seasonal Theil–Sen slope estimators and performs the
    Mann–Kendall test for trend evaluation.

    Parameters
    ----------
    q : xarray.DataArray
        Observed streamflow vector.
    qsim : xarray.DataArray
        Simulated streamflow vector.
    freq : str
        Resampling frequency. Default yearly
    month : str
        Anchor month in order to spit year into quarterly seasons

    Returns
    -------
    xarray.Dataset
        Dataset containing the following variables:

        - ``Sen_slope`` : Sen's slope estimates for seasonal and yearly averages.
        - ``p_value`` : Mann–Kendall metric indicating slope tendency.
        - If simulated flows are provided: ``Sen_slope_sim``, ``p_value_sim``,
          and the ratio of observed ``Sen_slope`` over simulated ``Sen_slope``.

    Notes
    -----
    - If p-value <= 0.05, the trend is statistically significant at the 5% level.
    - The ratio of observed Sen_slope over simulated Sen_slope is considered
      acceptable within the range 0.5–2 and is optimal when equal to 1
      (Sauquet et al., 2025).

    References
    ----------
    :cite:cts:`sauquet_2025`
    """
    seasons = ["DJF", "MAM", "JJA", "SON", "Year"]

    def compute_seasonal_stats(x: xarray.DataArray) -> tuple[list, list]:
        """
        Seasonal statistics.

        Parameters
        ----------
        x : xarray.DataArray
            Time series of streamflow.

        Returns
        -------
        tuple of list
            Lists containing the following variables:

            - ``Sen_slope`` : Sen's slope estimates for seasonal and yearly averages.
            - ``p_value`` : Mann–Kendall metric indicating slope tendency.
        """
        if mk is None:
            msg = f"{sen_slope.__name__} requires access to the `pymannkendall` library."
            raise ModuleNotFoundError(msg)

        # Convert to pandas Series with DatetimeIndex
        x_year = x.resample(time="YS-DEC").mean()
        x_season = x.resample(time="QS-DEC").mean()

        x_series = x_season.to_series()

        # Create a MultiIndex: year + season (0–3)
        season_index = (
            x_series.index.month % 12 // 3  # 0 for DJF, 1 for MAM, etc.
        )
        x_df = pd.DataFrame({"value": x_series.values, "season": season_index, "year": x_series.index.year})
        #  Pivot to shape (n_years, 4 seasons)
        df_seasons = x_df.pivot(index="season", columns="year", values="value")

        # rename columns
        df_seasons.index = ["DJF", "MAM", "JJA", "SON"]

        ss_DJF = mk.original_test(df_seasons.iloc[0])
        ss_MAM = mk.original_test(df_seasons.iloc[1])
        ss_JJA = mk.original_test(df_seasons.iloc[2])
        ss_SON = mk.original_test(df_seasons.iloc[3])
        ss_an = mk.original_test(x_year)

        _slopes = [ss_DJF.slope, ss_MAM.slope, ss_JJA.slope, ss_SON.slope, ss_an.slope]
        _p_vals = [ss_DJF.p, ss_MAM.p, ss_JJA.p, ss_SON.p, ss_an.p]

        return _slopes, _p_vals

    if qsim is not None:
        slopes, p_vals = compute_seasonal_stats(q)
        slopes_sim, p_vals_sim = compute_seasonal_stats(qsim)
        slopes_np = np.array(slopes)
        slopes_sim_np = np.array(slopes_sim)
        ratio = slopes_np / slopes_sim_np
        ds = xarray.Dataset(
            data_vars={
                "Sen_slope_obs": ("season", slopes),
                "p_value_obs": ("season", p_vals),
                "Sen_slope_sim": ("season", slopes_sim),
                "p_value_sim": ("season", p_vals_sim),
                "ratio": ("season", ratio),
            },
            coords={"season": seasons},
        )
        for var in ds.data_vars:
            ds[var].attrs["units"] = ""
        return (
            ds["Sen_slope_obs"],
            ds["p_value_obs"],
            ds["Sen_slope_sim"],
            ds["p_value_sim"],
            ds["ratio"],
        )
    else:
        slopes, p_vals = compute_seasonal_stats(q)
        # Create labeled xarray
        ds = xarray.Dataset(
            data_vars={"Sen_slope": ("season", slopes), "p_value": ("season", p_vals)}, coords={"season": seasons}
        )

        # Assign empty units to all variables
        return (
            ds["Sen_slope"],
            ds["p_value"],
        )


@declare_units(q="[discharge]")
def base_flow_index_seasonal_ratio(
    q: xarray.DataArray, freq: str = "QS-DEC"
) -> tuple[DataArray, DataArray, DataArray, DataArray, DataArray]:
    """
    Seasonal Base flow index (bfi) and ratio of winter to summer base flow index.

    Return yearly base flow index per season, defined as the minimum 7-day average flow divided by the mean flow
    as well as yearly winter to summer bfi ratio.

    Parameters
    ----------
    q : xarray.DataArray
        Rate of river discharge.
    freq : str
        Resampling frequency.

    Returns
    -------
    xarray.DataArray, [dimensionless]
        Winter_bfi.
    xarray.DataArray, [dimensionless]
        Spring_bfi.
    xarray.DataArray, [dimensionless]
        Summer_bfi.
    xarray.DataArray, [dimensionless]
        Fall_bfi.
    xarray.DataArray, [dimensionless]
        Winter_to summer_ratio.

    Notes
    -----
    It is recommended to have at least 70% of valid data per month in order to compute significant values.

    References
    ----------
    :cite:cts:`singh_2019`
    :cite:cts:`jaffres_2021`
    """
    # 7-day minimum of raw daily flow
    q7min = q.rolling(time=7, center=True).min(skipna=False).resample(time=freq).min()

    qmean = q.resample(time=freq).mean()
    bfi = q7min / qmean

    winter_bfi = bfi.where(bfi["time"].dt.month == 12).groupby("time.year").mean(dim="time")
    winter_bfi.attrs["units"] = ""

    spring_bfi = bfi.where(bfi["time"].dt.month == 3).groupby("time.year").mean(dim="time")
    spring_bfi.attrs["units"] = ""

    summer_bfi = bfi.where(bfi["time"].dt.month == 6).groupby("time.year").mean(dim="time")
    summer_bfi.attrs["units"] = ""

    fall_bfi = bfi.where(bfi["time"].dt.month == 9).groupby("time.year").mean(dim="time")
    fall_bfi.attrs["units"] = ""

    # Shift timestamp forward by one year since winter starts in dec the year prior
    winter_1 = winter_bfi.assign_coords(year=winter_bfi["year"] + 1)
    winter_1_aligned = winter_1.reindex(year=summer_bfi.year)
    epsilon = 1e-3  # To avoid division by zero

    w_s_ratio = (winter_1_aligned / (summer_bfi + epsilon)).sel(year=summer_bfi.year)
    w_s_ratio = w_s_ratio.isel(year=slice(1, None))
    w_s_ratio.attrs["units"] = ""

    return winter_bfi, spring_bfi, summer_bfi, fall_bfi, w_s_ratio

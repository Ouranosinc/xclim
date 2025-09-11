"""Hydrological indice definitions."""

from __future__ import annotations

from functools import partial

import numpy as np
import pandas as pd
import pymannkendall as mk
import xarray
from scipy.stats import circmean, rv_continuous
from xarray import Dataset

from xclim.core._types import DateStr, Quantified
from xclim.core.calendar import get_calendar
from xclim.core.missing import at_least_n_valid
from xclim.core.units import convert_units_to, declare_units, rate2amount, to_agg_units
from xclim.indices.generic import threshold_count
from xclim.indices.stats import standardized_index

from . import generic

__all__ = [
    "annual_maxima",
    "annual_maxima",
    "antecedent_precipitation_index",
    "aridity_index",
    "base_flow_index",
    "days_with_snowpack",
    "elasticity_index",
    "fdc_slope",
    "flow_index",
    "high_flow_frequency",
    "lag_snowpack_flow_peaks",
    "low_flow_frequency",
    "melt_and_precip_max",
    "rb_flashiness_index",
    "season_annual_runoff_ratio",
    "sen_slope",
    "sen_slope",
    "snd_max",
    "snd_max_doy",
    "snow_melt_we_max",
    "snw_max",
    "snw_max_doy",
    "standardized_groundwater_index",
    "standardized_streamflow_index",
    "total_runoff_ratio",
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
    freq : str, optional
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
    freq : str, optional
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
def total_runoff_ratio(q: xarray.DataArray, a: xarray.DataArray, pr: xarray.DataArray) -> xarray.DataArray:
    """
    Total Runoff ratio

    Runoff ratio : ratio of runoff volume measured at the stream to the total precipitation volume over the watershed.
    Aggregated analysis : Single value as a long-term benchmark

    Parameters
    ----------
    q : xarray.DataArray
        Streamflow in [discharge] units, will be converted to [m3/s].
    a : xarray.DataArray
        Watershed area [area] units, will be converted to in [km²].
    pr : xarray.DataArray
        mean daily Precipitation [precipitation] units, will be converted to [mm/hr].

    Returns
    -------
    xarray.DataArray
        Single value rainfall-runoff ratio (RRR) as long-term benchmark.

    Notes
    -----
    - Total Runoff ratio values are comparable to Runoff coefficients,
    - Values near 0 mean most precipitation infiltrates watershed soil or is lost to evapotranspiration.
    - Values near 1 mean most precipitation leaves the watershed as runoff;
      possible causes are impervious surfaces from urban sprawl, thin soils, steep slopes, etc.
    - Long-term averages are typically ≤ 1

    Reference
    -----
    HydroBM https://hydrobm.readthedocs.io/en/latest/usage.html#benchmarks
    """
    q = convert_units_to(q, "m3/s")
    a = convert_units_to(a, "km2")
    pr = convert_units_to(pr, "mm/hr")

    runoff = q * 3.6 / a  # unit conversion for runoff in mm/h : 3.6 [s/h * km2/m2]
    total_rr = runoff.sum() / pr.sum()
    total_rr.attrs["units"] = ""
    total_rr.attrs["long_name"] = "Total Rainfall-Runoff Ratio"

    return total_rr


@declare_units(q="[discharge]", a="[area]", pr="[precipitation]")
def season_annual_runoff_ratio(
    q: xarray.DataArray,
    a: xarray.DataArray,
    pr: xarray.DataArray,
) -> xarray.DataArray:
    """
    Seasonal and annual runoff ratio

    Runoff ratio : Ratio of runoff volume measured at the stream to the total precipitation volume over the watershed,
    Temporal analysis : Yearly values computed from Seasonal daily data and yearly data.

    Parameters
    ----------
    q : xarray.DataArray
        Streamflow in [discharge] units, will be converted to [m3/s].
    a : xarray.DataArray
        Watershed area in [area] units, will be converted to in [km²].
    pr : xarray.DataArray
        Mean daily Precipitation in [precipitation] units, will be converted to [mm/hr].


    Returns
    -------
    xarray.DataArray
        Seasonal and yearly runoff ratio (dimensionless)
        Where 'DJF' = Winter months, 'JJA' = Summer months, 'MAM' = Spring months, 'SON' = Fall months

    Notes
    -----
    - Runoff ratio values are comparable to Runoff coefficients,
    - Values near 0 mean most precipitation infiltrates watershed soil or is lost to evapotranspiration.
    - Values near 1 mean most precipitation leaves the watershed as runoff;
      possible causes are impervious surfaces from urban sprawl, thin soils, steep slopes, etc.
    - Annual runoff ratios are typically ≤ 1.
    - Annual runoff ratios are typically of higher values than summer runoff ratios due to
      higher levels of evapotranspiration in summer months.
    - For snow-driven watersheds, spring runoff ratios are typically higher than Annual runoff ratios,
      as snowmelt generates concentrated runoff events.

    Reference
    -----
    HydroBM https://hydrobm.readthedocs.io/en/latest/usage.html#benchmarks

    """
    q = convert_units_to(q, "m3/s")
    a = convert_units_to(a, "km2")
    pr = convert_units_to(pr, "mm/hr")

    runoff = q * 3.6 / a  # unit conversion for runoff in mm/hr : 3.6[s/hr *km2/m2]

    season_year = q["time"].dt.season.str.cat(q["time"].dt.year.astype(str), sep="-")

    runoff.coords["season_year"] = ("time", season_year.data)
    pr.coords["season_year"] = ("time", season_year.data)

    # separate season and year coordinates from season_year strings:
    seasons = [s.split("-")[0] for s in season_year.values]
    years = [int(s.split("-")[1]) for s in season_year.values]

    # Assign as new coordinates on the original time dimension:
    runoff.coords["season"] = ("time", seasons)
    runoff.coords["year"] = ("time", years)

    pr.coords["season"] = ("time", seasons)
    pr.coords["year"] = ("time", years)

    # Group by season-year and sum
    runoff_seasonal = runoff.groupby(["season", "year"]).sum(dim="time", skipna=True)
    pr_seasonal = pr.groupby(["season", "year"]).sum(dim="time", skipna=True)

    rrr_season = runoff_seasonal / pr_seasonal

    # Group by year and sum
    runoff_year = runoff.groupby(["year"]).sum(dim="time", skipna=True)
    pr_year = pr.groupby(["year"]).sum(dim="time", skipna=True)

    rrr_yearly = runoff_year / pr_year
    rrr_season.attrs["units"] = ""
    rrr_yearly.attrs["units"] = ""

    return rrr_season, rrr_yearly


@declare_units(q="[discharge]", pr="[precipitation]")
def elasticity_index(q: xarray.DataArray, pr: xarray.DataArray, freq: str = "YS") -> xarray.DataArray:
    """
    Calculate the median of yearly streamflow elasticity index for given catchments,
    where elasticity (εₚ) is defined as the relative change in streamflow (ΔQ/Q)
    divided by the relative change in precipitation (ΔP/P)

    Aggregated reactivity analysis : single value as a long-term benchmark

    Parameters
    ----------
    q : xarray.DataArray
        Daily discharge data
    pr : xarray.DataArray
        Daily precipitation data
    freq: Resampling frequency (e.g., 'YS' for year starting in Jan)

    Returns
    -------
    xarray.DataArray
        nonparametric estimator for streamflow elasticity index (dimensionless)


    Note
    -------

    A value of εp greater than 1 indicates that streamflow is highly sensitive to precipitation changes,
    meaning a 1% change in precipitation will lead to a greater than 1% change in streamflow.
    A value less than 1 suggests a less sensitive relationship.

    References
    ----------
    Sankarasubramanian, A., Vogel, R. M., & Limbrunner, J. F. (2001). Climate elasticity of streamflow
    in the United States. Water Resources Research, 37(6), 1771–1781. https://doi.org/10.1029/2000WR900330

    """
    # Calculate single value streamflow elasticity index per catchment

    p_annual = pr.resample(time=freq).mean()
    q_annual = q.resample(time=freq).mean()

    p_mean = p_annual.mean(dim="time")
    q_mean = q_annual.mean(dim="time")

    # Year-to-year changes
    delta_p = p_annual.diff(dim="time")
    delta_q = q_annual.diff(dim="time")

    # Avoid division by zero
    epsilon = 1e-6

    # Relative changes
    rel_delta_p = delta_p / (p_mean + epsilon)
    rel_delta_q = delta_q / (q_mean + epsilon)

    # Cumpute yearly streamflow elasticity (not mathematically robust values)
    yearly_elasticity = rel_delta_q / (rel_delta_p + epsilon)

    # Cumpute single value using median (more robust value)
    elasticity_index = yearly_elasticity.median(dim="time")
    elasticity_index.attrs["units"] = ""
    return elasticity_index


@declare_units(swe="[length]", thresh="[length]")
def days_with_snowpack(
    swe: xarray.DataArray,
    thresh: str = "10 mm",
    freq: str = "YS-OCT",
) -> xarray.DataArray:
    """
    Count days per year with snowpack on the ground above a given threshold.

    Parameters
    ----------
    swe : xarray.DataArray
        Daily surface snow amount (or SWE: snow water equivalent), [kg/m2] or in millimeters of water [mm]
    thresh : float, optional
        Minimum swe value to consider a snow-covered day. Default is 10 mm.
    freq: str, optional
        Resampling frequency, here water year stating on the 1st of October

    Returns
    -------
    xarray.DataArray, [days]
        Number of days with snowpack over threshold

    Warnings
    --------
    The default `freq` is the water year used in the northern hemisphere, from October to September.
    It is recommended to have at least 70% of valid data per water year in order to compute significant values.

    Note
    -------
    Years with larger snowpacks tend to produce bigger spring floods.
    Additional Spring flood analysis can be carried out using the annual_maxima and lag_snowpack_flow_peaks functions.

    References
    ----------
    Alonso-González, E., Revuelto, J., Fassnacht, S. R., & López-Moreno, J. I. (2022).
    Combined influence of maximum accumulation and melt rates on the duration of
    the seasonal snowpack over temperate mountains. Journal of Hydrology, 608, 127574.
    """
    thresh = convert_units_to(thresh, swe)

    # compute signature:
    days_with_sp = swe >= thresh
    result = days_with_sp.resample(time=freq).sum()  # convert results to water years
    result = result.rename("days_with_snowpack")
    result["year"] = result["time.year"]
    result = result.set_index(time="year")
    result.attrs["units"] = "days"
    return result


@declare_units(pr="[precipitation]", pet="[precipitation]")
def aridity_index(pr: xarray.DataArray, pet: xarray.DataArray, freq: str = "YS") -> xarray.DataArray:
    """
    Aridity index: Ratio of total precipitation (PR) over potential evapotranspiration (PET)

    Parameters
    ----------
    pr : array_like
        Precipitation
    pet : array_like
        Potential Evapotranspiration
    freq : str, optional
        Resampling frequency. A monthly or yearly frequency is expected. Option `None` assumes
        that the desired resampling has already been applied input dataset and will skip the resampling step.

    Returns
    -------
    float
        Aridity index per time step. (Unitless)

    Notes
    -----
    - An aridity index below 0.65 indicates an arid environment,
      while values above this threshold correspond to more humid environments.
    - Northern regions tend to have lower evapotranspiration due to colder temperatures.
      Therefore, higher aridity index values are generally associated with colder climates
      characterized by snow precipitation.

    References
    ----------
    Zomer, R. J., Xu, J., & Trabucco, A. (2022). Version 3 of the Global Aridity Index and
    Potential Evapotranspiration Database. Scientific Data, 9(1), 409. https://doi.org/10.1038/s41597-022-01493-1
    """
    pr = pr.resample(time=freq).sum()
    pet = pet.resample(time=freq).sum()
    ai = pr / pet
    ai.attrs["units"] = ""
    ai.name = "aridity_index"

    return ai


@declare_units(swe="[length]", q="[discharge]")
def lag_snowpack_flow_peaks(
    swe: xarray.DataArray,
    q: xarray.DataArray,
    freq: str = "YS-OCT",
    percentile: int = 90,
    min_ratio_missing_days: float = None,
) -> xarray.DataArray:
    """
    High flow temporal analysis: Calculate lag in days between max SWE and circular mean high flows of each year

    Parameters
    ----------
    swe : xarray.DataArray
        Daily surface snow amount (or SWE: snow water equivalent), [kg/m-2] or in millimeters of water [mm]
    q : xarray.DataArray
        Daily streamflow [e.g., m³/s]
    freq: str, optional
        Resampling frequency, here water year stating on the 1st of October
    percentile: int, optional
        frequency percentile for high flows, default is 90% of non-exceedance frequency.

    Returns
    -------
    xarray.DataArray, [days]
    Calculation of the number of days between maximum snowpack (SWE) and
    the circular average of high flows (flows with non-exceedance frequency established by percentile variable).

    Warnings
    --------
    The default `freq` is the water year used in the northern hemisphere, from October to September
    It is recommended to have at least 70% of valid data per water year in order to compute significant values.

    Note
    -------
    - Nival regime is characterized by a hydrological response dominated by snowmelt,
      where maximum flows occur shortly after peak snow cover (Burn et al., 2010).
    - A lag of 50 days or less indicates that the watershed is possibly in a nival regime.
    - This 50-day interval is approximate, as it depends on the specific responsiveness of each watershed.
    - Responsiveness can be analysed with the rb_flashiness_index and elasticity_index functions.
    - A negative value means the high flows occur before the peak snow cover.

    References
    ----------
    Burn, D. H., Sharif, M., & Zhang, K. (2010). Detection of trends in hydrological extremes for Canadian watersheds.
    Hydrological Processes, 24(13), 1781–1790. https://doi.org/10.1002/hyp.7625
    """
    # Find time of max SWE per year
    t_swe_max = swe.resample(time=freq).map(lambda x: x.idxmax())
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
    lag.name = "lag_snowpack_flow_peaks"
    return lag


@declare_units(q="[discharge]")
def sen_slope(q: xarray.DataArray, qsim: xarray.DataArray = None) -> Dataset:
    """
    Temporal robustness analysis: Annual and Seasonal Theil-Sen Slope (SS) estimators and
    Mann-Kendall test for trend evaluations

    Parameters
    ----------
    q : array_like
        Observed streamflow vector.
    qsim : array_like
        Simulated streamflow vector.

    Returns
    -------
    xr.Dataset
        'Sen_slope': Sen's slope estimates for season averages and yearly average
        'p_value': Mann-Kendall metric to verify overall slope tendency.

        If simulated flows are provided :
        Sen_slope_sim, p_value_sim and ratio of observed Sen_slope over simulated Sen_slope are returned as well.

    Notes
    -----
    If p-value <= 0.05, the trend is statistically significant at the 5% level.
    Ratio of observed Sen_slope over simulated Sen_slope is considered acceptable within the range 0.5 to 2 and
    is optimal when equal to 1. (Sauquet et al., 2025)

    References
    ----------
    Hussain et al., (2019). pyMannKendall: a Python package for non parametric Mann-Kendall family of trend tests.
    Journal of Open Source Software, 4(39), 1556, https://doi.org/10.21105/joss.01556
    https://pypi.org/project/pymannkendall/

    Sauquet, E., Evin, G., Siauve, S., Aissat, R., Arnaud, P., Bérel, M., Bonneau, J., Branger, F., Caballero, Y.
    , Colléoni, F., Ducharne, A., Gailhard, J., Habets, F., Hendrickx, F., Héraut, L., Hingray, B., Huang, P.,
    Jaouen, T., Jeantet, A., … Vidal, J.-P. (2025). A large transient multi-scenario multi-model ensemble
    of future streamflow and groundwater projections in France. https://doi.org/10.5194/egusphere-2025-1788.
    Article in preprint stage.
    """
    seasons = ["DJF", "MAM", "JJA", "SON", "Year"]

    def compute_seasonal_stats(x):
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

        slopes = [ss_DJF.slope, ss_MAM.slope, ss_JJA.slope, ss_SON.slope, ss_an.slope]
        pvals = [ss_DJF.p, ss_MAM.p, ss_JJA.p, ss_SON.p, ss_an.p]

        return slopes, pvals

    if qsim is not None:
        slopes, pvals = compute_seasonal_stats(q)
        slopes_sim, pvals_sim = compute_seasonal_stats(qsim)
        slopes_np = np.array(slopes)
        slopes_sim_np = np.array(slopes_sim)
        ratio = slopes_np / slopes_sim_np
        ds = xarray.Dataset(
            data_vars={
                "Sen_slope_obs": ("season", slopes),
                "p_value_obs": ("season", pvals),
                "Sen_slope_sim": ("season", slopes_sim),
                "p_value_sim": ("season", pvals_sim),
                "ratio": ("season", ratio),
            },
            coords={"season": seasons},
        )

    else:
        slopes, pvals = compute_seasonal_stats(q)
        # Create labeled xarray
        ds = xarray.Dataset(
            data_vars={"Sen_slope": ("season", slopes), "p_value": ("season", pvals)}, coords={"season": seasons}
        )

    # Assign empty units to all variables
    for var in ds.data_vars:
        ds[var].attrs["units"] = ""

    return ds


@declare_units(q="[discharge]")
def annual_maxima(
    q: xarray.DataArray,
    freq: str = "YS-OCT",
) -> Dataset:
    """
    High flow temporal analysis: Finds annual maxima and corresponding Day Of Year (DOY) within a series of flows

    Parameters
    ----------
    q : array_like
        Daily streamflow vector.
    freq: str, optional
        Resampling frequency, here water year stating on the 1st of October

    Returns
    -------
    xarray.DataArray, [days]
        Peak streamflow per year and corresponding DOY

    Warnings
    --------
    The default `freq` is the water year used in the northern hemisphere, from October to September

    """
    # Find time of streamflow peak per year
    t_q_max = q.resample(time=freq).map(lambda x: x.idxmax()).dt.dayofyear
    t_q_max.attrs["units"] = "days"
    t_q_max.attrs["long_name"] = "day of year"

    # Find magnitude of streamflow peak per year
    peak_vals = q.resample(time=freq).max()
    peak_summary = xarray.Dataset(
        {
            "peak_flow": peak_vals,
            "peak_doy": t_q_max,
        }
    )

    return peak_summary


# @declare_units(q="[discharge]")
def fdc_slope(q: xr.DataArray) -> xr.DataArray:
    """
    Calculate the slope of the flow duration curve mid-section between the 33% and 66% exceedance probabilities.
    Aggregated analysis : Single value as a long-term benchmark

    Parameters
    ----------
    q : xarray.DataArray
        Daily streamflow data, expected to have a discharge unit.

    Returns
    -------
    xarray.DataArray
        Slope of the FDC between the 33% and 66% exceedance probabilities, unitless.

    Notes
    -----
    Single Returned numeric values are positive.

    High slope value :
    Steep slopes of the midsegment FDC are typically a characteristic behavior for watersheds having ‘‘flashy’’ responses
    (for exemple due to small soil storage capacity and hence larger percentage of overland flow).

    Lower slope value :
    Flatter slopes of the midsegment FDC are associated with watersheds having slower and
    more sustained groundwater low response.

    References
    ----------
    Vogel, R. M., & Fennessey, N. M. (1994). Flow-duration curves. I: New interpretation and confidence intervals.
    Journal of Water Resources Planning and Management, 120(4), 485-504.
    DOI:10.1061/(ASCE)0733-9496(1994)120:4(485)

    Yilmaz, K. K., Gupta, H. V., & Wagener, T. (2008). A process‐based diagnostic approach to model evaluation:
    Application to the NWS distributed hydrologic model. Water resources research, 44(9).
    DOI:10.1029/2007WR006716

    """
    # Calculate the 33rd and 66th percentiles directly across the 'time' dimension
    q33 = q.quantile(0.33, dim="time", skipna=True)
    q66 = q.quantile(0.66, dim="time", skipna=True)

    # Calculate the natural logarithm of the quantiles
    ln_q33 = np.log(q33)
    ln_q66 = np.log(q66)

    # Calculate the slope (unitless)
    slope = (ln_q33 - ln_q66) / (0.33 - 0.66)
    slope.attrs["units"] = " "
    slope.attrs["long_name"] = "Slope of FDC between 33% and 66% exceedance probabilities"
    return slope.rename("fdc_slope")

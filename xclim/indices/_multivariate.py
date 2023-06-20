# noqa: D100
from __future__ import annotations

from typing import Callable

import numpy as np
import xarray

from xclim.core.bootstrapping import percentile_bootstrap
from xclim.core.calendar import resample_doy
from xclim.core.units import (
    convert_units_to,
    declare_units,
    pint2cfunits,
    rate2amount,
    str2pint,
    to_agg_units,
)
from xclim.core.utils import Quantified

from . import run_length as rl
from ._conversion import rain_approximation, snowfall_approximation
from .generic import compare, select_resample_op, threshold_count

# Frequencies : YS: year start, QS-DEC: seasons starting in december, MS: month start
# See https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html

# -------------------------------------------------- #
# ATTENTION: ASSUME ALL INDICES WRONG UNTIL TESTED ! #
# -------------------------------------------------- #

__all__ = [
    "blowing_snow",
    "cold_and_dry_days",
    "cold_and_wet_days",
    "cold_spell_duration_index",
    "daily_temperature_range",
    "daily_temperature_range_variability",
    "days_over_precip_thresh",
    "extreme_temperature_range",
    "fraction_over_precip_thresh",
    "heat_wave_frequency",
    "heat_wave_max_length",
    "heat_wave_total_length",
    "high_precip_low_temp",
    "liquid_precip_ratio",
    "multiday_temperature_swing",
    "precip_accumulation",
    "precip_average",
    "rain_on_frozen_ground_days",
    "tg10p",
    "tg90p",
    "tn10p",
    "tn90p",
    "tx10p",
    "tx90p",
    "tx_tn_days_above",
    "warm_and_dry_days",
    "warm_and_wet_days",
    "warm_spell_duration_index",
    "winter_rain_ratio",
]


@declare_units(tasmin="[temperature]", tasmin_per="[temperature]")
@percentile_bootstrap
def cold_spell_duration_index(
    tasmin: xarray.DataArray,
    tasmin_per: xarray.DataArray,
    window: int = 6,
    freq: str = "YS",
    resample_before_rl: bool = True,
    bootstrap: bool = False,  # noqa  # noqa
    op: str = "<",
) -> xarray.DataArray:
    r"""Cold spell duration index.

    Number of days with at least `window` consecutive days when the daily minimum temperature is below the
    `tasmin_per` percentiles.

    Parameters
    ----------
    tasmin : xarray.DataArray
        Minimum daily temperature.
    tasmin_per : xarray.DataArray
        nth percentile of daily minimum temperature with `dayofyear` coordinate.
    window : int
        Minimum number of days with temperature below threshold to qualify as a cold spell.
    freq : str
      Resampling frequency.
    resample_before_rl : bool
      Determines if the resampling should take place before or after the run
      length encoding (or a similar algorithm) is applied to runs.
    bootstrap : bool
        Flag to run bootstrapping of percentiles. Used by percentile_bootstrap decorator.
        Bootstrapping is only useful when the percentiles are computed on a part of the studied sample.
        This period, common to percentiles and the sample must be bootstrapped to avoid inhomogeneities with
        the rest of the time series.
        Keep bootstrap to False when there is no common period, it would give wrong results
        plus, bootstrapping is computationally expensive.
    op : {"<", "<=", "lt", "le"}
        Comparison operation. Default: "<".

    Returns
    -------
    xarray.DataArray, [time]
        Count of days with at least six consecutive days when the daily minimum temperature is below the 10th
        percentile.

    Notes
    -----
    Let :math:`TN_i` be the minimum daily temperature for the day of the year :math:`i` and :math:`TN10_i` the 10th
    percentile of the minimum daily temperature over the 1961-1990 period for day of the year :math:`i`, the cold spell
    duration index over period :math:`\phi` is defined as:

    .. math::

       \sum_{i \in \phi} \prod_{j=i}^{i+6} \left[ TN_j < TN10_j \right]

    where :math:`[P]` is 1 if :math:`P` is true, and 0 if false.

    References
    ----------
    From the Expert Team on Climate Change Detection, Monitoring and Indices (ETCCDMI; :cite:p:`zhang_indices_2011`).

    Examples
    --------
    >>> from xclim.core.calendar import percentile_doy
    >>> from xclim.indices import cold_spell_duration_index
    >>> tasmin = xr.open_dataset(path_to_tasmin_file).tasmin.isel(lat=0, lon=0)
    >>> tn10 = percentile_doy(tasmin, per=10).sel(percentiles=10)
    >>> cold_spell_duration_index(tasmin, tn10)

    Note that this example does not use a proper 1961-1990 reference period.
    """
    tasmin_per = convert_units_to(tasmin_per, tasmin)

    # Create time series out of doy values.
    thresh = resample_doy(tasmin_per, tasmin)

    below = compare(tasmin, op, thresh, constrain=("<", "<="))
    out = rl.resample_and_rl(
        below,
        resample_before_rl,
        rl.windowed_run_count,
        window=window,
        freq=freq,
    )

    return to_agg_units(out, tasmin, "count")


@declare_units(
    tas="[temperature]",
    pr="[precipitation]",
    tas_per="[temperature]",
    pr_per="[precipitation]",
)
def cold_and_dry_days(
    tas: xarray.DataArray,
    pr: xarray.DataArray,
    tas_per: xarray.DataArray,
    pr_per: xarray.DataArray,
    freq: str = "YS",
) -> xarray.DataArray:
    r"""Cold and dry days.

    Returns the total number of days when "Cold" and "Dry" conditions coincide.

    Parameters
    ----------
    tas : xarray.DataArray
      Mean daily temperature values
    pr : xarray.DataArray
      Daily precipitation.
    tas_per : xarray.DataArray
      First quartile of daily mean temperature computed by month.
    pr_per : xarray.DataArray
      First quartile of daily total precipitation computed by month.
    freq : str
      Resampling frequency.

    Warnings
    --------
    Before computing the percentiles, all the precipitation below 1mm must be filtered out!
    Otherwise, the percentiles will include non-wet days.

    Returns
    -------
    xarray.DataArray
      The total number of days when cold and dry conditions coincide.

    Notes
    -----
    Bootstrapping is not available for quartiles because it would make no significant difference to bootstrap
    percentiles so far from the extremes.

    Formula to be written (:cite:t:`beniston_trends_2009`)

    References
    ----------
    :cite:cts:`beniston_trends_2009`

    """
    tas_per = convert_units_to(tas_per, tas)
    thresh = resample_doy(tas_per, tas)
    tg25 = tas < thresh

    pr_per = convert_units_to(pr_per, pr, context="hydro")
    thresh = resample_doy(pr_per, pr)
    pr25 = pr < thresh

    cold_and_dry = np.logical_and(tg25, pr25).resample(time=freq).sum(dim="time")
    return to_agg_units(cold_and_dry, tas, "count")


@declare_units(
    tas="[temperature]",
    pr="[precipitation]",
    tas_per="[temperature]",
    pr_per="[precipitation]",
)
def warm_and_dry_days(
    tas: xarray.DataArray,
    pr: xarray.DataArray,
    tas_per: xarray.DataArray,
    pr_per: xarray.DataArray,
    freq: str = "YS",
) -> xarray.DataArray:
    r"""Warm and dry days.

    Returns the total number of days when "warm" and "Dry" conditions coincide.

    Parameters
    ----------
    tas : xarray.DataArray
      Mean daily temperature values
    pr : xarray.DataArray
      Daily precipitation.
    tas_per : xarray.DataArray
      Third quartile of daily mean temperature computed by month.
    pr_per : xarray.DataArray
      First quartile of daily total precipitation computed by month.
    freq : str
      Resampling frequency.

    Warnings
    --------
    Before computing the percentiles, all the precipitation below 1mm must be filtered out!
    Otherwise, the percentiles will include non-wet days.

    Returns
    -------
    xarray.DataArray,
      The total number of days when warm and dry conditions coincide.

    Notes
    -----
    Bootstrapping is not available for quartiles because it would make no significant difference to bootstrap
    percentiles so far from the extremes.

    Formula to be written (:cite:t:`beniston_trends_2009`)

    References
    ----------
    :cite:cts:`beniston_trends_2009`

    """
    tas_per = convert_units_to(tas_per, tas)
    thresh = resample_doy(tas_per, tas)
    tg75 = tas > thresh

    pr_per = convert_units_to(pr_per, pr, context="hydro")
    thresh = resample_doy(pr_per, pr)
    pr25 = pr < thresh

    warm_and_dry = np.logical_and(tg75, pr25).resample(time=freq).sum(dim="time")
    return to_agg_units(warm_and_dry, tas, "count")


@declare_units(
    tas="[temperature]",
    pr="[precipitation]",
    tas_per="[temperature]",
    pr_per="[precipitation]",
)
def warm_and_wet_days(
    tas: xarray.DataArray,
    pr: xarray.DataArray,
    tas_per: xarray.DataArray,
    pr_per: xarray.DataArray,
    freq: str = "YS",
) -> xarray.DataArray:
    r"""Warm and wet days.

    Returns the total number of days when "warm" and "wet" conditions coincide.

    Parameters
    ----------
    tas : xarray.DataArray
      Mean daily temperature values
    pr : xarray.DataArray
      Daily precipitation.
    tas_per : xarray.DataArray
      Third quartile of daily mean temperature computed by month.
    pr_per : xarray.DataArray
      Third quartile of daily total precipitation computed by month.
    freq : str
      Resampling frequency.

    Warnings
    --------
    Before computing the percentiles, all the precipitation below 1mm must be filtered out!
    Otherwise, the percentiles will include non-wet days.

    Returns
    -------
    xarray.DataArray
      The total number of days when warm and wet conditions coincide.

    Notes
    -----
    Bootstrapping is not available for quartiles because it would make no significant difference
    to bootstrap percentiles so far from the extremes.

    Formula to be written (:cite:t:`beniston_trends_2009`)

    References
    ----------
    :cite:cts:`beniston_trends_2009`
    """
    tas_per = convert_units_to(tas_per, tas)
    thresh = resample_doy(tas_per, tas)
    tg75 = tas > thresh

    pr_per = convert_units_to(pr_per, pr, context="hydro")
    thresh = resample_doy(pr_per, pr)
    pr75 = pr > thresh

    warm_and_wet = np.logical_and(tg75, pr75).resample(time=freq).sum(dim="time")
    return to_agg_units(warm_and_wet, tas, "count")


@declare_units(
    tas="[temperature]",
    pr="[precipitation]",
    tas_per="[temperature]",
    pr_per="[precipitation]",
)
def cold_and_wet_days(
    tas: xarray.DataArray,
    pr: xarray.DataArray,
    tas_per: xarray.DataArray,
    pr_per: xarray.DataArray,
    freq: str = "YS",
) -> xarray.DataArray:
    r"""Cold and wet days.

    Returns the total number of days when "cold" and "wet" conditions coincide.

    Parameters
    ----------
    tas : xarray.DataArray
      Mean daily temperature values
    pr : xarray.DataArray
      Daily precipitation.
    tas_per : xarray.DataArray
      First quartile of daily mean temperature computed by month.
    pr_per : xarray.DataArray
      Third quartile of daily total precipitation computed by month.
    freq : str
      Resampling frequency.

    Warnings
    --------
    Before computing the percentiles, all the precipitation below 1mm must be filtered out!
    Otherwise, the percentiles will include non-wet days.

    Returns
    -------
    xarray.DataArray
      The total number of days when cold and wet conditions coincide.

    Notes
    -----
    Bootstrapping is not available for quartiles because it would make no significant
    difference to bootstrap percentiles so far from the extremes.

    Formula to be written (:cite:t:`beniston_trends_2009`)

    References
    ----------
    :cite:cts:`beniston_trends_2009`
    """
    tas_per = convert_units_to(tas_per, tas)
    thresh = resample_doy(tas_per, tas)
    tg25 = tas < thresh

    pr_per = convert_units_to(pr_per, pr, context="hydro")
    thresh = resample_doy(pr_per, pr)
    pr75 = pr > thresh

    cold_and_wet = np.logical_and(tg25, pr75).resample(time=freq).sum(dim="time")
    return to_agg_units(cold_and_wet, tas, "count")


@declare_units(
    tasmin="[temperature]",
    tasmax="[temperature]",
    thresh_tasmin="[temperature]",
    thresh_tasmax="[temperature]",
)
def multiday_temperature_swing(
    tasmin: xarray.DataArray,
    tasmax: xarray.DataArray,
    thresh_tasmin: Quantified = "0 degC",
    thresh_tasmax: Quantified = "0 degC",
    window: int = 1,
    op: str = "mean",
    op_tasmin: str = "<=",
    op_tasmax: str = ">",
    freq: str = "YS",
    resample_before_rl: bool = True,
) -> xarray.DataArray:
    r"""Statistics of consecutive diurnal temperature swing events.

    A diurnal swing of max and min temperature event is when Tmax > thresh_tasmax and Tmin <= thresh_tasmin. This indice
    finds all days that constitute these events and computes statistics over the length and frequency of these events.

    Parameters
    ----------
    tasmin : xarray.DataArray
      Minimum daily temperature.
    tasmax : xarray.DataArray
      Maximum daily temperature.
    thresh_tasmin : Quantified
      The temperature threshold needed to trigger a freeze event.
    thresh_tasmax : Quantified
      The temperature threshold needed to trigger a thaw event.
    window : int
      The minimal length of spells to be included in the statistics.
    op : {'mean', 'sum', 'max', 'min', 'std', 'count'}
      The statistical operation to use when reducing the list of spell lengths.
    op_tasmin : {"<", "<=", "lt", "le"}
      Comparison operation for tasmin. Default: "<=".
    op_tasmax : {">", ">=", "gt", "ge"}
      Comparison operation for tasmax. Default: ">".
    freq : str
      Resampling frequency.
    resample_before_rl : bool
      Determines if the resampling should take place before or after the run
      length encoding (or a similar algorithm) is applied to runs.

    Returns
    -------
    xarray.DataArray, [time]
      {freq} {op} length of diurnal temperature cycles exceeding thresholds.

    Notes
    -----
    Let :math:`TX_{i}` be the maximum temperature at day :math:`i` and :math:`TN_{i}` be the daily minimum temperature
    at day :math:`i`. Then freeze thaw spells during a given period are consecutive days where:

    .. math::

        TX_{i} > 0℃ \land TN_{i} <  0℃

    This indice returns a given statistic of the found lengths, optionally dropping those shorter than the `window`
    argument. For example, `window=1` and `op='sum'` returns the same value as :py:func:`daily_freezethaw_cycles`.
    """
    thaw_threshold = convert_units_to(thresh_tasmax, tasmax)
    freeze_threshold = convert_units_to(thresh_tasmin, tasmin)

    freeze = compare(tasmin, op_tasmin, freeze_threshold, constrain=("<", "<="))
    thaw = compare(tasmax, op_tasmax, thaw_threshold, constrain=(">", ">="))
    ft = freeze * thaw

    if op == "count":
        out = rl.resample_and_rl(
            ft,
            resample_before_rl,
            rl.windowed_run_events,
            window=window,
            freq=freq,
        )
    else:
        out = rl.resample_and_rl(
            ft,
            resample_before_rl,
            rl.rle_statistics,
            reducer=op,
            window=window,
            freq=freq,
        )

    return to_agg_units(out, tasmin, "count")


@declare_units(tasmax="[temperature]", tasmin="[temperature]")
def daily_temperature_range(
    tasmin: xarray.DataArray,
    tasmax: xarray.DataArray,
    freq: str = "YS",
    op: str | Callable = "mean",
) -> xarray.DataArray:
    r"""Statistics of daily temperature range.

    The mean difference between the daily maximum temperature and the daily minimum temperature.

    Parameters
    ----------
    tasmin : xarray.DataArray
      Minimum daily temperature.
    tasmax : xarray.DataArray
      Maximum daily temperature.
    freq : str
      Resampling frequency.
    op : {'min', 'max', 'mean', 'std'} or func
      Reduce operation. Can either be a DataArray method or a function that can be applied to a DataArray.

    Returns
    -------
    xarray.DataArray, [same units as tasmin]
      The average variation in daily temperature range for the given time period.

    Notes
    -----
    For a default calculation using `op='mean'` :

    Let :math:`TX_{ij}` and :math:`TN_{ij}` be the daily maximum and minimum temperature at day :math:`i` of period
    :math:`j`. Then the mean diurnal temperature range in period :math:`j` is:

    .. math::

        DTR_j = \frac{ \sum_{i=1}^I (TX_{ij} - TN_{ij}) }{I}
    """
    tasmax = convert_units_to(tasmax, tasmin)
    dtr = tasmax - tasmin
    out = select_resample_op(dtr, op=op, freq=freq)
    u = str2pint(tasmax.units)
    out.attrs["units"] = pint2cfunits(u - u)
    return out


@declare_units(tasmax="[temperature]", tasmin="[temperature]")
def daily_temperature_range_variability(
    tasmin: xarray.DataArray, tasmax: xarray.DataArray, freq: str = "YS"
) -> xarray.DataArray:
    r"""Mean absolute day-to-day variation in daily temperature range.

    Mean absolute day-to-day variation in daily temperature range.

    Parameters
    ----------
    tasmin : xarray.DataArray
      Minimum daily temperature.
    tasmax : xarray.DataArray
      Maximum daily temperature.
    freq : str
      Resampling frequency.

    Returns
    -------
    xarray.DataArray, [same units as tasmin]
      The average day-to-day variation in daily temperature range for the given time period.

    Notes
    -----
    Let :math:`TX_{ij}` and :math:`TN_{ij}` be the daily maximum and minimum temperature at day :math:`i` of period
    :math:`j`. Then calculated is the absolute day-to-day differences in period :math:`j` is:

    .. math::

       vDTR_j = \frac{ \sum_{i=2}^{I} |(TX_{ij}-TN_{ij})-(TX_{i-1,j}-TN_{i-1,j})| }{I}
    """
    tasmax = convert_units_to(tasmax, tasmin)
    vdtr = abs((tasmax - tasmin).diff(dim="time"))
    out = vdtr.resample(time=freq).mean(dim="time")
    u = str2pint(tasmax.units)
    out.attrs["units"] = pint2cfunits(u - u)
    return out


@declare_units(tasmax="[temperature]", tasmin="[temperature]")
def extreme_temperature_range(
    tasmin: xarray.DataArray, tasmax: xarray.DataArray, freq: str = "YS"
) -> xarray.DataArray:
    r"""Extreme intra-period temperature range.

    The maximum of max temperature (TXx) minus the minimum of min temperature (TNn) for the given time period.

    Parameters
    ----------
    tasmin : xarray.DataArray
      Minimum daily temperature.
    tasmax : xarray.DataArray
      Maximum daily temperature.
    freq : str
      Resampling frequency.

    Returns
    -------
    xarray.DataArray, [same units as tasmin]
      Extreme intra-period temperature range for the given time period.

    Notes
    -----
    Let :math:`TX_{ij}` and :math:`TN_{ij}` be the daily maximum and minimum temperature at day :math:`i` of period
    :math:`j`. Then the extreme temperature range in period :math:`j` is:

    .. math::

        ETR_j = max(TX_{ij}) - min(TN_{ij})
    """
    tasmax = convert_units_to(tasmax, tasmin)
    tx_max = tasmax.resample(time=freq).max(dim="time")
    tn_min = tasmin.resample(time=freq).min(dim="time")

    out = tx_max - tn_min
    u = str2pint(tasmax.units)
    out.attrs["units"] = pint2cfunits(u - u)
    return out


@declare_units(
    tasmin="[temperature]",
    tasmax="[temperature]",
    thresh_tasmin="[temperature]",
    thresh_tasmax="[temperature]",
)
def heat_wave_frequency(
    tasmin: xarray.DataArray,
    tasmax: xarray.DataArray,
    thresh_tasmin: Quantified = "22.0 degC",
    thresh_tasmax: Quantified = "30 degC",
    window: int = 3,
    freq: str = "YS",
    op: str = ">",
    resample_before_rl: bool = True,
) -> xarray.DataArray:
    r"""Heat wave frequency.

    Number of heat waves over a given period. A heat wave is defined as an event where the minimum and maximum daily
    temperature both exceed specific thresholds over a minimum number of days.

    Parameters
    ----------
    tasmin : xarray.DataArray
      Minimum daily temperature.
    tasmax : xarray.DataArray
      Maximum daily temperature.
    thresh_tasmin : Quantified
      The minimum temperature threshold needed to trigger a heatwave event.
    thresh_tasmax : Quantified
      The maximum temperature threshold needed to trigger a heatwave event.
    window:  int
      Minimum number of days with temperatures above thresholds to qualify as a heatwave.
    freq : str
      Resampling frequency.
    op: {">", ">=", "gt", "ge"}
      Comparison operation. Default: ">".
    resample_before_rl : bool
      Determines if the resampling should take place before or after the run
      length encoding (or a similar algorithm) is applied to runs.

    Returns
    -------
    xarray.DataArray, [dimensionless]
      Number of heatwave at the requested frequency.

    Notes
    -----
    The thresholds of 22° and 25°C for night temperatures and 30° and 35°C for day temperatures were selected by
    Health Canada professionals, following a temperature–mortality analysis. These absolute temperature thresholds
    characterize the occurrence of hot weather events that can result in adverse health outcomes for Canadian
    communities :cite:p:`casati_regional_2013`.

    In :cite:t:`robinson_definition_2001`, the parameters would be `thresh_tasmin=27.22, thresh_tasmax=39.44, window=2` (81F, 103F).

    References
    ----------
    :cite:cts:`casati_regional_2013,robinson_definition_2001`
    """
    thresh_tasmax = convert_units_to(thresh_tasmax, tasmax)
    thresh_tasmin = convert_units_to(thresh_tasmin, tasmin)

    constrain = (">", ">=")
    cond = (compare(tasmin, op, thresh_tasmin, constrain)) & (
        compare(tasmax, op, thresh_tasmax, constrain)
    )

    out = rl.resample_and_rl(
        cond,
        resample_before_rl,
        rl.windowed_run_events,
        window=window,
        freq=freq,
    )
    out.attrs["units"] = ""
    return out


@declare_units(
    tasmin="[temperature]",
    tasmax="[temperature]",
    thresh_tasmin="[temperature]",
    thresh_tasmax="[temperature]",
)
def heat_wave_max_length(
    tasmin: xarray.DataArray,
    tasmax: xarray.DataArray,
    thresh_tasmin: Quantified = "22.0 degC",
    thresh_tasmax: Quantified = "30 degC",
    window: int = 3,
    freq: str = "YS",
    op: str = ">",
    resample_before_rl: bool = True,
) -> xarray.DataArray:
    r"""Heat wave max length.

    Maximum length of heat waves over a given period. A heat wave is defined as an event where the minimum and maximum
    daily temperature both exceeds specific thresholds over a minimum number of days.

    By definition heat_wave_max_length must be >= window.

    Parameters
    ----------
    tasmin : xarray.DataArray
      Minimum daily temperature.
    tasmax : xarray.DataArray
      Maximum daily temperature.
    thresh_tasmin : Quantified
      The minimum temperature threshold needed to trigger a heatwave event.
    thresh_tasmax : Quantified
      The maximum temperature threshold needed to trigger a heatwave event.
    window : int
      Minimum number of days with temperatures above thresholds to qualify as a heatwave.
    freq : str
      Resampling frequency.
    op : {">", ">=", "gt", "ge"}
      Comparison operation. Default: ">".
    resample_before_rl : bool
      Determines if the resampling should take place before or after the run
      length encoding (or a similar algorithm) is applied to runs.

    Returns
    -------
    xarray.DataArray, [time]
      Maximum length of heatwave at the requested frequency.

    Notes
    -----
    The thresholds of 22° and 25°C for night temperatures and 30° and 35°C for day temperatures were selected by
    Health Canada professionals, following a temperature–mortality analysis. These absolute temperature thresholds
    characterize the occurrence of hot weather events that can result in adverse health outcomes for Canadian
    communities :cite:p:`casati_regional_2013`.

    In :cite:t:`robinson_definition_2001`, the parameters would be:
    `thresh_tasmin=27.22, thresh_tasmax=39.44, window=2` (81F, 103F).

    References
    ----------
    :cite:cts:`casati_regional_2013,robinson_definition_2001`
    """
    thresh_tasmax = convert_units_to(thresh_tasmax, tasmax)
    thresh_tasmin = convert_units_to(thresh_tasmin, tasmin)

    constrain = (">", ">=")
    cond = (compare(tasmin, op, thresh_tasmin, constrain)) & (
        compare(tasmax, op, thresh_tasmax, constrain)
    )
    out = rl.resample_and_rl(
        cond,
        resample_before_rl,
        rl.rle_statistics,
        reducer="max",
        window=window,
        freq=freq,
    )
    return to_agg_units(out, tasmax, "count")


@declare_units(
    tasmin="[temperature]",
    tasmax="[temperature]",
    thresh_tasmin="[temperature]",
    thresh_tasmax="[temperature]",
)
def heat_wave_total_length(
    tasmin: xarray.DataArray,
    tasmax: xarray.DataArray,
    thresh_tasmin: Quantified = "22.0 degC",
    thresh_tasmax: Quantified = "30 degC",
    window: int = 3,
    freq: str = "YS",
    op: str = ">",
    resample_before_rl: bool = True,
) -> xarray.DataArray:
    r"""Heat wave total length.

    Total length of heat waves over a given period. A heat wave is defined as an event where the minimum and maximum
    daily temperature both exceeds specific thresholds over a minimum number of days.
    This the sum of all days in such events.

    Parameters
    ----------
    tasmin : xarray.DataArray
      Minimum daily temperature.
    tasmax : xarray.DataArray
      Maximum daily temperature.
    thresh_tasmin : str
      The minimum temperature threshold needed to trigger a heatwave event.
    thresh_tasmax : str
      The maximum temperature threshold needed to trigger a heatwave event.
    window : int
      Minimum number of days with temperatures above thresholds to qualify as a heatwave.
    freq : str
      Resampling frequency.
    op: {">", ">=", "gt", "ge"}
      Comparison operation. Default: ">".
    resample_before_rl : bool
      Determines if the resampling should take place before or after the run
      length encoding (or a similar algorithm) is applied to runs.

    Returns
    -------
    xarray.DataArray, [time]
      Total length of heatwave at the requested frequency.

    Notes
    -----
    See notes and references of `heat_wave_max_length`
    """
    thresh_tasmax = convert_units_to(thresh_tasmax, tasmax)
    thresh_tasmin = convert_units_to(thresh_tasmin, tasmin)

    constrain = (">", ">=")
    cond = compare(tasmin, op, thresh_tasmin, constrain) & compare(
        tasmax, op, thresh_tasmax, constrain
    )
    out = rl.resample_and_rl(
        cond,
        resample_before_rl,
        rl.windowed_run_count,
        window=window,
        freq=freq,
    )

    return to_agg_units(out, tasmin, "count")


@declare_units(
    pr="[precipitation]",
    prsn="[precipitation]",
    tas="[temperature]",
    thresh="[temperature]",
)
def liquid_precip_ratio(
    pr: xarray.DataArray,
    prsn: xarray.DataArray | None = None,
    tas: xarray.DataArray | None = None,
    thresh: Quantified = "0 degC",
    freq: str = "QS-DEC",
) -> xarray.DataArray:
    r"""Ratio of rainfall to total precipitation.

    The ratio of total liquid precipitation over the total precipitation. If solid precipitation is not provided,
    it is approximated with pr, tas and thresh, using the `snowfall_approximation` function with method 'binary'.

    Parameters
    ----------
    pr : xarray.DataArray
      Mean daily precipitation flux.
    prsn : xarray.DataArray, optional
      Mean daily solid precipitation flux.
    tas : xarray.DataArray, optional
      Mean daily temperature.
    thresh : Quantified
      Threshold temperature under which precipitation is assumed to be solid.
    freq : str
      Resampling frequency.

    Returns
    -------
    xarray.DataArray, [dimensionless]
      Ratio of rainfall to total precipitation.

    Notes
    -----
    Let :math:`PR_i` be the mean daily precipitation of day :math:`i`, then for a period :math:`j` starting at
    day :math:`a` and finishing on day :math:`b`:

    .. math::

        PR_{ij} = \sum_{i=a}^{b} PR_i

        PRwet_{ij}

    See Also
    --------
    winter_rain_ratio
    """
    if prsn is None and tas is not None:
        prsn = snowfall_approximation(pr, tas=tas, thresh=thresh, method="binary")
    elif prsn is None:
        raise KeyError("prsn or tas must be supplied.")

    tot = pr.resample(time=freq).sum(dim="time")
    rain = tot - prsn.resample(time=freq).sum(dim="time")
    ratio = rain / tot
    ratio.attrs["units"] = ""
    return ratio


@declare_units(pr="[precipitation]", tas="[temperature]", thresh="[temperature]")
def precip_accumulation(
    pr: xarray.DataArray,
    tas: xarray.DataArray = None,
    phase: str | None = None,
    thresh: Quantified = "0 degC",
    freq: str = "YS",
) -> xarray.DataArray:
    r"""Accumulated total (liquid and/or solid) precipitation.

    Resample the original daily mean precipitation flux and accumulate over each period.
    If a daily temperature is provided, the `phase` keyword can be used to sum precipitation of a given phase only.
    When the temperature is under the given threshold, precipitation is assumed to be snow, and liquid rain otherwise.
    This indice is agnostic to the type of daily temperature (tas, tasmax or tasmin) given.

    Parameters
    ----------
    pr : xarray.DataArray
      Mean daily precipitation flux.
    tas : xarray.DataArray, optional
      Mean, maximum or minimum daily temperature.
    phase : {None, 'liquid', 'solid'}
      Which phase to consider, "liquid" or "solid", if None (default), both are considered.
    thresh : Quantified
      Threshold of `tas` over which the precipication is assumed to be liquid rain.
    freq : str
      Resampling frequency.

    Returns
    -------
    xarray.DataArray, [length]
      The total daily precipitation at the given time frequency for the given phase.

    Notes
    -----
    Let :math:`PR_i` be the mean daily precipitation of day :math:`i`, then for a period :math:`j` starting at
    day :math:`a` and finishing on day :math:`b`:

    .. math::

       PR_{ij} = \sum_{i=a}^{b} PR_i

    If tas and phase are given, the corresponding phase precipitation is estimated before computing the accumulation,
    using one of `snowfall_approximation` or `rain_approximation` with the `binary` method.

    Examples
    --------
    The following would compute, for each grid cell of a dataset, the total
    precipitation at the seasonal frequency, ie DJF, MAM, JJA, SON, DJF, etc.:

    >>> from xclim.indices import precip_accumulation
    >>> pr_day = xr.open_dataset(path_to_pr_file).pr
    >>> prcp_tot_seasonal = precip_accumulation(pr_day, freq="QS-DEC")
    """
    if phase == "liquid":
        pr = rain_approximation(pr, tas=tas, thresh=thresh, method="binary")
    elif phase == "solid":
        pr = snowfall_approximation(pr, tas=tas, thresh=thresh, method="binary")
    pram = rate2amount(pr)
    return pram.resample(time=freq).sum(dim="time").assign_attrs(units=pram.units)


@declare_units(pr="[precipitation]", tas="[temperature]", thresh="[temperature]")
def precip_average(
    pr: xarray.DataArray,
    tas: xarray.DataArray = None,
    phase: str | None = None,
    thresh: Quantified = "0 degC",
    freq: str = "YS",
) -> xarray.DataArray:
    r"""Averaged (liquid and/or solid) precipitation.

    Resample the original daily mean precipitation flux and average over each period.
    If a daily temperature is provided, the `phase` keyword can be used to average precipitation of a given phase only.
    When the temperature is under the given threshold, precipitation is assumed to be snow, and liquid rain otherwise.
    This indice is agnostic to the type of daily temperature (tas, tasmax or tasmin) given.

    Parameters
    ----------
    pr : xarray.DataArray
      Mean daily precipitation flux.
    tas : xarray.DataArray, optional
      Mean, maximum or minimum daily temperature.
    phase : {None, 'liquid', 'solid'}
      Which phase to consider, "liquid" or "solid", if None (default), both are considered.
    thresh : Quantified
      Threshold of `tas` over which the precipication is assumed to be liquid rain.
    freq : str
      Resampling frequency.

    Returns
    -------
    xarray.DataArray, [length]
      The averaged daily precipitation at the given time frequency for the given phase.

    Notes
    -----
    Let :math:`PR_i` be the mean daily precipitation of day :math:`i`, then for a period :math:`j` starting at
    day :math:`a` and finishing on day :math:`b`:

    .. math::

       PR_{ij} =\frac{ \sum_{i=a}^{b} PR_i }{b - a + 1}

    If tas and phase are given, the corresponding phase precipitation is estimated before computing the accumulation,
    using one of `snowfall_approximation` or `rain_approximation` with the `binary` method.

    Examples
    --------
    The following would compute, for each grid cell of a dataset, the total
    precipitation at the seasonal frequency, ie DJF, MAM, JJA, SON, DJF, etc.:

    >>> from xclim.indices import precip_average
    >>> pr_day = xr.open_dataset(path_to_pr_file).pr
    >>> prcp_tot_seasonal = precip_average(pr_day, freq="QS-DEC")
    """
    if phase == "liquid":
        pr = rain_approximation(pr, tas=tas, thresh=thresh, method="binary")
    elif phase == "solid":
        pr = snowfall_approximation(pr, tas=tas, thresh=thresh, method="binary")
    pram = rate2amount(pr)
    return pram.resample(time=freq).mean(dim="time").assign_attrs(units=pram.units)


# FIXME: Resample after run length?
@declare_units(pr="[precipitation]", tas="[temperature]", thresh="[precipitation]")
def rain_on_frozen_ground_days(
    pr: xarray.DataArray,
    tas: xarray.DataArray,
    thresh: Quantified = "1 mm/d",
    freq: str = "YS",
) -> xarray.DataArray:
    """Number of rain on frozen ground events.

    Number of days with rain above a threshold after a series of seven days below freezing temperature.
    Precipitation is assumed to be rain when the temperature is above 0℃.

    Parameters
    ----------
    pr : xarray.DataArray
      Mean daily precipitation flux.
    tas : xarray.DataArray
      Mean daily temperature.
    thresh : Quantified
      Precipitation threshold to consider a day as a rain event.
    freq : str
      Resampling frequency.

    Returns
    -------
    xarray.DataArray, [time]
      The number of rain on frozen ground events per period.

    Notes
    -----
    Let :math:`PR_i` be the mean daily precipitation and :math:`TG_i` be the mean daily temperature of day :math:`i`.
    Then for a period :math:`j`, rain on frozen grounds days are counted where:

    .. math::

        PR_{i} > Threshold [mm]

    and where

    .. math::

        TG_{i} ≤ 0℃

    is true for continuous periods where :math:`i ≥ 7`
    """
    t = convert_units_to(thresh, pr, context="hydro")
    frz = convert_units_to("0 C", tas)

    def func(x, axis):
        """Check that temperature conditions are below 0 for seven days and above after."""
        frozen = x == np.array([0, 0, 0, 0, 0, 0, 0, 1], bool)
        return frozen.all(axis=axis)

    tcond = (tas > frz).rolling(time=8).reduce(func)
    pcond = pr > t

    out = (tcond * pcond * 1).resample(time=freq).sum(dim="time")
    return to_agg_units(out, tas, "count")


@declare_units(
    pr="[precipitation]",
    tas="[temperature]",
    pr_thresh="[precipitation]",
    tas_thresh="[temperature]",
)
def high_precip_low_temp(
    pr: xarray.DataArray,
    tas: xarray.DataArray,
    pr_thresh: Quantified = "0.4 mm/d",
    tas_thresh: Quantified = "-0.2 degC",
    freq: str = "YS",
) -> xarray.DataArray:
    """Number of days with precipitation above threshold and temperature below threshold.

    Number of days when precipitation is greater or equal to some threshold, and temperatures are colder than some
    threshold. This can be used for example to identify days with the potential for freezing rain or icing conditions.

    Parameters
    ----------
    pr : xarray.DataArray
      Mean daily precipitation flux.
    tas : xarray.DataArray
      Daily mean, minimum or maximum temperature.
    pr_thresh : Quantified
      Precipitation threshold to exceed.
    tas_thresh : Quantified
      Temperature threshold not to exceed.
    freq : str
      Resampling frequency.

    Returns
    -------
    xarray.DataArray, [time]
      Count of days with high precipitation and low temperatures.

    Example
    -------
    To compute the number of days with intense rainfall while minimum temperatures dip below -0.2C:
    >>> pr = xr.open_dataset(path_to_pr_file).pr
    >>> tasmin = xr.open_dataset(path_to_tasmin_file).tasmin
    >>> high_precip_low_temp(
    ...     pr, tas=tasmin, pr_thresh="10 mm/d", tas_thresh="-0.2 degC"
    ... )
    """
    pr_thresh = convert_units_to(pr_thresh, pr, context="hydro")
    tas_thresh = convert_units_to(tas_thresh, tas)

    cond = (pr >= pr_thresh) * (tas < tas_thresh) * 1
    out = cond.resample(time=freq).sum(dim="time")
    return to_agg_units(out, pr, "count")


@declare_units(pr="[precipitation]", pr_per="[precipitation]", thresh="[precipitation]")
@percentile_bootstrap
def days_over_precip_thresh(
    pr: xarray.DataArray,
    pr_per: xarray.DataArray,
    thresh: Quantified = "1 mm/day",
    freq: str = "YS",
    bootstrap: bool = False,  # noqa
    op: str = ">",
) -> xarray.DataArray:
    r"""Number of wet days with daily precipitation over a given percentile.

    Number of days over period where the precipitation is above a threshold defining wet days and above a given
    percentile for that day.

    Parameters
    ----------
    pr : xarray.DataArray
      Mean daily precipitation flux.
    pr_per : xarray.DataArray
      Percentile of wet day precipitation flux. Either computed daily (one value per day
      of year) or computed over a period (one value per spatial point).
    thresh : Quantified
       Precipitation value over which a day is considered wet.
    freq : str
      Resampling frequency.
    bootstrap : bool
      Flag to run bootstrapping of percentiles. Used by percentile_bootstrap decorator.
      Bootstrapping is only useful when the percentiles are computed on a part of the studied sample.
      This period, common to percentiles and the sample must be bootstrapped to avoid inhomogeneities with
      the rest of the time series.
      Keep bootstrap to False when there is no common period, it would give wrong results
      plus, bootstrapping is computationally expensive.
    op: {">", ">=", "gt", "ge"}
      Comparison operation. Default: ">".

    Returns
    -------
    xarray.DataArray, [time]
      Count of days with daily precipitation above the given percentile [days].

    Examples
    --------
    >>> from xclim.indices import days_over_precip_thresh
    >>> pr = xr.open_dataset(path_to_pr_file).pr
    >>> p75 = pr.quantile(0.75, dim="time", keep_attrs=True)
    >>> r75p = days_over_precip_thresh(pr, p75)
    """
    pr_per = convert_units_to(pr_per, pr, context="hydro")
    thresh = convert_units_to(thresh, pr, context="hydro")

    tp = pr_per.where(pr_per > thresh, thresh)
    if "dayofyear" in pr_per.coords:
        # Create time series out of doy values.
        tp = resample_doy(tp, pr)

    # Compute the days when precip is both over the wet day threshold and the percentile threshold.
    out = threshold_count(pr, op, tp, freq, constrain=(">", ">="))
    return to_agg_units(out, pr, "count")


@declare_units(pr="[precipitation]", pr_per="[precipitation]", thresh="[precipitation]")
@percentile_bootstrap
def fraction_over_precip_thresh(
    pr: xarray.DataArray,
    pr_per: xarray.DataArray,
    thresh: Quantified = "1 mm/day",
    freq: str = "YS",
    bootstrap: bool = False,  # noqa
    op: str = ">",
) -> xarray.DataArray:
    r"""Fraction of precipitation due to wet days with daily precipitation over a given percentile.

    Percentage of the total precipitation over period occurring in days when the precipitation is above a threshold
    defining wet days and above a given percentile for that day.

    Parameters
    ----------
    pr : xarray.DataArray
      Mean daily precipitation flux.
    pr_per : xarray.DataArray
      Percentile of wet day precipitation flux. Either computed daily (one value per day
      of year) or computed over a period (one value per spatial point).
    thresh : Quantified
       Precipitation value over which a day is considered wet.
    freq : str
      Resampling frequency.
    bootstrap : bool
      Flag to run bootstrapping of percentiles. Used by percentile_bootstrap decorator.
      Bootstrapping is only useful when the percentiles are computed on a part of the studied sample.
      This period, common to percentiles and the sample must be bootstrapped to avoid inhomogeneities with
      the rest of the time series.
      Keep bootstrap to False when there is no common period, it would give wrong results
      plus, bootstrapping is computationally expensive.
    op: {">", ">=", "gt", "ge"}
      Comparison operation. Default: ">".

    Returns
    -------
    xarray.DataArray, [dimensionless]
      Fraction of precipitation over threshold during wet days.

    """
    pr_per = convert_units_to(pr_per, pr, context="hydro")
    thresh = convert_units_to(thresh, pr, context="hydro")

    tp = pr_per.where(pr_per > thresh, thresh)
    if "dayofyear" in pr_per.coords:
        # Create time series out of doy values.
        tp = resample_doy(tp, pr)

    constrain = (">", ">=")
    # Total precip during wet days over period
    total = (
        pr.where(compare(pr, op, thresh, constrain), 0)
        .resample(time=freq)
        .sum(dim="time")
    )

    # Compute the days when precip is both over the wet day threshold and the percentile threshold.
    over = (
        pr.where(compare(pr, op, tp, constrain), 0).resample(time=freq).sum(dim="time")
    )

    out = over / total
    out.attrs["units"] = ""
    return out


@declare_units(tas="[temperature]", tas_per="[temperature]")
@percentile_bootstrap
def tg90p(
    tas: xarray.DataArray,
    tas_per: xarray.DataArray,
    freq: str = "YS",
    bootstrap: bool = False,  # noqa
    op: str = ">",
) -> xarray.DataArray:
    r"""Number of days with daily mean temperature over the 90th percentile.

    Number of days with daily mean temperature over the 90th percentile.

    Parameters
    ----------
    tas : xarray.DataArray
      Mean daily temperature.
    tas_per : xarray.DataArray
      90th percentile of daily mean temperature.
    freq : str
      Resampling frequency.
    bootstrap : bool
      Flag to run bootstrapping of percentiles. Used by percentile_bootstrap decorator.
      Bootstrapping is only useful when the percentiles are computed on a part of the studied sample.
      This period, common to percentiles and the sample must be bootstrapped to avoid inhomogeneities with
      the rest of the time series.
      Keep bootstrap to False when there is no common period, it would give wrong results
      plus, bootstrapping is computationally expensive.
    op: {">", ">=", "gt", "ge"}
      Comparison operation. Default: ">".

    Returns
    -------
    xarray.DataArray, [time]
      Count of days with daily mean temperature below the 10th percentile [days].

    Notes
    -----
    The 90th percentile should be computed for a 5-day window centered on each calendar day for a reference period.

    Examples
    --------
    >>> from xclim.core.calendar import percentile_doy
    >>> from xclim.indices import tg90p
    >>> tas = xr.open_dataset(path_to_tas_file).tas
    >>> tas_per = percentile_doy(tas, per=90).sel(percentiles=90)
    >>> hot_days = tg90p(tas, tas_per)
    """
    tas_per = convert_units_to(tas_per, tas)

    # Create time series out of doy values.
    thresh = resample_doy(tas_per, tas)

    # Identify the days over the 90th percentile
    out = threshold_count(tas, op, thresh, freq, constrain=(">", ">="))
    return to_agg_units(out, tas, "count")


@declare_units(tas="[temperature]", tas_per="[temperature]")
@percentile_bootstrap
def tg10p(
    tas: xarray.DataArray,
    tas_per: xarray.DataArray,
    freq: str = "YS",
    bootstrap: bool = False,  # noqa
    op: str = "<",
) -> xarray.DataArray:
    r"""Number of days with daily mean temperature below the 10th percentile.

    Number of days with daily mean temperature below the 10th percentile.

    Parameters
    ----------
    tas : xarray.DataArray
      Mean daily temperature.
    tas_per : xarray.DataArray
      10th percentile of daily mean temperature.
    freq : str
      Resampling frequency.
    bootstrap : bool
      Flag to run bootstrapping of percentiles. Used by percentile_bootstrap decorator.
      Bootstrapping is only useful when the percentiles are computed on a part of the studied sample.
      This period, common to percentiles and the sample must be bootstrapped to avoid inhomogeneities with
      the rest of the time series.
      Keep bootstrap to False when there is no common period, it would give wrong results
      plus, bootstrapping is computationally expensive.
    op: {"<", "<=", "lt", "le"}
      Comparison operation. Default: "<".

    Returns
    -------
    xarray.DataArray, [time]
      Count of days with daily mean temperature below the 10th percentile [days].

    Notes
    -----
    The 10th percentile should be computed for a 5-day window centered on each calendar day for a reference period.

    Examples
    --------
    >>> from xclim.core.calendar import percentile_doy
    >>> from xclim.indices import tg10p
    >>> tas = xr.open_dataset(path_to_tas_file).tas
    >>> tas_per = percentile_doy(tas, per=10).sel(percentiles=10)
    >>> cold_days = tg10p(tas, tas_per)
    """
    tas_per = convert_units_to(tas_per, tas)

    # Create time series out of doy values.
    thresh = resample_doy(tas_per, tas)

    # Identify the days below the 10th percentile
    out = threshold_count(tas, op, thresh, freq, constrain=("<", "<="))
    return to_agg_units(out, tas, "count")


@declare_units(tasmin="[temperature]", tasmin_per="[temperature]")
@percentile_bootstrap
def tn90p(
    tasmin: xarray.DataArray,
    tasmin_per: xarray.DataArray,
    freq: str = "YS",
    bootstrap: bool = False,  # noqa
    op: str = ">",
) -> xarray.DataArray:
    r"""Number of days with daily minimum temperature over the 90th percentile.

    Number of days with daily minimum temperature over the 90th percentile.

    Parameters
    ----------
    tasmin : xarray.DataArray
      Minimum daily temperature.
    tasmin_per : xarray.DataArray
      90th percentile of daily minimum temperature.
    freq : str
      Resampling frequency.
    bootstrap : bool
      Flag to run bootstrapping of percentiles. Used by percentile_bootstrap decorator.
      Bootstrapping is only useful when the percentiles are computed on a part of the studied sample.
      This period, common to percentiles and the sample must be bootstrapped to avoid inhomogeneities with
      the rest of the time series.
      Keep bootstrap to False when there is no common period, it would give wrong results
      plus, bootstrapping is computationally expensive.
    op: {">", ">=", "gt", "ge"}
      Comparison operation. Default: ">".

    Returns
    -------
    xarray.DataArray, [time]
      Count of days with daily minimum temperature below the 10th percentile [days].

    Notes
    -----
    The 90th percentile should be computed for a 5-day window centered on each calendar day for a reference period.

    Examples
    --------
    >>> from xclim.core.calendar import percentile_doy
    >>> from xclim.indices import tn90p
    >>> tas = xr.open_dataset(path_to_tas_file).tas
    >>> tas_per = percentile_doy(tas, per=90).sel(percentiles=90)
    >>> hot_days = tn90p(tas, tas_per)
    """
    tasmin_per = convert_units_to(tasmin_per, tasmin)

    # Create time series out of doy values.
    thresh = resample_doy(tasmin_per, tasmin)

    # Identify the days with min temp above 90th percentile.
    out = threshold_count(tasmin, op, thresh, freq, constrain=(">", ">="))
    return to_agg_units(out, tasmin, "count")


@declare_units(tasmin="[temperature]", tasmin_per="[temperature]")
@percentile_bootstrap
def tn10p(
    tasmin: xarray.DataArray,
    tasmin_per: xarray.DataArray,
    freq: str = "YS",
    bootstrap: bool = False,  # noqa
    op: str = "<",
) -> xarray.DataArray:
    r"""Number of days with daily minimum temperature below the 10th percentile.

    Number of days with daily minimum temperature below the 10th percentile.

    Parameters
    ----------
    tasmin : xarray.DataArray
      Mean daily temperature.
    tasmin_per : xarray.DataArray
      10th percentile of daily minimum temperature.
    freq : str
      Resampling frequency.
    bootstrap : bool
      Flag to run bootstrapping of percentiles. Used by percentile_bootstrap decorator.
      Bootstrapping is only useful when the percentiles are computed on a part of the studied sample.
      This period, common to percentiles and the sample must be bootstrapped to avoid inhomogeneities with
      the rest of the time series.
      Keep bootstrap to False when there is no common period, it would give wrong results
      plus, bootstrapping is computationally expensive.
    op: {"<", "<=", "lt", "le"}
      Comparison operation. Default: "<".

    Returns
    -------
    xarray.DataArray, [time]
      Count of days with daily minimum temperature below the 10th percentile [days].

    Notes
    -----
    The 10th percentile should be computed for a 5-day window centered on each calendar day for a reference period.

    Examples
    --------
    >>> from xclim.core.calendar import percentile_doy
    >>> from xclim.indices import tn10p
    >>> tas = xr.open_dataset(path_to_tas_file).tas
    >>> tas_per = percentile_doy(tas, per=10).sel(percentiles=10)
    >>> cold_days = tn10p(tas, tas_per)
    """
    tasmin_per = convert_units_to(tasmin_per, tasmin)

    # Create time series out of doy values.
    thresh = resample_doy(tasmin_per, tasmin)

    # Identify the days below the 10th percentile
    out = threshold_count(tasmin, op, thresh, freq, constrain=("<", "<="))
    return to_agg_units(out, tasmin, "count")


@declare_units(tasmax="[temperature]", tasmax_per="[temperature]")
@percentile_bootstrap
def tx90p(
    tasmax: xarray.DataArray,
    tasmax_per: xarray.DataArray,
    freq: str = "YS",
    bootstrap: bool = False,  # noqa
    op: str = ">",
) -> xarray.DataArray:
    r"""Number of days with daily maximum temperature over the 90th percentile.

    Number of days with daily maximum temperature over the 90th percentile.

    Parameters
    ----------
    tasmax : xarray.DataArray
      Maximum daily temperature.
    tasmax_per : xarray.DataArray
      90th percentile of daily maximum temperature.
    freq : str
      Resampling frequency.
    bootstrap : bool
      Flag to run bootstrapping of percentiles. Used by percentile_bootstrap decorator.
      Bootstrapping is only useful when the percentiles are computed on a part of the studied sample.
      This period, common to percentiles and the sample must be bootstrapped to avoid inhomogeneities with
      the rest of the time series.
      Keep bootstrap to False when there is no common period, it would give wrong results
      plus, bootstrapping is computationally expensive.
    op: {">", ">=", "gt", "ge"}
      Comparison operation. Default: ">".

    Returns
    -------
    xarray.DataArray, [time]
      Count of days with daily maximum temperature below the 10th percentile [days].

    Notes
    -----
    The 90th percentile should be computed for a 5-day window centered on each calendar day for a reference period.

    Examples
    --------
    >>> from xclim.core.calendar import percentile_doy
    >>> from xclim.indices import tx90p
    >>> tas = xr.open_dataset(path_to_tas_file).tas
    >>> tasmax_per = percentile_doy(tas, per=90).sel(percentiles=90)
    >>> hot_days = tx90p(tas, tasmax_per)
    """
    tasmax_per = convert_units_to(tasmax_per, tasmax)

    # Create time series out of doy values.
    thresh = resample_doy(tasmax_per, tasmax)

    # Identify the days with max temp above 90th percentile.
    out = threshold_count(tasmax, op, thresh, freq, constrain=(">", ">="))
    return to_agg_units(out, tasmax, "count")


@declare_units(tasmax="[temperature]", tasmax_per="[temperature]")
@percentile_bootstrap
def tx10p(
    tasmax: xarray.DataArray,
    tasmax_per: xarray.DataArray,
    freq: str = "YS",
    bootstrap: bool = False,  # noqa
    op: str = "<",
) -> xarray.DataArray:
    r"""Number of days with daily maximum temperature below the 10th percentile.

    Number of days with daily maximum temperature below the 10th percentile.

    Parameters
    ----------
    tasmax : xarray.DataArray
      Maximum daily temperature.
    tasmax_per : xarray.DataArray
      10th percentile of daily maximum temperature.
    freq : str
      Resampling frequency.
    bootstrap : bool
      Flag to run bootstrapping of percentiles. Used by percentile_bootstrap decorator.
      Bootstrapping is only useful when the percentiles are computed on a part of the studied sample.
      This period, common to percentiles and the sample must be bootstrapped to avoid inhomogeneities with
      the rest of the time series.
      Keep bootstrap to False when there is no common period, it would give wrong results
      plus, bootstrapping is computationally expensive.
    op: {"<", "<=", "lt", "le"}
      Comparison operation. Default: "<".

    Returns
    -------
    xarray.DataArray, [time]
      Count of days with daily maximum temperature below the 10th percentile [days].

    Notes
    -----
    The 10th percentile should be computed for a 5-day window centered on each calendar day for a reference period.

    Examples
    --------
    >>> from xclim.core.calendar import percentile_doy
    >>> from xclim.indices import tx10p
    >>> tas = xr.open_dataset(path_to_tas_file).tas
    >>> tasmax_per = percentile_doy(tas, per=10).sel(percentiles=10)
    >>> cold_days = tx10p(tas, tasmax_per)
    """
    tasmax_per = convert_units_to(tasmax_per, tasmax)

    # Create time series out of doy values.
    thresh = resample_doy(tasmax_per, tasmax)

    # Identify the days below the 10th percentile
    out = threshold_count(tasmax, op, thresh, freq, constrain=("<", "<="))
    return to_agg_units(out, tasmax, "count")


@declare_units(
    tasmin="[temperature]",
    tasmax="[temperature]",
    thresh_tasmin="[temperature]",
    thresh_tasmax="[temperature]",
)
def tx_tn_days_above(
    tasmin: xarray.DataArray,
    tasmax: xarray.DataArray,
    thresh_tasmin: Quantified = "22 degC",
    thresh_tasmax: Quantified = "30 degC",
    freq: str = "YS",
    op: str = ">",
) -> xarray.DataArray:
    r"""Number of days with both hot maximum and minimum daily temperatures.

    The number of days per period with tasmin above a threshold and tasmax above another threshold.

    Parameters
    ----------
    tasmin : xarray.DataArray
      Minimum daily temperature.
    tasmax : xarray.DataArray
      Maximum daily temperature.
    thresh_tasmin : Quantified
      Threshold temperature for tasmin on which to base evaluation.
    thresh_tasmax : Quantified
      Threshold temperature for tasmax on which to base evaluation.
    freq : str
      Resampling frequency.
    op: {">", ">=", "gt", "ge"}
      Comparison operation. Default: ">".

    Returns
    -------
    xarray.DataArray, [time]
      the number of days with tasmin > thresh_tasmin and tasmax > thresh_tasmax per period.

    Notes
    -----
    Let :math:`TX_{ij}` be the maximum temperature at day :math:`i` of period :math:`j`, :math:`TN_{ij}`
    the daily minimum temperature at day :math:`i` of period :math:`j`, :math:`TX_{thresh}` the threshold for maximum
    daily temperature, and :math:`TN_{thresh}` the threshold for minimum daily temperature. Then counted is the number
    of days where:

    .. math::

        TX_{ij} > TX_{thresh} [℃]

    and where:

    .. math::

        TN_{ij} > TN_{thresh} [℃]

    """
    thresh_tasmax = convert_units_to(thresh_tasmax, tasmax)
    thresh_tasmin = convert_units_to(thresh_tasmin, tasmin)

    constrain = (">", ">=")
    events = (
        compare(tasmin, op, thresh_tasmin, constrain)
        & compare(tasmax, op, thresh_tasmax, constrain)
    ) * 1
    out = events.resample(time=freq).sum(dim="time")
    return to_agg_units(out, tasmin, "count")


@declare_units(tasmax="[temperature]", tasmax_per="[temperature]")
@percentile_bootstrap
def warm_spell_duration_index(
    tasmax: xarray.DataArray,
    tasmax_per: xarray.DataArray,
    window: int = 6,
    freq: str = "YS",
    resample_before_rl: bool = True,
    bootstrap: bool = False,  # noqa
    op: str = ">",
) -> xarray.DataArray:
    r"""Warm spell duration index.

    Number of days inside spells of a minimum number of consecutive days when the daily maximum temperature is above the
    90th percentile. The 90th percentile should be computed for a 5-day moving window, centered on each calendar day in
    the 1961-1990 period.

    Parameters
    ----------
    tasmax : xarray.DataArray
      Maximum daily temperature.
    tasmax_per : xarray.DataArray
      percentile(s) of daily maximum temperature.
    window : int
      Minimum number of days with temperature above threshold to qualify as a warm spell.
    freq : str
      Resampling frequency.
    resample_before_rl : bool
      Determines if the resampling should take place before or after the run
      length encoding (or a similar algorithm) is applied to runs.
    bootstrap : bool
      Flag to run bootstrapping of percentiles. Used by percentile_bootstrap decorator.
      Bootstrapping is only useful when the percentiles are computed on a part of the studied sample.
      This period, common to percentiles and the sample must be bootstrapped to avoid inhomogeneities with
      the rest of the time series.
      Keep bootstrap to False when there is no common period, it would give wrong results
      plus, bootstrapping is computationally expensive.
    op: {">", ">=", "gt", "ge"}
      Comparison operation. Default: ">".

    Returns
    -------
    xarray.DataArray, [time]
      Warm spell duration index.

    References
    ----------
    From the Expert Team on Climate Change Detection, Monitoring and Indices (ETCCDMI; :cite:p:`zhang_indices_2011`).
    Used in :cite:cts:`alexander_global_2006`

    Examples
    --------
    Note that this example does not use a proper 1961-1990 reference period.

    >>> from xclim.core.calendar import percentile_doy
    >>> from xclim.indices import warm_spell_duration_index

    >>> tasmax = xr.open_dataset(path_to_tasmax_file).tasmax.isel(lat=0, lon=0)
    >>> tasmax_per = percentile_doy(tasmax, per=90).sel(percentiles=90)
    >>> warm_spell_duration_index(tasmax, tasmax_per)
    """
    thresh = convert_units_to(tasmax_per, tasmax)

    # Create time series out of doy values.
    thresh = resample_doy(thresh, tasmax)

    above = compare(tasmax, op, thresh, constrain=(">", ">="))
    out = rl.resample_and_rl(
        above,
        resample_before_rl,
        rl.windowed_run_count,
        window=window,
        freq=freq,
    )

    return to_agg_units(out, tasmax, "count")


@declare_units(pr="[precipitation]", prsn="[precipitation]", tas="[temperature]")
def winter_rain_ratio(
    *,
    pr: xarray.DataArray,
    prsn: xarray.DataArray = None,
    tas: xarray.DataArray = None,
    freq: str = "QS-DEC",
) -> xarray.DataArray:
    """Ratio of rainfall to total precipitation during winter.

    The ratio of total liquid precipitation over the total precipitation over the winter months (DJF). If solid
    precipitation is not provided, then precipitation is assumed solid if the temperature is below 0°C.

    Parameters
    ----------
    pr : xarray.DataArray
      Mean daily precipitation flux.
    prsn : xarray.DataArray, optional
      Mean daily solid precipitation flux.
    tas : xarray.DataArray, optional
      Mean daily temperature.
    freq : str
      Resampling frequency.

    Returns
    -------
    xarray.DataArray
      Ratio of rainfall to total precipitation during winter months (DJF).
    """
    ratio = liquid_precip_ratio(pr, prsn, tas, freq=freq)
    winter = ratio.indexes["time"].month == 12
    return ratio.sel(time=winter)


@declare_units(
    snd="[length]", sfcWind="[speed]", snd_thresh="[length]", sfcWind_thresh="[speed]"
)
def blowing_snow(
    snd: xarray.DataArray,
    sfcWind: xarray.DataArray,  # noqa
    snd_thresh: Quantified = "5 cm",
    sfcWind_thresh: Quantified = "15 km/h",  # noqa
    window: int = 3,
    freq: str = "AS-JUL",
) -> xarray.DataArray:
    """Blowing snow days.

    Number of days when both snowfall over the last days and daily wind speeds are above respective thresholds.

    Parameters
    ----------
    snd : xarray.DataArray
      Surface snow depth.
    sfcWind : xr.DataArray
      Wind velocity
    snd_thresh : Quantified
      Threshold on net snowfall accumulation over the last `window` days.
    sfcWind_thresh : Quantified
      Wind speed threshold.
    window : int
      Period over which snow is accumulated before comparing against threshold.
    freq : str
      Resampling frequency.

    Returns
    -------
    xarray.DataArray
      Number of days when snowfall and wind speeds are above respective thresholds.
    """
    snd_thresh = convert_units_to(snd_thresh, snd)
    sfcWind_thresh = convert_units_to(sfcWind_thresh, sfcWind)  # noqa

    # Net snow accumulation over the last `window` days
    snow = snd.diff(dim="time").rolling(time=window, center=False).sum()

    # Blowing snow conditions
    cond = (snow >= snd_thresh) * (sfcWind >= sfcWind_thresh) * 1

    out = cond.resample(time=freq).sum(dim="time")
    out.attrs["units"] = to_agg_units(out, snd, "count")
    return out

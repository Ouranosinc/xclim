# -*- coding: utf-8 -*-

"""
Indices module
"""
import logging
from warnings import warn

import numpy as np
import xarray as xr

from . import run_length as rl

logging.basicConfig(level=logging.DEBUG)
logging.captureWarnings(True)

xr.set_options(enable_cftimeindex=True)  # Set xarray to use cftimeindex

# Frequencies : YS: year start, QS-DEC: seasons starting in december, MS: month start
# See http://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases
K2C = 273.15
ftomm = np.nan


# TODO: Define a unit conversion system for temperature [K, C, F] and precipitation [mm h-1, Kg m-2 s-1] metrics
# TODO: Move utility functions to another file.
# TODO: Should we reference the standard vocabulary we're using ?
# E.g. http://vocab.nerc.ac.uk/collection/P07/current/BHMHISG2/


# -------------------------------------------------- #
# ATTENTION: ASSUME ALL INDICES WRONG UNTIL TESTED ! #
# -------------------------------------------------- #


def base_flow_index(q, freq='YS'):
    r"""Base flow index

    Return the base flow index, defined as the minimum 7-day average flow divided by the mean flow.

    Parameters
    ----------
    q : xarray.DataArray
      Rate of river discharge [m³/s]
    freq : str, optional
      Resampling frequency

    Returns
    -------
    xarray.DataArrray
      Base flow index.
    """
    m7 = q.rolling(time=7, center=True).mean(dim='time').resample(time=freq)
    mq = q.resample(time=freq)

    m7m = m7.min(dim='time')
    return m7m / mq.mean(dim='time')


def cold_spell_duration_index(tasmin, tn10, freq='YS'):
    r"""Cold spell duration index

    Resamples the daily minimum temperature series by returning the number of days per
    period where the temperature is below the calendar day 10th percentile (calculated
    over a centered 5-day window for values during a 30-year reference period) for a
    minimum of at least six consecutive days.

    Parameters
    ----------
    tasmin : xarray.DataArray
      Minimum daily temperature values [C] or [K]
    tn10 : xarray.DataArray
      The daily climatological 10th percentile (using a centered 5-day window) of minimum daily temperature for
      a 30-year reference period.
    freq : str, optional
      Resampling frequency

    Returns
    -------
    xarray.DataArray
      Cold spell duration index.

    Note
    ----
    # TODO: Add a formula or example to better illustrate the cold spell duration metric

    See also
    --------
    percentile_doy
    """
    window = 6

    return tasmin.pipe(lambda x: x - tn10) \
        .resample(time=freq) \
        .apply(rl.windowed_run_count_ufunc, window=window)


def cold_spell_index(tas, thresh=-10, window=5, freq='AS-JUL'):
    r"""Cold spell index

    Days that are part of a cold spell (5 or more consecutive days with Tavg < -10°C)
    """
    over = tas < K2C + thresh
    group = over.resample(time=freq)

    return group.apply(rl.windowed_run_count_ufunc, window=window)


def cold_and_dry_days(tas, tgin25, pr, wet25, freq='YS'):
    r"""Cold and dry days.

    Returns the total number of days where "Cold" and "Dry" conditions coincide.

    Parameters
    ----------
    tas : xarray.DataArray
      Minimum daily temperature values [℃] or [K]
    pr : xarray.DataArray
    tgin25 : unknown
    wet25: unknown
    freq : str, optional
      Resampling frequency

    Note
    ----
    See Beniston (2009) for more details. https://doi.org/10.1029/2008GL037119

    """
    c1 = tas < tgin25
    c2 = (pr > 1 * ftomm) * (pr < wet25)

    c = (c1 * c2) * 1
    return c.resample(time=freq).sum(dim='time')


def maximum_consecutive_dry_days(pr, thresh=1, freq='YS'):
    r"""Maximum number of consecutive dry days

    Return the maximum number of consecutive days within the period where precipitation
    is below a certain threshold.

    Parameters
    ----------
    pr : xarray.DataArray
      Mean daily precipitation flux [mm]
    thresh : float
      Threshold precipitation on which to base evaluation [mm]
    freq : str, optional
      Resampling frequency

    Returns
    -------
    xarray.DataArray
      The maximum number of consecutive dry days.

    """
    group = (pr < thresh).resample(time=freq)
    return group.apply(rl.longest_run_ufunc)


def consecutive_frost_days(tasmin, freq='AS-JUL'):
    r"""Maximum number of consecutive frost days (Tmin < 0℃).

    Resample the daily minimum temperature series by computing the maximum number
    of days below the freezing point over each period.

    Parameters
    ----------
    tasmin : xarray.DataArray
      Minimum daily temperature values [℃] or [K]
    freq : str, optional
      Resampling frequency

    Returns
    -------
    xarray.DataArray
      The maximum number of consecutive days below the freezing point.

    Note
    ----
    Let :math:`Tmin_i` be the minimum daily temperature of day `i`, then for a period `p` starting at
    day `a` and finishing on day `b`

    .. math::

       CFD_p = max(run_l(Tmin_i < 273.15))

    for :math:`a <= i <= b`

    where run_l returns the length of each consecutive series of true values.

    """
    group = (tasmin < K2C).resample(time=freq)
    return group.apply(rl.longest_run_ufunc)


def maximum_consecutive_wet_days(pr, thresh=1.0, freq='YS'):
    r"""Consecutive wet days.

    Returns the maximum number of consecutive wet days.

    Parameters
    ---------
    pr : xarray.DataArray
      Mean daily precipitation flux [Kg m-2 s-1] or [mm]
    thresh : float
      Threshold precipitation on which to base evaluation [Kg m-2 s-1] or [mm]
    freq : str, optional
      Resampling frequency
    """
    group = (pr > thresh).resample(time=freq)
    return group.apply(rl.longest_run_ufunc)


def cooling_degree_days(tas, thresh=18, freq='YS'):
    r"""Cooling degree days above threshold."""

    return tas.pipe(lambda x: x - thresh - K2C) \
        .clip(min=0) \
        .resample(time=freq) \
        .sum(dim='time')


def daily_freezethaw_cycles(tasmax, tasmin, freq='YS'):
    r"""Number of freeze-thaw cycle days.

    The number of days where Tmax > 0℃ and Tmin < 0℃.
    """

    ft = (tasmin < K2C) * (tasmax > K2C) * 1
    return ft.resample(time=freq).sum(dim='time')


def daily_temperature_range(tasmax, tasmin, freq='YS'):
    r"""Mean of daily temperature range.

    Parameters
    ----------
    tasmax : xarray.DataArray
      Maximum daily temperature values [℃] or [K]
    tasmin : xarray.DataArray
      Minimum daily temperature values [℃] or [K]
    freq : str, optional
      Resampling frequency

    Returns
    -------
    xarray.DataArray
      The average variation in daily temperature range for the given time period.

    Note
    ----
    Let :math:`TX_{ij}` and :math:`TN_{ij}` be the daily maximum and minimum temperature at day :math:`{i}`
    of period :math:`{j}`. Then the mean diurnal temperature range in period :math:`{j}` is:

    .. math::

        DTR_j = \frac{ \sum_{i=1}^I (TX_{ij} - TN_{ij}) }{I}

    """

    dtr = tasmax - tasmin
    return dtr.resample(time=freq).mean(dim='time')


def daily_temperature_range_variability(tasmax, tasmin, freq="YS"):
    r"""Mean absolute day-to-day variation in daily temperature range.

    Parameters
    ----------
    tasmax : xarray.DataArray
      Maximum daily temperature values [℃] or [K]
    tasmin : xarray.DataArray
      Minimum daily temperature values [℃] or [K]
    freq : str, optional
      Resampling frequency

    Returns
    -------
    xarray.DataArray
      The average day-to-day variation in daily temperature range for the given time period.

    Note
    ----
    Let :math:`TX_{ij}` and :math:`TN_{ij}` be the daily maximum and minimum temperature at
    day :math:`i` of period :math:`j`. Then calculated is the absolute day-to-day differences in
    period :math:`j` is:

    .. math::

       vDTR_j = \frac{ \sum_{i=2}^{I} |(TX_{ij}-TN_{ij})-(TX_{i-1,j}-TN_{i-1,j})| }{I}

    """

    vdtr = abs((tasmax - tasmin).diff(dim='time'))
    return vdtr.resample(time=freq).mean(dim='time')


def freshet_start(tas, thresh=0.0, window=5, freq='YS'):
    r"""First day consistently exceeding threshold temperature.

    Returns first day of year when a temperature threshold is exceeded
    over a given number of days.

    Parameters
    ----------
    tas : xarray.DataArray
      Mean daily temperature [℃] or [K]
    thresh : float
      Threshold temperature on which to base evaluation [℃] or [K]
    window : int
      Minimum number of days with temperature above threshold needed for evaluation
    freq : str, optional
      Resampling frequency

    """

    i = xr.DataArray(np.arange(tas.time.size), dims='time')
    ind = xr.broadcast(i, tas)[0]

    over = ((tas > K2C + thresh) * 1).rolling(time=window).sum(dim='time')
    i = ind.where(over == window)
    return i.resample(time=freq).min(dim='time')


def frost_days(tasmin, freq='YS'):
    r"""Frost days index

    Number of days where daily minimum temperatures are below 0℃.

    Parameters
    ----------
    tasmin : xarray.DataArray
      Minimum daily temperature [℃] or [K]
    freq : str, optional
      Resampling frequency

    Returns
    -------
    xarray.DataArray
      Frost days index.
    """
    f = (tasmin < K2C) * 1
    return f.resample(time=freq).sum(dim='time')


def growing_degree_days(tas, thresh=4.0, freq='YS'):
    r"""Growing degree-days over 4℃.

    The sum of degree-days over 4℃.

    Parameters
    ---------
    tas : xarray.DataArray
      Mean daily temperature [℃] or [K[
    thresh : float
      Threshold temperature on which to base evaluation [℃] or [K]. Default: 4℃.
    freq : str, optional
      Resampling frequency

    Returns
    -------
    xarray.DataArray
      The sum of growing degree-days above 4℃

    Note
    ----



    """
    return tas.pipe(lambda x: x - thresh - K2C) \
        .clip(min=0) \
        .resample(time=freq) \
        .sum(dim='time')


def growing_season_length(tas, thresh=5.0, window=6, freq='YS'):
    r"""Growing season length.

    The number of days between the first occurrence of at least
    six consecutive days with mean daily temperature over 5℃ and
    the first occurrence of at least six consecutive days with
    mean daily temperature below 5℃ after July 1st in the northern
    hemisphere and January 1st in the southern hemisphere.

    Parameters
    ---------
    tas : xarray.DataArray
      Mean daily temperature [℃] or [K[
    thresh : float
      Threshold temperature on which to base evaluation [℃] or [K]. Default: 5℃.
    window : int
      Minimum number of days with temperature above threshold to mark the beginning and end of growing season.
    freq : str, optional
      Resampling frequency

    Returns
    -------
    xarray.DataArray
      Growing season length
    """
    i = xr.DataArray(np.arange(tas.time.size), dims='time')
    ind = xr.broadcast(i, tas)[0]

    c = ((tas > thresh + K2C) * 1).rolling(time=window).sum(dim='time')
    i1 = ind.where(c == window).resample(time=freq).min(dim='time')

    # Resample sets the time to T00:00.
    i11 = i1.reindex_like(c, method='ffill')

    # TODO: Adjust for southern hemisphere
    i2 = ind.where(c == 0).where(tas.time.dt.month >= 7)
    d = i2 - i11

    return d.resample(time=freq).max(dim='time')


def heat_wave_index(tasmax, thresh=25.0, window=5, freq='YS'):
    r"""Heat wave index.

    Number of days that are part of a heatwave, defined as five or more consecutive days over 25℃.

    Parameters
    ----------
    tasmax : xarrray.DataArray
      Maximum daily temperature [℃] or [K[
    thresh : float
      Threshold temperature on which to designate a heatwave [℃] or [K]. Default: 25℃.
    window : int
      Minimum number of days with temperature above threshold to qualify as a heatwave.
    freq : str, optional
      Resampling frequency

    Returns
    -------
    DataArray
      Heat wave index.
    """
    # TODO: Deal better with boundary effects.
    # TODO: Deal with attributes

    over = tasmax > K2C + thresh
    group = over.resample(time=freq)

    return group.apply(rl.windowed_run_count_ufunc, window=window)


def heating_degree_days(tas, freq='YS', thresh=17.0):
    r"""Heating degree days

    Sum of daily positive values for a temperature threshold minus mean daily temperature.

    Parameters
    ----------
    tas : xarray.DataArray
      Mean daily temperature [℃] or [K]
    thresh : float
      Threshold temperature on which to base evaluation [℃] or [K]. Default: 17℃.
    freq : str, optional
      Resampling frequency

    Returns
    -------
    xarray.DataArray
      Heating degree days index.
    """
    return tas.pipe(lambda x: thresh - x) \
        .clip(0) \
        .resample(time=freq) \
        .sum(dim='time')


def hot_days(tasmax, thresh=30.0, freq='YS'):
    r"""Number of very hot days.

    Number of days with max temperature exceeding a base threshold.

    Parameters
    ----------
    tasmax : xarrray.DataArray
      Maximum daily temperature [℃] or [K]
    thresh : float
      Threshold temperature on which to base evaluation [℃] or [K]. Default: 30℃.
    freq : str, optional
      Resampling frequency

    Returns
    -------
    xarray.DataArray
      Number of hot days.
    """
    hd = (tasmax > K2C + thresh) * 1
    return hd.resample(time=freq).sum(dim='time')


def ice_days(tasmax, freq='YS'):
    r"""Number of freezing days

    Number of days where daily maximum temperatures are below 0℃.

    Parameters
    ----------
    tasmax : xarrray.DataArray
      Maximum daily temperature [℃] or [K]
    freq : str, optional
      Resampling frequency

    Returns
    -------
    xarray.DataArray
      Number of freezing days.
    """
    f = (tasmax < K2C) * 1
    return f.resample(time=freq).sum(dim='time')


def summer_days(tasmax, thresh=25.0, freq='YS'):
    r"""Number of summer days

    Number of days where daily maximum temperature exceeds 25℃.

    Parameters
    ----------
    tasmax : xarray.DataArray
      Maximum daily temperature [℃] or [K]
    thresh : float
      Threshold temperature on which to base evaluation [℃] or [K]. Default: 25℃
    freq : str, optional
      Resampling frequency

    Returns
    -------
    xarray.DataArray
      Number of summer days.
    """
    f = (tasmax > thresh + K2C) * 1
    return f.resample(time=freq).sum(dim='time')


def tg_mean(tas, freq='YS'):
    r"""Mean of daily average temperature.

    Resample the original daily mean temperature series by taking the mean over each period.

    Parameters
    ----------
    tas : xarray.DataArray
      Mean daily temperature [℃] or [K[
    freq : str, optional
      Resampling frequency

    Returns
    -------
    xarray.DataArray
      The mean daily temperature at the given time frequency


    Note
    ----
    Let :math:`T_i` be the mean daily temperature of day `i`, then for a period `p` starting at
    day `a` and finishing on day `b`

    .. math::

       TG_p = \frac{\sum_{i=a}^{b} T_i}{b - a + 1}


    Examples
    --------
    The following would compute for each grid cell of file `tas.day.nc` the mean temperature
    at the seasonal frequency, ie DJF, MAM, JJA, SON, DJF, etc.

    >>> t = xr.open_dataset('tas.day.nc')
    >>> tg = tg_mean(t, freq="QS-DEC")

    """
    arr = tas.resample(time=freq) if freq else tas
    return arr.mean(dim='time')


def tn10p(tasmin, p10, freq='YS'):
    """Days with daily minimum temperature below the 10th percentile of the reference period.

    Parameters
    ----------
    tasmin : xarray.DataArray
      Minimum daily temperature [℃] or [K]
    p10 : float
    freq : str, optional
      Resampling frequency

    """
    return (tasmin.groupby('time.dayofyear') < p10).resample(time=freq).sum(dim='time')


def tn_max(tasmin, freq='YS'):
    """Highest minimum temperature

    Maximum of daily minimum temperature.

    Parameters
    ----------
    tasmin : xarray.DataArray
      Minimum daily temperature [℃] or [K]
    freq : str, optional
      Resampling frequency
    """
    return tasmin.resample(time=freq).max(dim='time')


def tn_mean(tasmin, freq='YS'):
    """Mean minimum temperature

    Mean of daily minimum temperature.

    Parameters
    ----------
    tasmin : xarray.DataArray
      Minimum daily temperature [℃] or [K]
    freq : str, optional
      Resampling frequency
    """
    arr = tasmin.resample(time=freq) if freq else tasmin
    return arr.mean(dim='time')


def tn_min(tasmin, freq='YS'):
    """Lowest minimum temperature

    Minimum of daily minimum temperature.

    Parameters
    ----------
    tasmin : xarray.DataArray
      Minimum daily temperature [℃] or [K]
    freq : str, optional
      Resampling frequency
    """
    return tasmin.resample(time=freq).min(dim='time')


def max_n_day_precipitation_amount(da, window, freq='YS'):
    """Highest precipitation amount cumulated over a n-day moving window.

    Calculate the n-day rolling sum of the original daily total precipitation series
    and determine the maximum value over each period.

    Parameters
    ----------
    da : xarray.DataArray
      Daily precipitation values.
    window : int
      Window size in days.
    freq : str, optional
      Resampling frequency : default 'YS' (yearly)

    Returns
    -------
    xarray.DataArray
      The highest cumulated n-day precipitation value at the given time frequency.


    Examples
    --------
    The following would compute for each grid cell of file `pr.day.nc` the highest 5-day total precipitation
    at an annual frequency.

    >>> da = xr.open_dataset('pr.day.nc').pr
    >>> window = 5
    >>> output = max_n_day_precipitation_amount(da, window, freq="YS")

    """

    # rolling sum of the values
    arr = da.rolling(time=window, center=False).sum(dim='time')
    return arr.resample(time=freq).max(dim='time')


def max_1day_precipitation_amount(pr, freq='YS'):
    """Highest 1-day precipitation amount for a period (frequency).

    Resample the original daily total precipitation temperature series by taking the max over each period.

    Parameters
    ----------
    pr : xarray.DataArray
      Daily precipitation values.
    freq : str, optional
      Resampling frequency one of : 'YS' (yearly) ,'M' (monthly), or 'QS-DEC' (seasonal - quarters starting in december)


    Returns
    -------
    xarray.DataArray
      The highest 1-day precipitation value at the given time frequency.


    Examples
    --------
    The following would compute for each grid cell of file `pr.day.nc` the highest 1-day total
    at an annual frequency.

    >>> pr = xr.open_dataset('pr.day.nc').pr
    >>> rx1day = max_1day_precipitation_amount(pr, freq="YS")

    """
    return pr.resample(time=freq).max(dim='time')


def prcp_tot(pr, freq='YS', units='kg m-2 s-1'):
    r"""Accumulated total (liquid + solid) precipitation.

    Resample the original daily mean precipitation flux and accumulate over each period.

    Parameters
    ----------
    pr : xarray.DataArray
      Mean daily precipitation flux [Kg m-2 s-1] or [mm].
    freq : str, optional
      Resampling frequency as defined in
      http://pandas.pydata.org/pandas-docs/stable/timeseries.html#resampling.
    units: str, optional
      Units of the precipitation data. Must be within ['kg m-2 s-2', 'mm']

    Returns
    -------
    xarray.DataArray
      The total daily precipitation at the given time frequency in [mm].

    Note
    ----

    Let :math:`pr_i` be the mean daily precipitation of day `i`, then for a period `p` starting at
    day `a` and finishing on day `b`

    .. math::
       out_p = \sum_{i=a}^{b} pr_i

    Examples
    --------
    The following would compute for each grid cell of file `pr_day.nc` the total
    precipitation at the seasonal frequency, ie DJF, MAM, JJA, SON, DJF, etc.

    >>> pr_day = xr.open_dataset('pr_day.nc').pr
    >>> prcp_tot_seasonal = prcp_tot(pr_day, freq="QS-DEC")

    """

    # TODO deal with the time_boundaries

    # resample the precipitation to the wanted frequency
    arr = pr.resample(time=freq)
    # cumulate the values over the season
    output = arr.sum(dim='time')
    # unit conversion as needed
    if units == 'kg m-2 s-1':
        # convert from km m-2 s-1 to mm day-1
        e = 'units converted from [kg m-2 s-1] to [mm day-1]'
        warn(e)
        output *= 86400  # number of sec in 24h
    elif units == 'mm':
        # nothing to do
        pass
    else:
        raise RuntimeError('non-conforming units')
    return output


def tropical_nights(tasmin, thresh=20.0, freq='YS'):
    r"""Tropical nights

    Number of days with minimum daily temperature above threshold.

    Parameters
    ----------
    tasmin : xarray.DataArray
      Minimum daily temperature [℃] or [K]
    thresh : float
      Threshold temperature on which to base evaluation [℃] or [K]. Default: 20℃.
    freq : str, optional
      Resampling frequency
    """

    return tasmin.pipe(lambda x: (tasmin > thresh + K2C) * 1) \
        .resample(time=freq) \
        .sum(dim='time')


def tx_max(tasmax, freq='YS'):
    r"""Highest max temperature

    Maximum of daily maximum temperature.

    Parameters
    ----------
    tasmax : xarray.DataArray
      Maximum daily temperature [℃] or [K]
    freq : str, optional
      Resampling frequency
    """

    return tasmax.resample(time=freq).max(dim='time')


def tx_mean(tasmax, freq='YS'):
    r"""Mean max temperature

    Mean of daily maximum temperature.

    Parameters
    ----------
    tasmax : xarray.DataArray
      Maximum daily temperature [℃] or [K]
    freq : str, optional
      Resampling frequency
    """

    arr = tasmax.resample(time=freq) if freq else tasmax
    return arr.mean(dim='time')


def tx_min(tasmax, freq='YS'):
    """Lowest max temperature

    Minimum of daily maximum temperature.

    Parameters
    ----------
    tasmax : xarray.DataArray
      Maximum daily temperature [℃] or [K]
    freq : str, optional
      Resampling frequency
    """

    return tasmax.resample(time=freq).min(dim='time')


def percentile_doy(arr, window=5, per=.1):
    """Percentile day of year

    Returns the climatological percentile over a moving window
    around the day of the year.

    Parameters
    ----------
    arr : xarray.DataArray
    window : int
    per : float
    """

    # TODO: Support percentile array, store percentile in attributes.
    rr = arr.rolling(1, center=True, time=window).construct('window')

    # Create empty percentile array
    g = rr.groupby('time.dayofyear')
    c = g.count(dim=('time', 'window'))

    p = xr.full_like(c, np.nan).astype(float).load()

    for doy, ind in rr.groupby('time.dayofyear'):
        p.loc[{'dayofyear': doy}] = ind.compute().quantile(per, dim=('time', 'window'))

    return p


def wet_days(pr, thresh=1.0, freq='YS'):
    r"""Wet days

    Return the total number of days during period with precipitation over threshold.

    Parameters
    ----------
    pr : xarray.DataArray
      Daily precipitation [mm]
    thresh : float
      Precipitation value over which a day is considered wet. Default: 1mm.
    freq : str, optional
      Resampling frequency defining the periods
      defined in http://pandas.pydata.org/pandas-docs/stable/timeseries.html#resampling.

    Returns
    -------
    xarray.DataArray
      The number of wet days for each period [day]

    Examples
    --------
    The following would compute for each grid cell of file `pr.day.nc` the number days
    with precipitation over 5 mm at the seasonal frequency, ie DJF, MAM, JJA, SON, DJF, etc.

    >>> pr = xr.open_dataset('pr.day.nc')
    >>> wd = wet_days(pr, pr_min = 5., freq="QS-DEC")

    """

    wd = (pr >= thresh) * 1
    return wd.resample(time=freq).sum(dim='time')


def daily_intensity(pr, thresh=1.0, freq='YS'):
    r"""Average daily precipitation intensity

    Return the average precipitation over wet days.

    Parameters
    ----------
    pr : xarray.DataArray
      Daily precipitation [mm]
    thresh : float
      precipitation value over which a day is considered wet. Default: 1mm.
    freq : str, optional
      Resampling frequency defining the periods
      defined in http://pandas.pydata.org/pandas-docs/stable/timeseries.html#resampling.

    Returns
    -------
    xarray.DataArray
      The average precipitation over wet days for each period

    Examples
    --------
    The following would compute for each grid cell of file `pr.day.nc` the average
    precipitation fallen over days with precipitation >= 5 mm at seasonal
    frequency, ie DJF, MAM, JJA, SON, DJF, etc.

    >>> pr = xr.open_dataset('pr.day.nc')
    >>> daily_int = daily_intensity(pr, thresh=5., freq="QS-DEC")

    """

    # put pr=0 for non wet-days
    pr_wd = xr.where(pr >= thresh, pr, 0)

    # sum over wanted period
    s = pr_wd.resample(time=freq).sum(dim='time')

    # get number of wet_days over period
    wd = wet_days(pr, thresh=thresh, freq=freq)

    return s / wd

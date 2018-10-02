# -*- coding: utf-8 -*-

"""Main module.
"""
from functools import wraps

import numpy as np
import xarray as xr

from . import run_length as rl
from .checks import valid_daily_mean_temperature, valid_daily_max_min_temperature, valid_daily_min_temperature, \
    valid_daily_max_temperature, valid_daily_mean_discharge  # , assert_daily

# Frequencies : YS: year start, QS-DEC: seasons starting in december, MS: month start
# See http://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases
K2C = 273.15
ftomm = np.nan


# TODO: Move utility functions to another file.
# TODO: Should we reference the standard vocabulary we're using ?
# E.g. http://vocab.nerc.ac.uk/collection/P07/current/BHMHISG2/

def first_paragraph(txt):
    """Return the first paragraph of a text."""
    return txt.split('\n\n')[0]


def with_attrs(**func_attrs):
    """Set attributes in the decorated function at definition time,
    and assign these attributes to the function output at the
    execution time.

    Note
    ----
    Assumes the output has an attrs dictionary attribute (e.g. xarray.DataArray).
    """

    def attr_decorator(fn):
        # Use the docstring as the description attribute.
        func_attrs['description'] = first_paragraph(fn.__doc__)

        @wraps(fn)
        def wrapper(*args, **kwargs):
            out = fn(*args, **kwargs)
            out.attrs.update(func_attrs)
            return out

        # Assign the attributes to the function itself
        for attr, value in func_attrs.items():
            setattr(wrapper, attr, value)

        return wrapper

    return attr_decorator


# -------------------------------------------------- #
# ATTENTION: ASSUME ALL INDICES WRONG UNTIL TESTED ! #
# -------------------------------------------------- #

@with_attrs(standard_name='water_volume_transport_in_river_channel', long_name='discharge', units='m3 s-1')
@valid_daily_mean_discharge
def base_flow_index(q, freq='YS'):
    """Base flow index.

    Return the base flow index, defined as the minimum 7-day average flow divided by the mean flow.

    Parameters
    ----------
    q : xarray.DataArray
      The river flow [m³/s].
    freq : str, optional
      Resampling frequency.
    """
    m7 = q.rolling(time=7, center=True).mean(dim='time').resample(time=freq)
    mq = q.resample(time=freq)

    m7m = m7.min(dim='time')
    return m7m / mq.mean(dim='time')


@with_attrs(standard_name='cold_spell_duration_index', long_name='', units='days')
@valid_daily_min_temperature
def cold_spell_duration_index(tasmin, tn10, freq='YS'):
    """Cold spell duration index.

    Resample the daily minimum temperature series by returning the number of days per
    period where the temperature is below the calendar day 10th percentile (calculated
    over a centered 5-day window for values during the 1961–1990 period) for a minimum of
    at least six consecutive days.

    Parameters
    ----------
    tasmin : xarray.DataArray
      Minimum daily temperature values [K].
    tn10 : xarray.DataArray
      The climatological 10th percentile of the minimum daily temperature computed over
      the period 1961-1990, computed for each day of the year using a centered 5-day window.
    freq : str, optional
      Resampling frequency.


    See also
    --------
    percentile_doy
    """

    def func(x):
        xr.apply_ufunc(rl.windowed_run_count,
                       x,
                       input_core_dims=[['time'], ],
                       vectorize=True,
                       dask='parallelized',
                       output_dtypes=[np.int, ],
                       keep_attrs=True,
                       kwargs={'window': 6})

    return tasmin.pipe(lambda x: x - tn10) \
        .resample(time=freq) \
        .apply(func)


@valid_daily_mean_temperature
def cold_spell_index(tas, thresh=-10, window=5, freq='AS-JUL'):
    """Cold spell index.

    Days that are part of a cold spell (5 or more consecutive days with Tavg < -10°C)
    """
    over = tas < K2C + thresh
    group = over.resample(time=freq)

    def func(x):
        xr.apply_ufunc(rl.windowed_run_count,
                       x,
                       input_core_dims=[['time'], ],
                       vectorize=True,
                       dask='parallelized',
                       output_dtypes=[np.int, ],
                       keep_attrs=True,
                       kwargs={'window': window})

    return group.apply(func)


def cold_and_dry_days(tas, tgin25, pr, wet25, freq='YS'):
    """Cold and dry days.

    See Beniston (2009)...

    """
    c1 = tas < tgin25
    c2 = (pr > 1 * ftomm) * (pr < wet25)

    c = (c1 * c2) * 1
    return c.resample(time=freq).sum(dim='time')


@valid_daily_min_temperature
def consecutive_frost_days(tasmin, freq='AS-JUL'):
    """Maximum number of consecutive frost days (Tmin < 0℃).

    Resample the daily minimum temperature series by computing the maximum number
    of days below the freezing point over each period.

    Parameters
    ----------
    tasmin : xarray.DataArray
      Minimum daily temperature values [K].
    freq : str, optional
      Resampling frequency.

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

    # TODO: Deal with start and end boundaries
    # TODO: Check that input array has no missing dates (holes)

    # Create an monotonously increasing index [0,1,2,...] along the time dimension.
    i = xr.DataArray(np.arange(tasmin.time.size), dims='time')
    ind = xr.broadcast(i, tasmin)[0]

    # Mask index  values where tasmin > K2C
    d = ind.where(tasmin > K2C)

    # Fill NaNs with the following valid value
    b = d.bfill(dim='time')

    # Find the difference between start and end indices
    d = b.diff(dim='time') - 1

    return d.resample(time=freq).max(dim='time')


def consecutive_wet_days(pr, thresh=1, freq='YS'):
    """Maximum number of consecutive wet days."""
    raise NotImplementedError


# TODO: Clarify valid daily min temperature function and implement it
# @valid_daily_min_temperature
# def CFD2(tasmin, freq='AS-JUL'):
#     # Results are different from CFD, possibly because resampling occurs at first.
#     # CFD looks faster anyway.
#     f = tasmin < K2C
#     group = f.resample(time=freq)
#
#     def func(x):
#         return xr.apply_ufunc(rl.longest_run,
#                                 x,
#                                 input_core_dims=[['time'], ],
#                                 vectorize=True,
#                                 dask='parallelized',
#                                 output_dtypes=[np.int, ],
#                                 keep_attrs=True,
#                                 )
#
#     return group.apply(func)


@with_attrs(standard_name='cooling_degree_days', long_name='cooling degree days', units='K*day')
@valid_daily_mean_temperature
def cooling_degree_days(tas, thresh=18, freq='YS'):
    """Cooling degree days above threshold."""

    return tas.pipe(lambda x: x - thresh - K2C) \
        .clip(min=0) \
        .resample(time=freq) \
        .sum(dim='time')


@valid_daily_max_min_temperature
def daily_freezethaw_cycles(tasmax, tasmin, freq='YS'):
    """Number of days with a freeze-thaw cycle.

    The number of days where Tmax > 0℃ and Tmin < 0℃.
    """
    ft = (tasmin < K2C) * (tasmax > K2C) * 1
    return ft.resample(time=freq).sum(dim='time')


@valid_daily_max_min_temperature
def daily_temperature_range(tasmax, tasmin, freq='YS'):
    """Mean of daily temperature range."""
    dtr = tasmax - tasmin
    return dtr.resample(time=freq).mean(dim='time')


@valid_daily_mean_temperature
def freshet_start(tas, thresh=0.0, window=5, freq='YS'):
    r"""First day of consistent threshold temperature.

    Return first day of year when a temperature threshold is exceeded
    over a given number of days.

    Parameters
    ----------
    tas : xarray.DataArray
      Mean daily temperature
    thresh : float
      Threshold temperature to base evaluation [℃]
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


@valid_daily_min_temperature
def frost_days(tasmin, freq='YS'):
    """Number of frost days (Tmin < 0℃)."""
    f = (tasmin < K2C) * 1
    return f.resample(time=freq).sum(dim='time')


@valid_daily_mean_temperature
def growing_degree_days(tas, thresh=4, freq='YS'):
    """Growing degree days over 4℃.

    The sum of degree-days over 4℃.
    """
    return tas.pipe(lambda x: x - thresh - K2C) \
        .clip(min=0) \
        .resample(time=freq) \
        .sum(dim='time')


@valid_daily_mean_temperature
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
      Mean daily temperature
    thresh : float
      Threshold temperature to base evaluation [℃]
    window : int
      Minimum number of days with temperature above threshold to mark the beginning and end of growing season.
    freq : str, optional
      Resampling frequency
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


@valid_daily_max_temperature
def heat_wave_index(tasmax, thresh=25.0, window=5, freq='YS'):
    r"""Heat wave index.

    Number of days that are part of a heatwave, defined as five or more consecutive days over 25℃.

    Parameters
    ----------
    tasmax : xarrray.DataArray
      Maximum daily temperature.
    thresh : float
      Threshold temperature to designate a heatwave [℃].
    window : int
      Minimum number of days with temperature above threshold to qualify as a heatwave.
    freq : str, optional
      Resampling frequency.

    Returns
    -------
    DataArray
      Heat wave index.
    """
    # TODO: Deal better with boundary effects.
    # TODO: Deal with attributes

    over = tasmax > K2C + thresh
    group = over.resample(time=freq)

    def func(x):
        xr.apply_ufunc(rl.windowed_run_count,
                       x,
                       input_core_dims=[['time'], ],
                       vectorize=True,
                       dask='parallelized',
                       output_dtypes=[np.int, ],
                       keep_attrs=True,
                       kwargs={'window': window})

    return group.apply(func)


@valid_daily_mean_temperature
def heating_degree_days(tas, freq='YS', thresh=17):
    """Heating degree days.

    Sum of positive values of threshold - tas.
    """
    return tas.pipe(lambda x: thresh - x) \
        .clip(0) \
        .resample(time=freq) \
        .sum(dim='time')


@valid_daily_max_temperature
def hot_days(tasmax, thresh=30, freq='YS'):
    """Number of very hot days.

    The number of days exceeding a threshold. """
    hd = (tasmax > K2C + thresh) * 1
    return hd.resample(time=freq).sum(dim='time')


@valid_daily_max_temperature
def ice_days(tasmax, freq='YS'):
    """Number of days where the daily maximum temperature is below 0℃."""
    f = (tasmax < K2C) * 1
    return f.resample(time=freq).sum(dim='time')


@valid_daily_max_temperature
def summer_days(tasmax, thresh=25, freq='YS'):
    f = (tasmax > thresh + K2C) * 1
    return f.resample(time=freq).sum(dim='time')


@valid_daily_mean_temperature
def tg_mean(tas, freq='YS'):
    r"""Mean of daily average temperature.

    Resample the original daily mean temperature series by taking the mean over each period.

    Parameters
    ----------
    tas : xarray.DataArray
      Mean daily temperature values [K].
    freq : str, optional
      Resampling frequency.


    Returns
    -------
    xarray.DataArray
      The mean daily temperature at the given time frequency.


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


# @valid_daily_min_temperature
def tn10p(tasmin, p10, freq='YS'):
    """Days with daily minimum temperature below the 10th percentile of the reference period."""
    return (tasmin.groupby('time.dayofyear') < p10).resample(time=freq).sum(dim='time')


@valid_daily_min_temperature
def tn_max(tasmin, freq='YS'):
    """Maximum of daily minimum temperature."""
    return tasmin.resample(time=freq).max(dim='time')


@valid_daily_min_temperature
def tn_mean(tasmin, freq='YS'):
    """Mean of daily minimum temperature."""
    arr = tasmin.resample(time=freq) if freq else tasmin
    return arr.mean(dim='time')


@valid_daily_min_temperature
def tn_min(tasmin, freq='YS'):
    """Minimum of daily minimum temperature."""
    return tasmin.resample(time=freq).min(dim='time')


# @check_is_dataarray
def prcp_tot(pr, freq='YS', units='kg m-2 s-1'):
    r"""Accumulated total (liquid + solid) precipitation

    Resample the original daily mean precipitation flux and cumulate over each period

    Parameters
    ----------
    pr : xarray.DataArray
      Mean daily precipitation flux [Kg m-2 s-1] or [mm].
    freq : str, optional
      Resampling frequency as defined in
      http://pandas.pydata.org/pandas-docs/stable/timeseries.html#resampling.
    units: str, optional
      units of the precipitation data. Must be within ['kg m-2 s-2', 'mm']

    Returns
    -------
    xarray.DataArray
      The total daily precipitation at the given time frequency in [mm].


    Note
    ----
    Let :math:`T_i` be the mean daily temperature of day `i`, then for a period `p` starting at
    day `a` and finishing on day `b`

    .. math::

       TG_p = \frac{\sum_{i=a}^{b} T_i}{b - a + 1}


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
        output *= 86400  # number of sec in 24h
    elif units == 'mm':
        # nothing to do
        pass
    else:
        raise RuntimeError('non conform units')
    return output


@valid_daily_min_temperature
def tropical_nights(tasmin, thresh=20, freq='YS'):
    """Number of days with minimum daily temperature above threshold."""
    return tasmin.pipe(lambda x: (tasmin > thresh + K2C) * 1) \
        .resample(time=freq) \
        .sum(dim='time')


@valid_daily_max_temperature
def tx_max(tasmax, freq='YS'):
    """Maximum of daily maximum temperature."""
    return tasmax.resample(time=freq).max(dim='time')


@valid_daily_max_temperature
def tx_mean(tasmax, freq='YS'):
    """Mean of daily maximum temperature."""
    arr = tasmax.resample(time=freq) if freq else tasmax
    return arr.mean(dim='time')


@valid_daily_max_temperature
def tx_min(tasmax, freq='YS'):
    """Minimum of daily maximum temperature."""
    return tasmax.resample(time=freq).min(dim='time')


def percentile_doy(arr, window=5, per=.1):
    """Compute the climatological percentile over a moving window
    around the day of the year.
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

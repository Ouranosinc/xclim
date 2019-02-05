# -*- coding: utf-8 -*-

"""
Indices library
===============

This module describes climate indicator functions. Functions are listed in alphabetical order and describe the raw
computation performed over xarray.DataArrays that are assumed to be valid and with the correct units. The output's
attributes (CF-Convention) are not modified. Validation checks, output attributes and unit conversion are handled by
indicator classes described in files named by the physical variable (temperature, precip, streamflow).

Notes for docstring
-------------------

The docstrings adhere to the `NumPy`_ style convention and is meant as a way to store CF-Convention metadata as
well as information relevant to third party libraries (such as a WPS server).

The first line of the docstring (the short summary), will be assigned to the output's `long_name` attribute. The
`long_name` attribute is defined by the NetCDF User Guide to contain a long descriptive name which may, for example,
be used for labeling plots

The second paragraph will be considered as the "*abstract*", or the CF global "*comment*" (miscellaneous information
about the data or methods used to produce it).

The third and fourth sections are the **Parameters** and **Returns** sections describing the input and output values
respectively.

.. code-block:: python

   Parameters
   ----------
   <standard_name> : xarray.DataArray
     <Long_name> of variable [acceptable units].
   threshold : float
     Description of the threshold / units.
     e.g. The 10th percentile of historical temperature [K].
   freq : str, optional
     Resampling frequency.

   Returns
   -------
   xarray.DataArray
     Output's <long_name> [units]

The next sections would be **Notes** and **References**:

.. code-block:: python

    Notes
    -----
    This is where the mathematical equation is described.
    At the end of the description, convention suggests
    to add a reference [example]_:

        .. math::

            3987^12 + 4365^12 = 4472^12

    References
    ----------
    .. [example] Smith, T.J. and Huard, D. (2018). "CF Docstrings:
        A manifesto on conventions and the metaphysical nature
        of ontological python documentation." Climate Aesthetics,
        vol. 1, pp. 121-155.

Indice descriptions
===================
.. _`NumPy`: https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard
"""
import logging

import numpy as np
from . import utils
import xarray as xr

from . import run_length as rl

logging.basicConfig(level=logging.DEBUG)
logging.captureWarnings(True)

xr.set_options(enable_cftimeindex=True)  # Set xarray to use cftimeindex

# if six.PY2:
#     from funcsigs import signature
# elif six.PY3:
#     from inspect import signature


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

    m7 = q.rolling(time=7, center=True).mean().resample(time=freq)
    mq = q.resample(time=freq)

    m7m = m7.min(dim='time')
    return m7m / mq.mean(dim='time')


def cold_spell_duration_index(tasmin, tn10, window=6, freq='YS'):
    r"""Cold spell duration index

    Number of days with at least six consecutive days where the daily minimum temperature is below the 10th
    percentile. The 10th percentile should be computed for a 5-day window centred on each calendar day in the
    1961-1990 period.

    Parameters
    ----------
    tasmin : xarray.DataArray
      Minimum daily temperature [℃] or [K]
    tn10 : float
      10th percentile of daily minimum temperature [K]
    freq : str, optional
      Resampling frequency

    Returns
    -------
    xarray.DataArray
      Count of days with at least six consecutive days where the daily minimum temperature is below the 10th
      percentile [days].

    References
    ----------
    From the Expert Team on Climate Change Detection, Monitoring and Indices (ETCCDMI).

    Example
    -------
    >>> tn10 = percentile_doy(historical_tasmin, per=.1)
    >>> cold_spell_duration_index(reference_tasmin, tn10)
    """
    if 'dayofyear' not in tn10.coords.keys():
        raise AttributeError("tn10 should have dayofyear coordinates.")

    # The day of year value of the tasmin series.
    doy = tasmin.indexes['time'].dayofyear

    # Create an array with the shape and coords of tasmin, but with values set to tx90 according to the doy index.
    thresh = xr.full_like(tasmin, np.nan)
    thresh.data = tn10.sel(dayofyear=doy)

    below = (tasmin < thresh)

    return below.resample(time=freq).apply(rl.windowed_run_count, window=window, dim='time')


def cold_spell_days(tas, thresh=-10, window=5, freq='AS-JUL'):
    r"""Cold spell days

    The number of days that are part of a cold spell, defined as five or more consecutive days with mean daily
    temperature below a threshold in °C.

    Parameters
    ----------
    tas : xarrray.DataArray
      Mean daily temperature [℃] or [K]
    thresh : float
      Threshold temperature below which a cold spell begins [℃] or [K]
    window : int
      Minimum number of days with temperature below threshold to qualify as a cold spell.
    freq : str, optional
      Resampling frequency

    Returns
    -------
    DataArray
      Cold spell days
    """

    over = tas < (K2C + thresh)
    group = over.resample(time=freq)

    return group.apply(rl.windowed_run_count, window=window, dim='time')


# TODO: mix up in docsring for tas
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

    Returns
    -------
    xarray.DataArray
      The total number of days where "Cold" and "Dry conditions coincide.

    Notes
    -----
    Formula to be written [cold_dry_days]_.

    References
    ----------
    .. [cold_dry_days] Beniston, M. (2009). Trends in joint quantiles of temperature and precipitation in Europe
        since 1901 and projected for 2100. Geophysical Research Letters, 36(7). https://doi.org/10.1029/2008GL037119
    """

    c1 = tas < tgin25
    c2 = (pr > 1 * ftomm) * (pr < wet25)

    c = (c1 * c2) * 1
    return c.resample(time=freq).sum(dim='time')


def daily_pr_intensity(pr, thresh=1.0, freq='YS'):
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
    frequency, ie DJF, MAM, JJA, SON, DJF, etc.:

    >>> pr = xr.open_dataset('pr.day.nc')
    >>> daily_int = daily_pr_intensity(pr, thresh=5., freq="QS-DEC")

    """

    # put pr=0 for non wet-days
    pr_wd = xr.where(pr >= thresh, pr, 0)

    # sum over wanted period
    s = pr_wd.resample(time=freq).sum(dim='time')

    # get number of wetdays over period
    wd = wetdays(pr, thresh=thresh, freq=freq)

    return s / wd


def maximum_consecutive_dry_days(pr, thresh=1.0, freq='YS'):
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
    return group.apply(rl.longest_run, dim='time')


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

    Notes
    -----
    Let :math:`Tmin_i` be the minimum daily temperature of day :math:`i`, then for a period :math:`p` starting at
    day :math:`a` and finishing on day :math:`b`

    .. math::

       CFD_p = max(run_l(Tmin_i < 273.15))

    for :math:`a ≤ i ≤ b`

    where run_l returns the length of each consecutive series of true values.
    """

    group = (tasmin < K2C).resample(time=freq)
    return group.apply(rl.longest_run, dim='time')


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

    Returns
    -------
    xarray.DataArray
      The maximum number of consecutive wet days.

    Notes
    -----
    Let :math:`RR_{ij}` be the daily precipitation amount for day :math:`i` of period :math:`j`. Then
    counted is the largest number of consecutive days where:

    .. math::

        RR_{ij} ≥ 1 mm
    """

    group = (pr > thresh).resample(time=freq)
    return group.apply(rl.longest_run, dim='time')


def cooling_degree_days(tas, thresh=18, freq='YS'):
    r"""Cooling degree days

    Sum of degree days above the temperature threshold at which spaces are cooled.

    Parameters
    ----------
    tas : xarray.DataArray
      Mean daily temperature [℃] or [K]
    thresh : float
      Temperature threshold above which air is cooled.
    freq : str, optional
      Resampling frequency

    Returns
    -------
    xarray.DataArray
      Cooling degree days
    """

    return tas.pipe(lambda x: x - thresh - K2C) \
        .clip(min=0) \
        .resample(time=freq) \
        .sum(dim='time')


def daily_freezethaw_cycles(tasmax, tasmin, freq='YS'):
    r"""Number of days with a diurnal freeze-thaw cycle

    The number of days where Tmax > 0℃ and Tmin < 0℃.


    Parameters
    ----------
    tasmax : xarray.DataArray
      Maximum daily temperature [℃] or [K]
    tasmin : xarray.DataArray
      Minimum daily temperature values [℃] or [K]
    freq : str
      Resampling frequency


    Returns
    -------
    xarray.DataArray
      Number of days with a diurnal freeze-thaw cycle

    """

    ft = (tasmin < K2C) * (tasmax > K2C) * 1
    return ft.resample(time=freq).sum(dim='time')


def daily_temperature_range(tasmax, tasmin, freq='YS'):
    r"""Mean of daily temperature range.

    The mean difference between the daily maximum temperature and the daily minimum temperature.

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

    Notes
    -----
    Let :math:`TX_{ij}` and :math:`TN_{ij}` be the daily maximum and minimum temperature at day :math:`i`
    of period :math:`j`. Then the mean diurnal temperature range in period :math:`j` is:

    .. math::

        DTR_j = \frac{ \sum_{i=1}^I (TX_{ij} - TN_{ij}) }{I}
    """

    dtr = tasmax - tasmin
    return dtr.resample(time=freq).mean(dim='time')


# TODO: Improve description.
def daily_temperature_range_variability(tasmax, tasmin, freq="YS"):
    r"""Mean absolute day-to-day variation in daily temperature range.

    Mean absolute day-to-day variation in daily temperature range.

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

    Notes
    -----
    Let :math:`TX_{ij}` and :math:`TN_{ij}` be the daily maximum and minimum temperature at
    day :math:`i` of period :math:`j`. Then calculated is the absolute day-to-day differences in
    period :math:`j` is:

    .. math::

       vDTR_j = \frac{ \sum_{i=2}^{I} |(TX_{ij}-TN_{ij})-(TX_{i-1,j}-TN_{i-1,j})| }{I}
    """

    vdtr = abs((tasmax - tasmin).diff(dim='time'))
    return vdtr.resample(time=freq).mean(dim='time')


def extreme_temperature_range(tasmax, tasmin, freq='YS'):
    r"""Extreme intra-period temperature range.

    The maximum of max temperature (TXx) minus the minimum of min temperature (TNn) for the given time period.

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
      Extreme intra-period temperature range for the given time period.

    Notes
    -----
    Let :math:`TX_{ij}` and :math:`TN_{ij}` be the daily maximum and minimum temperature at day :math:`i`
    of period :math:`j`. Then the extreme temperature range in period :math:`j` is:

    .. math::

        ETR_j = max(TX_{ij}) - min(TN_{ij})
    """

    tx_max = tasmax.resample(time=freq).max(dim='time')
    tn_min = tasmin.resample(time=freq).min(dim='time')

    return tx_max - tn_min


def freshet_start(tas, thresh=0.0, window=5, freq='YS'):
    r"""First day consistently exceeding threshold temperature.

    Returns first day of period where a temperature threshold is exceeded
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

    Returns
    -------
    float
      Day of the year when temperature exceeds threshold over a given number of days for the first time. If there are
      no such day, return np.nan.

    """

    over = (tas > K2C + thresh)
    group = over.resample(time=freq)
    return group.apply(rl.first_run_ufunc, window=window, index='dayofyear')


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

    Notes
    -----
    Let :math:`TN_{ij}` be the daily minimum temperature at day :math:`i` of period :math:`j`. Then
    counted is the number of days where:

    .. math::

        TN_{ij} < 0℃
    """

    f = (tasmin < K2C) * 1
    return f.resample(time=freq).sum(dim='time')


def growing_degree_days(tas, thresh=4.0, freq='YS'):
    r"""Growing degree-days over threshold temperature value [℃].

    The sum of degree-days over the threshold temperature.

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

    Notes
    -----
    Let :math:`TG_{ij}` be the daily mean temperature at day :math:`i` of period :math:`j`. Then the
    growing degree days are:

    .. math::

        GD4_j = \sum_{i=1}^I (TG_{ij}-{4} | TG_{ij} > {4}℃)
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
      Mean daily temperature [℃] or [K]
    thresh : float
      Threshold temperature on which to base evaluation [℃] or [K]. Default: 5℃.
    window : int
      Minimum number of days with temperature above threshold to mark the beginning and end of growing season.
    freq : str, optional
      Resampling frequency

    Returns
    -------
    xarray.DataArray
      Growing season length.

    Notes
    -----
    Let :math:`TG_{ij}` be the mean temperature at day :math:`i` of period :math:`j`. Then counted is
    the number of days between the first occurrence of at least 6 consecutive days with:

    .. math::

        TG_{ij} > 5 ℃

    and the first occurrence after 1 July of at least 6 consecutive days with:

    .. math::

        TG_{ij} < 5 ℃
    """

    # i = xr.DataArray(np.arange(tas.time.size), dims='time')
    # ind = xr.broadcast(i, tas)[0]
    #
    # c = ((tas > thresh + K2C) * 1).rolling(time=window).sum()
    # i1 = ind.where(c == window).resample(time=freq).min(dim='time')
    #
    # # Resample sets the time to T00:00.
    # i11 = i1.reindex_like(c, method='ffill')
    #
    # # TODO: Adjust for southern hemisphere
    #
    # #i2 = ind.where(c == 0).where(tas.time.dt.month >= 7)
    # # add check to make sure indice of end of growing season is after growing season start
    # i2 = ind.where((c==0) & (ind > i11)).where(tas.time.dt.month >= 7)
    #
    # d = i2 - i11
    #
    # # take min value (first occurence after july)
    # gsl = d.resample(time=freq).min(dim='time')
    #
    # # turn nan into 0
    # gsl = xr.where(np.isnan(gsl), 0, gsl)

    # compute growth season length on resampled data

    c = ((tas > thresh + K2C) * 1).rolling(time=window).sum().chunk(tas.chunks)

    def compute_gsl(c):
        nt = c.time.size
        i = xr.DataArray(np.arange(nt), dims='time').chunk({'time': 1})
        ind = xr.broadcast(i, c)[0].chunk(c.chunks)
        i1 = ind.where(c == window).min(dim='time')
        i1 = xr.where(np.isnan(i1), nt, i1)
        i11 = i1.reindex_like(c, method='ffill')
        i2 = ind.where((c == 0) & (ind > i11)).where(c.time.dt.month >= 7)
        i2 = xr.where(np.isnan(i2), nt, i2)
        d = (i2 - i1).min(dim='time')
        return d

    gsl = c.resample(time=freq).apply(compute_gsl)

    return gsl


def heat_wave_frequency(tasmin, tasmax, thresh_tasmin=22.0, thresh_tasmax=30,
                        window=3, freq='YS'):
    # Dev note : we should decide if it is deg K or C
    r"""Heat wave frequency

    Number of heat waves over a given period. A heat wave is defined as an event
    where the minimum and maximum daily temperature both exceeds specific thresholds
    over a minimum number of days.

    Parameters
    ----------

    tasmin : xarrray.DataArray
      Minimum daily temperature [℃] or [K]
    tasmax : xarrray.DataArray
      Maximum daily temperature [℃] or [K]
    thresh_tasmin : float
      The minimum temperature threshold needed to trigger a heatwave event [℃]
    thresh_tasmax : float
      The maximum temperature threshold needed to trigger a heatwave event [℃]
    window : int
      Minimum number of days with temperatures above thresholds to qualify as a heatwave.
    freq : str, optional
      Resampling frequency

    Returns
    -------
    xarray.DataArray
      Number of heatwave at the wanted frequency


    Notes
    -----
    The thresholds of 22° and 25°C for night temperatures and 30° and 35°C for day temperatures were selected by
    Health Canada professionals, following a temperature–mortality analysis. These absolute temperature thresholds
    characterize the occurrence of hot weather events that can result in adverse health outcomes for Canadian
    communities (Casati et al., 2013).

    In Robinson (2001), the parameters would be `thresh_tasmin=27.22, thresh_tasmax=39.44, window=2` (81F, 103F).

    References
    ----------
    Casati, B., A. Yagouti, and D. Chaumont, 2013: Regional Climate Projections of Extreme Heat Events in Nine Pilot
    Canadian Communities for Public Health Planning. J. Appl. Meteor. Climatol., 52, 2669–2698,
    https://doi.org/10.1175/JAMC-D-12-0341.1

    Robinson, P.J., 2001: On the Definition of a Heat Wave. J. Appl. Meteor., 40, 762–775,
    https://doi.org/10.1175/1520-0450(2001)040<0762:OTDOAH>2.0.CO;2
    """

    cond = (tasmin > thresh_tasmin + K2C) & (tasmax > thresh_tasmax + K2C)
    group = cond.resample(time=freq)
    return group.apply(rl.windowed_run_events, window=window, dim='time')


def heat_wave_index(tasmax, thresh=25.0, window=5, freq='YS'):
    r"""Heat wave index.

    Number of days that are part of a heatwave, defined as five or more consecutive days over 25℃.

    Parameters
    ----------
    tasmax : xarrray.DataArray
      Maximum daily temperature [℃] or [K]
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

    over = tasmax > K2C + thresh
    group = over.resample(time=freq)

    return group.apply(rl.windowed_run_count, window=window, dim='time')


def heat_wave_max_length(tasmin, tasmax, thresh_tasmin=22.0, thresh_tasmax=30,
                         window=3, freq='YS'):
    # Dev note : we should decide if it is deg K or C
    r"""Heat wave max length

    Maximum length of heat waves over a given period. A heat wave is defined as an event
    where the minimum and maximum daily temperature both exceeds specific thresholds
    over a minimum number of days.

    By definition heat_wave_max_length must be >= window.

    Parameters
    ----------

    tasmin : xarrray.DataArray
      Minimum daily temperature [℃] or [K]
    tasmax : xarrray.DataArray
      Maximum daily temperature [℃] or [K]
    thresh_tasmin : float
      The minimum temperature threshold needed to trigger a heatwave event [℃]
    thresh_tasmax : float
      The maximum temperature threshold needed to trigger a heatwave event [℃]
    window : int
      Minimum number of days with temperatures above thresholds to qualify as a heatwave.
    freq : str, optional
      Resampling frequency

    Returns
    -------
    xarray.DataArray
      Maximum length of heatwave at the wanted frequency


    Notes
    -----
    The thresholds of 22° and 25°C for night temperatures and 30° and 35°C for day temperatures were selected by
    Health Canada professionals, following a temperature–mortality analysis. These absolute temperature thresholds
    characterize the occurrence of hot weather events that can result in adverse health outcomes for Canadian
    communities (Casati et al., 2013).

    In Robinson (2001), the parameters would be `thresh_tasmin=27.22, thresh_tasmax=39.44, window=2` (81F, 103F).

    References
    ----------
    Casati, B., A. Yagouti, and D. Chaumont, 2013: Regional Climate Projections of Extreme Heat Events in Nine Pilot
    Canadian Communities for Public Health Planning. J. Appl. Meteor. Climatol., 52, 2669–2698,
    https://doi.org/10.1175/JAMC-D-12-0341.1

    Robinson, P.J., 2001: On the Definition of a Heat Wave. J. Appl. Meteor., 40, 762–775,
    https://doi.org/10.1175/1520-0450(2001)040<0762:OTDOAH>2.0.CO;2
    """

    cond = (tasmin > thresh_tasmin + K2C) & (tasmax > thresh_tasmax + K2C)
    group = cond.resample(time=freq)
    max_l = group.apply(rl.longest_run, dim='time')
    return max_l.where(max_l >= window, 0)


def heating_degree_days(tas, freq='YS', thresh=17.0):
    r"""Heating degree days

    Sum of degree days below the temperature threshold at which spaces are heated.

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

    Notes
    -----
    Let :math:`TG_{ij}` be the daily mean temperature at day :math:`i` of period :math:`j`. Then the
    heating degree days are:

    .. math::

        HD17_j = \sum_{i=1}^{I} (17℃ - TG_{ij})
    """

    return tas.pipe(lambda x: K2C + thresh - x) \
        .clip(0) \
        .resample(time=freq) \
        .sum(dim='time')


def ice_days(tasmax, freq='YS'):
    r"""Number of ice/freezing days

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
      Number of ice/freezing days.

    Notes
    -----
    Let :math:`TX_{ij}` be the daily maximum temperature at day :math:`i` of period :math:`j`. Then
    counted is the number of days where:

    .. math::

        TX_{ij} < 0℃
    """

    f = (tasmax < K2C) * 1
    return f.resample(time=freq).sum(dim='time')


def liquid_precip_ratio(pr, prsn=None, tas=None, freq='QS-DEC'):
    r"""Ratio of rainfall to total precipitation

    The ratio of total liquid precipitation over the total precipitation. If solid precipitation is not provided,
    then precipitation is assumed solid if the temperature is below 0°C.

    Parameters
    ----------
    pr : xarray.DataArray
      Mean daily precipitation flux [Kg m-2 s-1] or [mm].
    prsn : xarray.DataArray
      Mean daily solid precipitation flux [Kg m-2 s-1] or [mm].
    tas : xarray.DataArray
      Mean daily temperature [℃] or [K]
    freq : str
      Resampling frequency

    Returns
    -------
    xarray.DataArray
      Ratio of rainfall to total precipitation

    See also
    --------
    winter_rain_ratio
    """

    if prsn is None:
        prsn = pr.where(tas < K2C, 0)

    tot = pr.resample(time=freq).sum(dim='time')
    rain = tot - prsn.resample(time=freq).sum(dim='time')
    ratio = rain / tot
    return ratio


def tx_days_above(tasmax, thresh=25.0, freq='YS'):
    r"""Number of summer days

    Number of days where daily maximum temperature exceed a threshold.

    Parameters
    ----------
    tasmax : xarray.DataArray
      Maximum daily temperature [℃] or [K]
    thresh : float
      Threshold temperature on which to base evaluation [℃] or [K]. Default: 25℃.
    freq : str, optional
      Resampling frequency

    Returns
    -------
    xarray.DataArray
      Number of summer days.

    Notes
    -----
    Let :math:`TX_{ij}` be the daily maximum temperature at day :math:`i` of period :math:`j`. Then
    counted is the number of days where:

    .. math::

        TX_{ij} > 25℃
    """

    f = (tasmax > thresh + K2C) * 1
    return f.resample(time=freq).sum(dim='time')


def max_n_day_precipitation_amount(pr, window=1, freq='YS'):
    r"""Highest precipitation amount cumulated over a n-day moving window.

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
    at an annual frequency:

    >>> da = xr.open_dataset('pr.day.nc').pr
    >>> window = 5
    >>> output = max_n_day_precipitation_amount(da, window, freq="YS")
    """

    # rolling sum of the values
    arr = pr.rolling(time=window, center=False).sum()
    return arr.resample(time=freq).max(dim='time')


def max_1day_precipitation_amount(pr, freq='YS'):
    r"""Highest 1-day precipitation amount for a period (frequency).

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
    at an annual frequency:

    >>> pr = xr.open_dataset('pr.day.nc').pr
    >>> rx1day = max_1day_precipitation_amount(pr, freq="YS")
    """

    return pr.resample(time=freq).max(dim='time')


def precip_accumulation(pr, freq='YS'):
    r"""Accumulated total (liquid + solid) precipitation.

    Resample the original daily mean precipitation flux and accumulate over each period.

    Parameters
    ----------
    pr : xarray.DataArray
      Mean daily precipitation flux [Kg m-2 s-1] or [mm].
    freq : str, optional
      Resampling frequency as defined in
      http://pandas.pydata.org/pandas-docs/stable/timeseries.html#resampling.

    Returns
    -------
    xarray.DataArray
      The total daily precipitation at the given time frequency.

    Notes
    -----
    Let :math:`pr_i` be the mean daily precipitation of day `i`, then for a period `p` starting at
    day `a` and finishing on day `b`

    .. math::
       out_p = \sum_{i=a}^{b} pr_i

    Examples
    --------
    The following would compute for each grid cell of file `pr_day.nc` the total
    precipitation at the seasonal frequency, ie DJF, MAM, JJA, SON, DJF, etc.:

    >>> pr_day = xr.open_dataset('pr_day.nc').pr
    >>> prcp_tot_seasonal = precip_accumulation(pr_day, freq="QS-DEC")
    """

    return pr.resample(time=freq).sum(dim='time')


def rain_on_frozen_ground_days(pr, tas, thresh=1, freq='YS'):
    """Number of rain on frozen ground events

    Number of days with rain above a threshold after a series of seven days below freezing temperature.
    Precipitation is assumed to be rain when the temperature is above 0C.

    Parameters
    ----------
    pr : xarray.DataArray
      Mean daily precipitation flux
    tas : xarray.DataArray
      Mean daily temperature [℃] or [K]
    thresh : float
      Precipitation threshold to consider a day as a rain event [mm]
    freq : str, optional
      Resampling frequency

    Returns
    -------
    xarray.DataArray
      The number of rain on frozen ground events per period [days]

    """

    def func(x, axis):
        """Check that temperature conditions are below 0 for seven days and above after."""
        frozen = x == np.array([0, 0, 0, 0, 0, 0, 0, 1], bool)
        return frozen.all(axis=axis)

    tcond = (tas > K2C).rolling(time=8).reduce(func)
    pcond = (pr > thresh)

    return (tcond * pcond * 1).resample(time=freq).sum(dim='time')


# TODO: Improve description
def tg90p(tas, t90, freq='YS'):
    r"""Number of days with daily mean temperature over the 90th percentile.

    Number of days with daily mean temperature over the 90th percentile.

    Parameters
    ----------
    tas : xarray.DataArray
      Mean daily temperature
    t90 : xarray.DataArray
      90th percentile of daily mean temperature
    freq : str, optional
      Resampling frequency

    Returns
    -------

    xarray.DataArray
      Count of days with daily mean temperature below the 10th percentile [days]

    Notes
    -----
    The 90th percentile should be computed for a 5 day window centered on each calendar day for a reference period.


    Example
    -------
    >>> t90 = percentile_doy(historical_tas, per=0.9)
    >>> hot_days = tg90p(tas, t90)
    """
    if 'dayofyear' not in t90.coords.keys():
        raise AttributeError("t10 should have dayofyear coordinates.")

    # adjustment of t90 to tas doy range
    t90 = utils.adjust_doy_calendar(t90, tas)

    # create array of percentile with tas shape and coords
    thresh = xr.full_like(tas, np.nan)
    doy = thresh.time.dt.dayofyear.values
    thresh.data = t90.sel(dayofyear=doy)

    # compute the cold days
    over = (tas > thresh)

    return over.resample(time=freq).sum(dim='time')


# TODO: Improve description
def tg10p(tas, t10, freq='YS'):
    r"""Number of days with daily mean temperature below the 10th percentile.

    Number of days with daily mean temperature below the 10th percentile.

    Parameters
    ----------
    tas : xarray.DataArray
      Mean daily temperature
    t10 : xarray.DataArray
      10th percentile of daily mean temperature
    freq : str, optional
      Resampling frequency

    Returns
    -------
    xarray.DataArray
      Count of days with daily mean temperature below the 10th percentile [days]

    Notes
    -----
    The 10th percentile should be computed for a 5 day window centered on each calendar day for a reference period.

    Example
    -------
    >>> t10 = percentile_doy(historical_tas, per=0.1)
    >>> cold_days = tg10p(tas, t10)
    """
    if 'dayofyear' not in t10.coords.keys():
        raise AttributeError("t10 should have dayofyear coordinates.")

    # adjustment of t10 to tas doy range
    t10 = utils.adjust_doy_calendar(t10, tas)

    # create array of percentile with tas shape and coords
    thresh = xr.full_like(tas, np.nan)
    doy = thresh.time.dt.dayofyear.values
    thresh.data = t10.sel(dayofyear=doy)

    # compute the cold days
    below = (tas < thresh)

    return below.resample(time=freq).sum(dim='time')


def tg_max(tas, freq='YS'):
    r"""Highest mean temperature.

    The maximum of daily mean temperature.

    Parameters
    ----------
    tas : xarray.DataArray
      Mean daily temperature [℃] or [K]
    freq : str, optional
      Resampling frequency

    Returns
    -------
    xarray.DataArray
      Maximum of daily minimum temperature.

    Notes
    -----
    Let :math:`TN_{ij}` be the mean temperature at day :math:`i` of period :math:`j`. Then the maximum
    daily mean temperature for period :math:`j` is:

    .. math::

        TNx_j = max(TN_{ij})
    """

    return tas.resample(time=freq).max(dim='time')


def tg_mean(tas, freq='YS'):
    r"""Mean of daily average temperature.

    Resample the original daily mean temperature series by taking the mean over each period.

    Parameters
    ----------
    tas : xarray.DataArray
      Mean daily temperature [℃] or [K]
    freq : str, optional
      Resampling frequency

    Returns
    -------
    xarray.DataArray
      The mean daily temperature at the given time frequency


    Notes
    -----
    Let :math:`T_i` be the mean daily temperature of day `i`, then for a period `p` starting at
    day `a` and finishing on day `b`

    .. math::

       TG_p = \frac{\sum_{i=a}^{b} T_i}{b - a + 1}


    Examples
    --------
    The following would compute for each grid cell of file `tas.day.nc` the mean temperature
    at the seasonal frequency, ie DJF, MAM, JJA, SON, DJF, etc.:

    >>> t = xr.open_dataset('tas.day.nc')
    >>> tg = tm_mean(t, freq="QS-DEC")
    """

    arr = tas.resample(time=freq) if freq else tas
    return arr.mean(dim='time')


def tg_min(tas, freq='YS'):
    r"""Lowest mean temperature

    Minimum of daily mean temperature.

    Parameters
    ----------
    tas : xarray.DataArray
      Mean daily temperature [℃] or [K]
    freq : str, optional
      Resampling frequency

    Returns
    -------
    xarray.DataArray
      Minimum of daily minimum temperature.

    Notes
    -----
    Let :math:`TG_{ij}` be the mean temperature at day :math:`i` of period :math:`j`. Then the minimum
    daily mean temperature for period :math:`j` is:

    .. math::

        TGn_j = min(TG_{ij})
    """

    return tas.resample(time=freq).min(dim='time')


# TODO: Improve description
def tn90p(tasmin, t90, freq='YS'):
    r"""Number of days with daily minimum temperature over the 90th percentile.

    Number of days with daily minimum temperature over the 90th percentile.

    Parameters
    ----------
    tasmin : xarray.DataArray
      Minimum daily temperature
    t90 : xarray.DataArray
      90th percentile of daily minimum temperature
    freq : str, optional
      Resampling frequency

    Returns
    -------
    xarray.DataArray
      Count of days with daily minimum temperature below the 10th percentile [days]

    Notes
    -----
    The 90th percentile should be computed for a 5 day window centered on each calendar day for a reference period.

    Example
    -------
    >>> t90 = percentile_doy(historical_tas, per=0.9)
    >>> hot_days = tg90p(tas, t90)
    """
    if 'dayofyear' not in t90.coords.keys():
        raise AttributeError("t10 should have dayofyear coordinates.")

    # adjustment of t90 to tas doy range
    t90 = utils.adjust_doy_calendar(t90, tasmin)

    # create array of percentile with tas shape and coords
    thresh = xr.full_like(tasmin, np.nan)
    doy = thresh.time.dt.dayofyear.values
    thresh.data = t90.sel(dayofyear=doy)

    # compute the cold days
    over = (tasmin > thresh)

    return over.resample(time=freq).sum(dim='time')


# TODO: Improve description
def tn10p(tasmin, t10, freq='YS'):
    r"""Number of days with daily minimum temperature below the 10th percentile.

    Number of days with daily minimum temperature below the 10th percentile.

    Parameters
    ----------

    tasmin : xarray.DataArray
      Mean daily temperature
    t10 : xarray.DataArray
      10th percentile of daily minimum temperature
    freq : str, optional
      Resampling frequency

    Returns
    -------
    xarray.DataArray
      Count of days with daily minimum temperature below the 10th percentile [days]

    Notes
    -----
    The 10th percentile should be computed for a 5 day window centered on each calendar day for a reference period.

    Example
    -------
    >>> t10 = percentile_doy(historical_tas, per=0.1)
    >>> cold_days = tg10p(tas, t10)
    """
    if 'dayofyear' not in t10.coords.keys():
        raise AttributeError("t10 should have dayofyear coordinates.")

    # adjustment of t10 to tas doy range
    t10 = utils.adjust_doy_calendar(t10, tasmin)

    # create array of percentile with tas shape and coords
    thresh = xr.full_like(tasmin, np.nan)
    doy = thresh.time.dt.dayofyear.values
    thresh.data = t10.sel(dayofyear=doy)

    # compute the cold days
    below = (tasmin < thresh)

    return below.resample(time=freq).sum(dim='time')


def tn_max(tasmin, freq='YS'):
    r"""Highest minimum temperature.

    The maximum of daily minimum temperature.

    Parameters
    ----------
    tasmin : xarray.DataArray
      Minimum daily temperature [℃] or [K]
    freq : str, optional
      Resampling frequency

    Returns
    -------
    xarray.DataArray
      Maximum of daily minimum temperature.

    Notes
    -----
    Let :math:`TN_{ij}` be the minimum temperature at day :math:`i` of period :math:`j`. Then the maximum
    daily minimum temperature for period :math:`j` is:

    .. math::

        TNx_j = max(TN_{ij})
    """

    return tasmin.resample(time=freq).max(dim='time')


def tn_mean(tasmin, freq='YS'):
    r"""Mean minimum temperature.

    Mean of daily minimum temperature.

    Parameters
    ----------
    tasmin : xarray.DataArray
      Minimum daily temperature [℃] or [K]
    freq : str, optional
      Resampling frequency

    Returns
    -------
    xarray.DataArray
      Mean of daily minimum temperature.

    Notes
    -----
    Let :math:`TN_{ij}` be the minimum temperature at day :math:`i` of period :math:`j`. Then mean
    values in period :math:`j` are given by:

    .. math::

        TN_{ij} = \frac{ \sum_{i=1}^{I} TN_{ij} }{I}
    """

    arr = tasmin.resample(time=freq) if freq else tasmin
    return arr.mean(dim='time')


def tn_min(tasmin, freq='YS'):
    r"""Lowest minimum temperature

    Minimum of daily minimum temperature.

    Parameters
    ----------
    tasmin : xarray.DataArray
      Minimum daily temperature [℃] or [K]
    freq : str, optional
      Resampling frequency

    Returns
    -------
    xarray.DataArray
      Minimum of daily minimum temperature.

    Notes
    -----
    Let :math:`TN_{ij}` be the minimum temperature at day :math:`i` of period :math:`j`. Then the minimum
    daily minimum temperature for period :math:`j` is:

    .. math::

        TNn_j = min(TN_{ij})
    """

    return tasmin.resample(time=freq).min(dim='time')


def tropical_nights(tasmin, thresh=20.0, freq='YS'):
    r"""Tropical nights

    The number of days with minimum daily temperature above threshold.

    Parameters
    ----------
    tasmin : xarray.DataArray
      Minimum daily temperature [℃] or [K]
    thresh : float
      Threshold temperature on which to base evaluation [℃] or [K]. Default: 20℃.
    freq : str, optional
      Resampling frequency

    Returns
    -------
    xarray.DataArray
      Number of days with minimum daily temperature above threshold.

    Notes
    -----
    Let :math:`TN_{ij}` be the daily minimum temperature at day :math:`i` of period :math:`j`. Then
    counted is the number of days where:

    .. math::

        TN_{ij} > 20℃
    """

    return tasmin.pipe(lambda x: (tasmin > thresh + K2C) * 1) \
        .resample(time=freq) \
        .sum(dim='time')


# TODO: Improve description
def tx90p(tasmax, t90, freq='YS'):
    r"""Number of days with daily maximum temperature over the 90th percentile.

    Number of days with daily maximum temperature over the 90th percentile.

    Parameters
    ----------
    tasmax : xarray.DataArray
      Maximum daily temperature
    t90 : xarray.DataArray
      90th percentile of daily maximum temperature
    freq : str, optional
      Resampling frequency

    Returns
    -------
    xarray.DataArray
      Count of days with daily maximum temperature below the 10th percentile [days]

    Notes
    -----
    The 90th percentile should be computed for a 5 day window centered on each calendar day for a reference period.

    Example
    -------
    >>> t90 = percentile_doy(historical_tas, per=0.9)
    >>> hot_days = tg90p(tas, t90)
    """
    if 'dayofyear' not in t90.coords.keys():
        raise AttributeError("t10 should have dayofyear coordinates.")

    # adjustment of t90 to tas doy range
    t90 = utils.adjust_doy_calendar(t90, tasmax)

    # create array of percentile with tas shape and coords
    thresh = xr.full_like(tasmax, np.nan)
    doy = thresh.time.dt.dayofyear.values
    thresh.data = t90.sel(dayofyear=doy)

    # compute the cold days
    over = (tasmax > thresh)

    return over.resample(time=freq).sum(dim='time')


# TODO: Improve description
def tx10p(tasmax, t10, freq='YS'):
    r"""Number of days with daily maximum temperature below the 10th percentile.

    Number of days with daily maximum temperature below the 10th percentile.

    Parameters
    ----------
    tas : xarray.DataArray
      Maximum daily temperature
    t10 : xarray.DataArray
      10th percentile of daily maximum temperature
    freq : str, optional
      Resampling frequency

    Returns
    -------
    xarray.DataArray
      Count of days with daily maximum temperature below the 10th percentile [days]

    Notes
    -----
    The 10th percentile should be computed for a 5 day window centered on each calendar day for a reference period.

    Example
    -------
    >>> t10 = percentile_doy(historical_tas, per=0.1)
    >>> cold_days = tg10p(tas, t10)
    """
    if 'dayofyear' not in t10.coords.keys():
        raise AttributeError("t10 should have dayofyear coordinates.")

    # adjustment of t10 to tas doy range
    t10 = utils.adjust_doy_calendar(t10, tasmax)

    # create array of percentile with tas shape and coords
    thresh = xr.full_like(tasmax, np.nan)
    doy = thresh.time.dt.dayofyear.values
    thresh.data = t10.sel(dayofyear=doy)

    # compute the cold days
    below = (tasmax < thresh)

    return below.resample(time=freq).sum(dim='time')


def tx_max(tasmax, freq='YS'):
    r"""Highest max temperature

    The maximum value of daily maximum temperature.

    Parameters
    ----------
    tasmax : xarray.DataArray
      Maximum daily temperature [℃] or [K]
    freq : str, optional
      Resampling frequency

    Returns
    -------
    xarray.DataArray
      Maximum value of daily maximum temperature.

    Notes
    -----
    Let :math:`TX_{ij}` be the maximum temperature at day :math:`i` of period :math:`j`. Then the maximum
    daily maximum temperature for period :math:`j` is:

    .. math::

        TXx_j = max(TX_{ij})
    """

    return tasmax.resample(time=freq).max(dim='time')


def tx_mean(tasmax, freq='YS'):
    r"""Mean max temperature

    The mean of daily maximum temperature.

    Parameters
    ----------
    tasmax : xarray.DataArray
      Maximum daily temperature [℃] or [K]
    freq : str, optional
      Resampling frequency

    Returns
    -------
    xarray.DataArray
      Mean of daily maximum temperature.

    Notes
    -----
    Let :math:`TX_{ij}` be the maximum temperature at day :math:`i` of period :math:`j`. Then mean
    values in period :math:`j` are given by:

    .. math::

        TX_{ij} = \frac{ \sum_{i=1}^{I} TX_{ij} }{I}
    """

    arr = tasmax.resample(time=freq) if freq else tasmax
    return arr.mean(dim='time')


def tx_min(tasmax, freq='YS'):
    r"""Lowest max temperature

    The minimum of daily maximum temperature.

    Parameters
    ----------
    tasmax : xarray.DataArray
      Maximum daily temperature [℃] or [K]
    freq : str, optional
      Resampling frequency

    Returns
    -------
    xarray.DataArray
      Minimum of daily maximum temperature.

    Notes
    -----
    Let :math:`TX_{ij}` be the maximum temperature at day :math:`i` of period :math:`j`. Then the minimum
    daily maximum temperature for period :math:`j` is:

    .. math::

        TXn_j = min(TX_{ij})
    """

    return tasmax.resample(time=freq).min(dim='time')


def warm_day_frequency(tasmax, thresh=30, freq='YS'):
    r"""Frequency of extreme warm days

    Return the number of days with tasmax > thresh per period

    Parameters
    ----------
    tasmax : xarray.DataArray
      Mean daily temperature [℃] or [K]
    thresh : float
      Threshold temperature on which to base evaluation [℃] or [K]
    freq : str, optional
      Resampling frequency
    """

    events = (tasmax > thresh) * 1
    return events.resample(time=freq).sum(dim='time')


def tx_tn_days_above(tasmin, tasmax, thresh_tasmin=22,
                     thresh_tasmax=30, freq='YS'):
    r"""Number of days with both hot maximum and minimum daily temperatures.

    The number of days per period with tasmin above a threshold and tasmax above another threshold.

    Parameters
    ----------
    tasmin : xarray.DataArray
      Minimum daily temperature [℃] or [K]
    tasmax : xarray.DataArray
      Maximum daily temperature [℃] or [K]
    thresh_tasmin : float
      Threshold temperature for tasmin on which to base evaluation [℃] or [K]
    thresh_tasmax : float
      Threshold temperature for tasmax on which to base evaluation [℃] or [K]
    freq : str, optional
      Resampling frequency

    Returns
    -------
    xarray.DataArray
      the number of days with tasmin > thresh_tasmin and
      tasmax > thresh_tasamax per period
    """

    events = ((tasmin > (thresh_tasmin + K2C)) & (tasmax > (thresh_tasmax + K2C))) * 1
    return events.resample(time=freq).sum(dim='time')


def warm_night_frequency(tasmin, thresh=22, freq='YS'):
    r"""Frequency of extreme warm nights

    Return the number of days with tasmin > thresh per period

    Parameters
    ----------
    tasmin : xarray.DataArray
      Minimum daily temperature [℃] or [K]
    thresh : float
      Threshold temperature on which to base evaluation [℃] or [K]
    freq : str, optional
      Resampling frequency

    Returns
    -------
    xarray.DataArray
      The number of days with tasmin > thresh per period
    """
    events = (tasmin > thresh) * 1
    return events.resample(time=freq).sum(dim='time')


def warm_spell_duration_index(tasmax, tx90, window=6, freq='YS'):
    r"""Warm spell duration index

    Number of days with at least six consecutive days where the daily maximum temperature is above the 90th
    percentile. The 90th percentile should be computed for a 5-day window centred on each calendar day in the
    1961-1990 period.

    Parameters
    ----------
    tasmax : xarray.DataArray
      Maximum daily temperature [℃] or [K]
    tx90 : float
      90th percentile of daily maximum temperature [K]
    freq : str, optional
      Resampling frequency

    Returns
    -------
    xarray.DataArray
      Count of days with at least six consecutive days where the daily maximum temperature is above the 90th
      percentile [days].

    References
    ----------
    From the Expert Team on Climate Change Detection, Monitoring and Indices (ETCCDMI).
    Used in Alexander, L. V., et al. (2006), Global observed changes in daily climate extremes of temperature and
    precipitation, J. Geophys. Res., 111, D05109, doi: 10.1029/2005JD006290.

    """
    if 'dayofyear' not in tx90.coords.keys():
        raise AttributeError("tx90 should have dayofyear coordinates.")

    # The day of year value of the tasmax series.
    doy = tasmax.indexes['time'].dayofyear

    # Create an array with the shape and coords of tasmax, but with values set to tx90 according to the doy index.
    thresh = xr.full_like(tasmax, np.nan)
    thresh.data = tx90.sel(dayofyear=doy)

    above = (tasmax > thresh)

    return above.resample(time=freq).apply(rl.windowed_run_count, window=window, dim='time')


def wetdays(pr, thresh=1.0, freq='YS'):
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
    with precipitation over 5 mm at the seasonal frequency, ie DJF, MAM, JJA, SON, DJF, etc.:

    >>> pr = xr.open_dataset('pr.day.nc')
    >>> wd = wetdays(pr, pr_min = 5., freq="QS-DEC")
    """

    wd = (pr >= thresh) * 1
    return wd.resample(time=freq).sum(dim='time')


def winter_rain_ratio(pr, prsn=None, tas=None):
    """Ratio of rainfall to total precipitation during winter

    The ratio of total liquid precipitation over the total precipitation over the winter months (DJF. If solid
    precipitation is not provided, then precipitation is assumed solid if the temperature is below 0°C.

    Parameters
    ----------
    pr : xarray.DataArray
      Mean daily precipitation flux [Kg m-2 s-1] or [mm].
    prsn : xarray.DataArray
      Mean daily solid precipitation flux [Kg m-2 s-1] or [mm].
    tas : xarray.DataArray
      Mean daily temperature [℃] or [K]
    freq : str
      Resampling frequency

    Returns
    -------
    xarray.DataArray
      Ratio of rainfall to total precipitation during winter months (DJF)
    """
    ratio = liquid_precip_ratio(pr, prsn, tas, freq='QS-DEC')
    winter = ratio.indexes['time'].month == 12
    return ratio[winter]

import logging

import numpy as np
import xarray as xr

from xclim import utils, run_length as rl
from xclim.utils import declare_units, units

logging.basicConfig(level=logging.DEBUG)
logging.captureWarnings(True)

xr.set_options(enable_cftimeindex=True)  # Set xarray to use cftimeindex


# Frequencies : YS: year start, QS-DEC: seasons starting in december, MS: month start
# See http://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases

# -------------------------------------------------- #
# ATTENTION: ASSUME ALL INDICES WRONG UNTIL TESTED ! #
# -------------------------------------------------- #

__all__ = ['cold_spell_duration_index', 'cold_and_dry_days', 'daily_freezethaw_cycles', 'daily_temperature_range',
           'daily_temperature_range_variability', 'extreme_temperature_range', 'heat_wave_frequency',
           'heat_wave_max_length', 'liquid_precip_ratio', 'rain_on_frozen_ground_days', 'tg90p', 'tg10p',
           'tn90p', 'tn10p', 'tx90p', 'tx10p', 'tx_tn_days_above', 'warm_spell_duration_index', 'winter_rain_ratio']


@declare_units('days', tasmin='[temperature]', tn10='[temperature]')
def cold_spell_duration_index(tasmin, tn10, window=6, freq='YS'):
    r"""Cold spell duration index

    Number of days with at least six consecutive days where the daily minimum temperature is below the 10th
    percentile.

    Parameters
    ----------
    tasmin : xarray.DataArray
      Minimum daily temperature.
    tn10 : float
      10th percentile of daily minimum temperature.
    window : int
      Minimum number of days with temperature below threshold to qualify as a cold spell. Default: 6.
    freq : str, optional
      Resampling frequency

    Returns
    -------
    xarray.DataArray
      Count of days with at least six consecutive days where the daily minimum temperature is below the 10th
      percentile [days].

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

    tn10 = utils.convert_units_to(tn10, tasmin)
    # If calendar of `tn10` is different from `tasmin`, interpolate.
    tn10 = utils.adjust_doy_calendar(tn10, tasmin)

    # Create an array with the shape and coords of tasmin, but with values set to tx90 according to the doy index.
    thresh = xr.full_like(tasmin, np.nan)
    thresh.data = tn10.sel(dayofyear=doy)

    below = (tasmin < thresh)

    return below.resample(time=freq).apply(rl.windowed_run_count, window=window, dim='time')


def cold_and_dry_days(tas, tgin25, pr, wet25, freq='YS'):
    r"""Cold and dry days.

    Returns the total number of days where "Cold" and "Dry" conditions coincide.

    Parameters
    ----------
    tas : xarray.DataArray
      Mean daily temperature values [℃] or [K]
    tgin25 : xarray.DataArray
      First quartile of daily mean temperature computed by month.
    pr : xarray.DataArray
      Daily precipitation.
    wet25 : xarray.DataArray
      First quartile of daily total precipitation computed by month.
    freq : str, optional
      Resampling frequency

    Returns
    -------
    xarray.DataArray
      The total number of days where cold and dry conditions coincide.

    Notes
    -----
    Formula to be written [cold_dry_days]_.

    References
    ----------
    .. [cold_dry_days] Beniston, M. (2009). Trends in joint quantiles of temperature and precipitation in Europe
        since 1901 and projected for 2100. Geophysical Research Letters, 36(7). https://doi.org/10.1029/2008GL037119
    """
    raise NotImplementedError
    # There is an issue with the 1 mm threshold. It makes no sense to assume a day with < 1mm is not dry.
    # c1 = tas < utils.convert_units_to(tgin25, tas)
    # c2 = (pr > utils.convert_units_to('1 mm', pr)) * (pr < utils.convert_units_to(wet25, pr))

    # c = (c1 * c2) * 1
    # return c.resample(time=freq).sum(dim='time')


@declare_units('days', tasmax='[temperature]', tasmin='[temperature]')
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

    Notes
    -----
    Let :math:`TX_{i}` be the maximum temperature at day :math:`i` and :math:`TN_{i}` be
    the daily minimum temperature at day :math:`i`. Then the number of freeze thaw cycles
    during period :math:`\phi` is given by :

    .. math::

        \sum_{i \in \phi} [ TX_{i} > 0℃ ] [ TN_{i} <  0℃ ]

    where :math:`[P]` is 1 if :math:`P` is true, and 0 if false.
    """
    frz = utils.convert_units_to('0 degC', tasmax)
    ft = (tasmin < frz) * (tasmax > frz) * 1
    out = ft.resample(time=freq).sum(dim='time')
    return out


@declare_units('K', tasmax='[temperature]', tasmin='[temperature]')
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
    out = dtr.resample(time=freq).mean(dim='time', keep_attrs=True)
    out.attrs['units'] = tasmax.units
    return out


@declare_units('K', tasmax='[temperature]', tasmin='[temperature]')
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
    out = vdtr.resample(time=freq).mean(dim='time')
    out.attrs['units'] = tasmax.units
    return out


@declare_units('K', tasmax='[temperature]', tasmin='[temperature]')
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

    out = tx_max - tn_min
    out.attrs['units'] = tasmax.units
    return out


@declare_units('', tasmin='[temperature]', tasmax='[temperature]', thresh_tasmin='[temperature]',
               thresh_tasmax='[temperature]')
def heat_wave_frequency(tasmin, tasmax, thresh_tasmin='22.0 degC', thresh_tasmax='30 degC',
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
    thresh_tasmin : str
      The minimum temperature threshold needed to trigger a heatwave event [℃] or [K]. Default : '22 degC'
    thresh_tasmax : str
      The maximum temperature threshold needed to trigger a heatwave event [℃] or [K]. Default : '30 degC'
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
    thresh_tasmax = utils.convert_units_to(thresh_tasmax, tasmax)
    thresh_tasmin = utils.convert_units_to(thresh_tasmin, tasmin)

    cond = (tasmin > thresh_tasmin) & (tasmax > thresh_tasmax)
    group = cond.resample(time=freq)
    return group.apply(rl.windowed_run_events, window=window, dim='time')


@declare_units('days', tasmin='[temperature]', tasmax='[temperature]', thresh_tasmin='[temperature]',
               thresh_tasmax='[temperature]')
def heat_wave_max_length(tasmin, tasmax, thresh_tasmin='22.0 degC', thresh_tasmax='30 degC',
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
    thresh_tasmin : str
      The minimum temperature threshold needed to trigger a heatwave event [℃] or [K]. Default : '22 degC'
    thresh_tasmax : str
      The maximum temperature threshold needed to trigger a heatwave event [℃] or [K]. Default : '30 degC'
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
    thresh_tasmax = utils.convert_units_to(thresh_tasmax, tasmax)
    thresh_tasmin = utils.convert_units_to(thresh_tasmin, tasmin)

    cond = (tasmin > thresh_tasmin) & (tasmax > thresh_tasmax)
    group = cond.resample(time=freq)
    max_l = group.apply(rl.longest_run, dim='time')
    return max_l.where(max_l >= window, 0)


@declare_units('', pr='[precipitation]', prsn='[precipitation]', tas='[temperature]')
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

    Notes
    -----
    Let :math:`PR_i` be the mean daily precipitation of day :math:`i`, then for a period :math:`j` starting at
    day :math:`a` and finishing on day :math:`b`:

    .. math::

        PR_{ij} = \sum_{i=a}^{b} PR_i


        PRwet_{ij}

    See also
    --------
    winter_rain_ratio
    """

    if prsn is None:
        tu = units.parse_units(tas.attrs['units'].replace('-', '**-'))
        fu = 'degC'
        frz = 0
        if fu != tu:
            frz = units.convert(frz, fu, tu)
        prsn = pr.where(tas < frz, 0)

    tot = pr.resample(time=freq).sum(dim='time')
    rain = tot - prsn.resample(time=freq).sum(dim='time')
    ratio = rain / tot
    return ratio


@declare_units('days', pr='[precipitation]', tas='[temperature]', thresh='[precipitation]')
def rain_on_frozen_ground_days(pr, tas, thresh='1 mm/d', freq='YS'):
    """Number of rain on frozen ground events

    Number of days with rain above a threshold after a series of seven days below freezing temperature.
    Precipitation is assumed to be rain when the temperature is above 0℃.

    Parameters
    ----------
    pr : xarray.DataArray
      Mean daily precipitation flux [Kg m-2 s-1] or [mm]
    tas : xarray.DataArray
      Mean daily temperature [℃] or [K]
    thresh : str
      Precipitation threshold to consider a day as a rain event. Default : '1 mm/d'
    freq : str, optional
      Resampling frequency

    Returns
    -------
    xarray.DataArray
      The number of rain on frozen ground events per period [days]

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
    t = utils.convert_units_to(thresh, pr)
    frz = utils.convert_units_to('0 C', tas)

    def func(x, axis):
        """Check that temperature conditions are below 0 for seven days and above after."""
        frozen = x == np.array([0, 0, 0, 0, 0, 0, 0, 1], bool)
        return frozen.all(axis=axis)

    tcond = (tas > frz).rolling(time=8).reduce(func)
    pcond = (pr > t)

    return (tcond * pcond * 1).resample(time=freq).sum(dim='time')


@declare_units('days', tas='[temperature]', t90='[temperature]')
def tg90p(tas, t90, freq='YS'):
    r"""Number of days with daily mean temperature over the 90th percentile.

    Number of days with daily mean temperature over the 90th percentile.

    Parameters
    ----------
    tas : xarray.DataArray
      Mean daily temperature [℃] or [K]
    t90 : xarray.DataArray
      90th percentile of daily mean temperature [℃] or [K]
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

    t90 = utils.convert_units_to(t90, tas)

    # adjustment of t90 to tas doy range
    t90 = utils.adjust_doy_calendar(t90, tas)

    # create array of percentile with tas shape and coords
    thresh = xr.full_like(tas, np.nan)
    doy = thresh.time.dt.dayofyear.values
    thresh.data = t90.sel(dayofyear=doy)

    # compute the cold days
    over = (tas > thresh)

    return over.resample(time=freq).sum(dim='time')


@declare_units('days', tas='[temperature]', t10='[temperature]')
def tg10p(tas, t10, freq='YS'):
    r"""Number of days with daily mean temperature below the 10th percentile.

    Number of days with daily mean temperature below the 10th percentile.

    Parameters
    ----------
    tas : xarray.DataArray
      Mean daily temperature [℃] or [K]
    t10 : xarray.DataArray
      10th percentile of daily mean temperature [℃] or [K]
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

    t10 = utils.convert_units_to(t10, tas)

    # adjustment of t10 to tas doy range
    t10 = utils.adjust_doy_calendar(t10, tas)

    # create array of percentile with tas shape and coords
    thresh = xr.full_like(tas, np.nan)
    doy = thresh.time.dt.dayofyear.values
    thresh.data = t10.sel(dayofyear=doy)

    # compute the cold days
    below = (tas < thresh)

    return below.resample(time=freq).sum(dim='time')


@declare_units('days', tasmin='[temperature]', t90='[temperature]')
def tn90p(tasmin, t90, freq='YS'):
    r"""Number of days with daily minimum temperature over the 90th percentile.

    Number of days with daily minimum temperature over the 90th percentile.

    Parameters
    ----------
    tasmin : xarray.DataArray
      Minimum daily temperature [℃] or [K]
    t90 : xarray.DataArray
      90th percentile of daily minimum temperature [℃] or [K]
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
    t90 = utils.convert_units_to(t90, tasmin)
    # adjustment of t90 to tas doy range
    t90 = utils.adjust_doy_calendar(t90, tasmin)

    # create array of percentile with tas shape and coords
    thresh = xr.full_like(tasmin, np.nan)
    doy = thresh.time.dt.dayofyear.values
    thresh.data = t90.sel(dayofyear=doy)

    # compute the cold days
    over = (tasmin > thresh)

    return over.resample(time=freq).sum(dim='time')


@declare_units('days', tasmin='[temperature]', t10='[temperature]')
def tn10p(tasmin, t10, freq='YS'):
    r"""Number of days with daily minimum temperature below the 10th percentile.

    Number of days with daily minimum temperature below the 10th percentile.

    Parameters
    ----------

    tasmin : xarray.DataArray
      Mean daily temperature [℃] or [K]
    t10 : xarray.DataArray
      10th percentile of daily minimum temperature [℃] or [K]
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
    t10 = utils.convert_units_to(t10, tasmin)

    # adjustment of t10 to tas doy range
    t10 = utils.adjust_doy_calendar(t10, tasmin)

    # create array of percentile with tas shape and coords
    thresh = xr.full_like(tasmin, np.nan)
    doy = thresh.time.dt.dayofyear.values
    thresh.data = t10.sel(dayofyear=doy)

    # compute the cold days
    below = (tasmin < thresh)

    return below.resample(time=freq).sum(dim='time')


@declare_units('days', tasmax='[temperature]', t90='[temperature]')
def tx90p(tasmax, t90, freq='YS'):
    r"""Number of days with daily maximum temperature over the 90th percentile.

    Number of days with daily maximum temperature over the 90th percentile.

    Parameters
    ----------
    tasmax : xarray.DataArray
      Maximum daily temperature [℃] or [K]
    t90 : xarray.DataArray
      90th percentile of daily maximum temperature [℃] or [K]
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

    t90 = utils.convert_units_to(t90, tasmax)

    # adjustment of t90 to tas doy range
    t90 = utils.adjust_doy_calendar(t90, tasmax)

    # create array of percentile with tas shape and coords
    thresh = xr.full_like(tasmax, np.nan)
    doy = thresh.time.dt.dayofyear.values
    thresh.data = t90.sel(dayofyear=doy)

    # compute the cold days
    over = (tasmax > thresh)

    return over.resample(time=freq).sum(dim='time')


@declare_units('days', tasmax='[temperature]', t10='[temperature]')
def tx10p(tasmax, t10, freq='YS'):
    r"""Number of days with daily maximum temperature below the 10th percentile.

    Number of days with daily maximum temperature below the 10th percentile.

    Parameters
    ----------
    tasmax : xarray.DataArray
      Maximum daily temperature [℃] or [K]
    t10 : xarray.DataArray
      10th percentile of daily maximum temperature [℃] or [K]
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

    t10 = utils.convert_units_to(t10, tasmax)

    # adjustment of t10 to tas doy range
    t10 = utils.adjust_doy_calendar(t10, tasmax)

    # create array of percentile with tas shape and coords
    thresh = xr.full_like(tasmax, np.nan)
    doy = thresh.time.dt.dayofyear.values
    thresh.data = t10.sel(dayofyear=doy)

    # compute the cold days
    below = (tasmax < thresh)

    return below.resample(time=freq).sum(dim='time')


@declare_units('days', tasmin='[temperature]', tasmax='[temperature]', thresh_tasmin='[temperature]',
               thresh_tasmax='[temperature]')
def tx_tn_days_above(tasmin, tasmax, thresh_tasmin='22 degC',
                     thresh_tasmax='30 degC', freq='YS'):
    r"""Number of days with both hot maximum and minimum daily temperatures.

    The number of days per period with tasmin above a threshold and tasmax above another threshold.

    Parameters
    ----------
    tasmin : xarray.DataArray
      Minimum daily temperature [℃] or [K]
    tasmax : xarray.DataArray
      Maximum daily temperature [℃] or [K]
    thresh_tasmin : str
      Threshold temperature for tasmin on which to base evaluation [℃] or [K]. Default : '22 degC'
    thresh_tasmax : str
      Threshold temperature for tasmax on which to base evaluation [℃] or [K]. Default : '30 degC'
    freq : str, optional
      Resampling frequency

    Returns
    -------
    xarray.DataArray
      the number of days with tasmin > thresh_tasmin and
      tasmax > thresh_tasamax per period


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
    thresh_tasmax = utils.convert_units_to(thresh_tasmax, tasmax)
    thresh_tasmin = utils.convert_units_to(thresh_tasmin, tasmin)
    events = ((tasmin > (thresh_tasmin)) & (tasmax > (thresh_tasmax))) * 1
    return events.resample(time=freq).sum(dim='time')


@declare_units('days', tasmax='[temperature]', tx90='[temperature]')
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
      90th percentile of daily maximum temperature [℃] or [K]
    window : int
      Minimum number of days with temperature below threshold to qualify as a warm spell.
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

    # adjustment of tx90 to tasmax doy range
    tx90 = utils.adjust_doy_calendar(tx90, tasmax)

    # Create an array with the shape and coords of tasmax, but with values set to tx90 according to the doy index.
    thresh = xr.full_like(tasmax, np.nan)
    thresh.data = tx90.sel(dayofyear=doy)

    above = (tasmax > thresh)

    return above.resample(time=freq).apply(rl.windowed_run_count, window=window, dim='time')


@declare_units('', pr='[precipitation]', prsn='[precipitation]', tas='[temperature]')
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

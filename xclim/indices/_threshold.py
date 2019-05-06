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

__all__ = ['cold_spell_days', 'daily_pr_intensity', 'maximum_consecutive_wet_days', 'cooling_degree_days',
           'freshet_start', 'growing_degree_days', 'growing_season_length', 'heat_wave_index', 'heating_degree_days',
           'tn_days_below', 'tx_days_above', 'warm_day_frequency', 'warm_night_frequency', 'wetdays',
           'maximum_consecutive_dry_days', 'max_n_day_precipitation_amount', 'tropical_nights']


@declare_units('days', tas='[temperature]', thresh='[temperature]')
def cold_spell_days(tas, thresh='-10 degC', window=5, freq='AS-JUL'):
    r"""Cold spell days

    The number of days that are part of a cold spell, defined as five or more consecutive days with mean daily
    temperature below a threshold in °C.

    Parameters
    ----------
    tas : xarrray.DataArray
      Mean daily temperature [℃] or [K]
    thresh : str
      Threshold temperature below which a cold spell begins [℃] or [K]. Default : '-10 degC'
    window : int
      Minimum number of days with temperature below threshold to qualify as a cold spell.
    freq : str, optional
      Resampling frequency

    Returns
    -------
    xarray.DataArray
      Cold spell days.

    Notes
    -----
    Let :math:`T_i` be the mean daily temperature on day :math:`i`, the number of cold spell days during
    period :math:`\phi` is given by

    .. math::

       \sum_{i \in \phi} \prod_{j=i}^{i+5} [T_j < thresh]

    where :math:`[P]` is 1 if :math:`P` is true, and 0 if false.

    """
    t = utils.convert_units_to(thresh, tas)
    over = tas < t
    group = over.resample(time=freq)

    return group.apply(rl.windowed_run_count, window=window, dim='time')


@declare_units('mm/day', pr='[precipitation]', thresh='[precipitation]')
def daily_pr_intensity(pr, thresh='1 mm/day', freq='YS'):
    r"""Average daily precipitation intensity

    Return the average precipitation over wet days.

    Parameters
    ----------
    pr : xarray.DataArray
      Daily precipitation [mm/d or kg/m²/s]
    thresh : str
      precipitation value over which a day is considered wet. Default : '1 mm/day'
    freq : str, optional
      Resampling frequency defining the periods
      defined in http://pandas.pydata.org/pandas-docs/stable/timeseries.html#resampling. Default : '1 mm/day'

    Returns
    -------
    xarray.DataArray
      The average precipitation over wet days for each period

    Notes
    -----
    Let :math:`\mathbf{p} = p_0, p_1, \ldots, p_n` be the daily precipitation and :math:`thresh` be the precipitation
    threshold defining wet days. Then the daily precipitation intensity is defined as

    .. math::

       \frac{\sum_{i=0}^n p_i [p_i \leq thresh]}{\sum_{i=0}^n [p_i \leq thresh]}

    where :math:`[P]` is 1 if :math:`P` is true, and 0 if false.

    Examples
    --------
    The following would compute for each grid cell of file `pr.day.nc` the average
    precipitation fallen over days with precipitation >= 5 mm at seasonal
    frequency, ie DJF, MAM, JJA, SON, DJF, etc.:

    >>> pr = xr.open_dataset('pr.day.nc')
    >>> daily_int = daily_pr_intensity(pr, thresh='5 mm/day', freq="QS-DEC")

    """
    t = utils.convert_units_to(thresh, pr, 'hydro')

    # put pr=0 for non wet-days
    pr_wd = xr.where(pr >= t, pr, 0)
    pr_wd.attrs['units'] = pr.units

    # sum over wanted period
    s = pr_wd.resample(time=freq).sum(dim='time', keep_attrs=True)
    sd = utils.pint_multiply(s, 1 * units.day, 'mm')

    # get number of wetdays over period
    wd = wetdays(pr, thresh=thresh, freq=freq)
    return sd / wd


@declare_units('days', pr='[precipitation]', thresh='[precipitation]')
def maximum_consecutive_wet_days(pr, thresh='1 mm/day', freq='YS'):
    r"""Consecutive wet days.

    Returns the maximum number of consecutive wet days.

    Parameters
    ---------
    pr : xarray.DataArray
      Mean daily precipitation flux [Kg m-2 s-1] or [mm]
    thresh : str
      Threshold precipitation on which to base evaluation [Kg m-2 s-1] or [mm]. Default : '1 mm/day'
    freq : str, optional
      Resampling frequency

    Returns
    -------
    xarray.DataArray
      The maximum number of consecutive wet days.

    Notes
    -----
    Let :math:`\mathbf{x}=x_0, x_1, \ldots, x_n` be a daily precipitation series and
    :math:`\mathbf{s}` be the sorted vector of indices :math:`i` where :math:`[p_i > thresh] \neq [p_{i+1} >
    thresh]`, that is, the days when the precipitation crosses the *wet day* threshold.
    Then the maximum number of consecutive wet days is given by

    .. math::


       \max(\mathbf{d}) \quad \mathrm{where} \quad d_j = (s_j - s_{j-1}) [x_{s_j} > 0\celsius]

    where :math:`[P]` is 1 if :math:`P` is true, and 0 if false. Note that this formula does not handle sequences at
    the start and end of the series, but the numerical algorithm does.
    """
    thresh = utils.convert_units_to(thresh, pr, 'hydro')

    group = (pr > thresh).resample(time=freq)
    return group.apply(rl.longest_run, dim='time')


@declare_units('C days', tas='[temperature]', thresh='[temperature]')
def cooling_degree_days(tas, thresh='18 degC', freq='YS'):
    r"""Cooling degree days

    Sum of degree days above the temperature threshold at which spaces are cooled.

    Parameters
    ----------
    tas : xarray.DataArray
      Mean daily temperature [℃] or [K]
    thresh : str
      Temperature threshold above which air is cooled. Default : '18 degC'
    freq : str, optional
      Resampling frequency

    Returns
    -------
    xarray.DataArray
      Cooling degree days

    Notes
    -----
    Let :math:`x_i` be the daily mean temperature at day :math:`i`. Then the cooling degree days above
    temperature threshold :math:`thresh` over period :math:`\phi` is given by:

    .. math::

        \sum_{i \in \phi} (x_{i}-{thresh} [x_i > thresh]

    where :math:`[P]` is 1 if :math:`P` is true, and 0 if false.
    """
    thresh = utils.convert_units_to(thresh, tas)

    return tas.pipe(lambda x: x - thresh) \
        .clip(min=0) \
        .resample(time=freq) \
        .sum(dim='time')


@declare_units('', tas='[temperature]', thresh='[temperature]')
def freshet_start(tas, thresh='0 degC', window=5, freq='YS'):
    r"""First day consistently exceeding threshold temperature.

    Returns first day of period where a temperature threshold is exceeded
    over a given number of days.

    Parameters
    ----------
    tas : xarray.DataArray
      Mean daily temperature [℃] or [K]
    thresh : str
      Threshold temperature on which to base evaluation [℃] or [K]. Default '0 degC'
    window : int
      Minimum number of days with temperature above threshold needed for evaluation
    freq : str, optional
      Resampling frequency

    Returns
    -------
    float
      Day of the year when temperature exceeds threshold over a given number of days for the first time. If there are
      no such day, return np.nan.

    Notes
    -----
    Let :math:`x_i` be the daily mean temperature at day of the year :math:`i` for values of :math:`i` going from 1
    to 365 or 366. The start date of the freshet is given by the smallest index :math:`i` for which

    .. math::

       \prod_{j=i}^{i+w} [x_j > thresh]

    is true, where :math:`w` is the number of days the temperature threshold should be exceeded,  and :math:`[P]` is
    1 if :math:`P` is true, and 0 if false.
    """
    thresh = utils.convert_units_to(thresh, tas)
    over = (tas > thresh)
    group = over.resample(time=freq)
    return group.apply(rl.first_run_ufunc, window=window, index='dayofyear')


@declare_units('C days', tas='[temperature]', thresh='[temperature]')
def growing_degree_days(tas, thresh='4.0 degC', freq='YS'):
    r"""Growing degree-days over threshold temperature value [℃].

    The sum of degree-days over the threshold temperature.

    Parameters
    ---------
    tas : xarray.DataArray
      Mean daily temperature [℃] or [K]
    thresh : str
      Threshold temperature on which to base evaluation [℃] or [K]. Default: '4.0 degC'.
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
    thresh = utils.convert_units_to(thresh, tas)
    return tas.pipe(lambda x: x - thresh) \
        .clip(min=0) \
        .resample(time=freq) \
        .sum(dim='time')


@declare_units('days', tas='[temperature]', thresh='[temperature]')
def growing_season_length(tas, thresh='5.0 degC', window=6, freq='YS'):
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
    thresh : str
      Threshold temperature on which to base evaluation [℃] or [K]. Default: '5.0 degC'.
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
    # c = ((tas > thresh) * 1).rolling(time=window).sum()
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
    thresh = utils.convert_units_to(thresh, tas)

    c = ((tas > thresh) * 1).rolling(time=window).sum().chunk(tas.chunks)

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


@declare_units('days', tasmax='[temperature]', thresh='[temperature]')
def heat_wave_index(tasmax, thresh='25.0 degC', window=5, freq='YS'):
    r"""Heat wave index.

    Number of days that are part of a heatwave, defined as five or more consecutive days over 25℃.

    Parameters
    ----------
    tasmax : xarrray.DataArray
      Maximum daily temperature [℃] or [K]
    thresh : str
      Threshold temperature on which to designate a heatwave [℃] or [K]. Default: '25.0 degC'.
    window : int
      Minimum number of days with temperature above threshold to qualify as a heatwave.
    freq : str, optional
      Resampling frequency

    Returns
    -------
    DataArray
      Heat wave index.
    """
    thresh = utils.convert_units_to(thresh, tasmax)
    over = tasmax > thresh
    group = over.resample(time=freq)

    return group.apply(rl.windowed_run_count, window=window, dim='time')


@declare_units('C days', tas='[temperature]', thresh='[temperature]')
def heating_degree_days(tas, thresh='17.0 degC', freq='YS'):
    r"""Heating degree days

    Sum of degree days below the temperature threshold at which spaces are heated.

    Parameters
    ----------
    tas : xarray.DataArray
      Mean daily temperature [℃] or [K]
    thresh : str
      Threshold temperature on which to base evaluation [℃] or [K]. Default: '17.0 degC'.
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
    thresh = utils.convert_units_to(thresh, tas)

    return tas.pipe(lambda x: thresh - x) \
        .clip(0) \
        .resample(time=freq) \
        .sum(dim='time')


@declare_units('days', tasmin='[temperature]', thresh='[temperature]')
def tn_days_below(tasmin, thresh='-10.0 degC', freq='YS'):
    r"""Number of days with tmin below a threshold in

    Number of days where daily minimum temperature is below a threshold.

    Parameters
    ----------
    tasmin : xarray.DataArray
      Minimum daily temperature [℃] or [K]
    thresh : str
      Threshold temperature on which to base evaluation [℃] or [K] . Default: '-10 degC'.
    freq : str, optional
      Resampling frequency

    Returns
    -------
    xarray.DataArray
      Number of days Tmin < threshold.

    Notes
    -----
    Let :math:`TN_{ij}` be the daily minimum temperature at day :math:`i` of period :math:`j`. Then
    counted is the number of days where:

    .. math::

        TX_{ij} < Threshold [℃]
    """
    thresh = utils.convert_units_to(thresh, tasmin)
    f1 = utils.threshold_count(tasmin, '<', thresh, freq)
    return f1


@declare_units('days', tasmax='[temperature]', thresh='[temperature]')
def tx_days_above(tasmax, thresh='25.0 degC', freq='YS'):
    r"""Number of summer days

    Number of days where daily maximum temperature exceed a threshold.

    Parameters
    ----------
    tasmax : xarray.DataArray
      Maximum daily temperature [℃] or [K]
    thresh : str
      Threshold temperature on which to base evaluation [℃] or [K]. Default: '25 degC'.
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

        TX_{ij} > Threshold [℃]
    """
    thresh = utils.convert_units_to(thresh, tasmax)
    f = (tasmax > (thresh)) * 1
    return f.resample(time=freq).sum(dim='time')


@declare_units('days', tasmax='[temperature]', thresh='[temperature]')
def warm_day_frequency(tasmax, thresh='30 degC', freq='YS'):
    r"""Frequency of extreme warm days

    Return the number of days with tasmax > thresh per period

    Parameters
    ----------
    tasmax : xarray.DataArray
      Mean daily temperature [℃] or [K]
    thresh : str
      Threshold temperature on which to base evaluation [℃] or [K]. Default : '30 degC'
    freq : str, optional
      Resampling frequency

    Returns
    -------
    xarray.DataArray
      Number of days exceeding threshold.

    Notes:
    Let :math:`TX_{ij}` be the daily maximum temperature at day :math:`i` of period :math:`j`. Then
    counted is the number of days where:

    .. math::

        TN_{ij} > Threshold [℃]

    """
    thresh = utils.convert_units_to(thresh, tasmax)
    events = (tasmax > thresh) * 1
    return events.resample(time=freq).sum(dim='time')


@declare_units('days', tasmin='[temperature]', thresh='[temperature]')
def warm_night_frequency(tasmin, thresh='22 degC', freq='YS'):
    r"""Frequency of extreme warm nights

    Return the number of days with tasmin > thresh per period

    Parameters
    ----------
    tasmin : xarray.DataArray
      Minimum daily temperature [℃] or [K]
    thresh : str
      Threshold temperature on which to base evaluation [℃] or [K]. Default : '22 degC'
    freq : str, optional
      Resampling frequency

    Returns
    -------
    xarray.DataArray
      The number of days with tasmin > thresh per period
    """
    thresh = utils.convert_units_to(thresh, tasmin, )
    events = (tasmin > thresh) * 1
    return events.resample(time=freq).sum(dim='time')


@declare_units('days', pr='[precipitation]', thresh='[precipitation]')
def wetdays(pr, thresh='1.0 mm/day', freq='YS'):
    r"""Wet days

    Return the total number of days during period with precipitation over threshold.

    Parameters
    ----------
    pr : xarray.DataArray
      Daily precipitation [mm]
    thresh : str
      Precipitation value over which a day is considered wet. Default: '1 mm/day'.
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
    thresh = utils.convert_units_to(thresh, pr, 'hydro')

    wd = (pr >= thresh) * 1
    return wd.resample(time=freq).sum(dim='time')


@declare_units('days', pr='[precipitation]', thresh='[precipitation]')
def maximum_consecutive_dry_days(pr, thresh='1 mm/day', freq='YS'):
    r"""Maximum number of consecutive dry days

    Return the maximum number of consecutive days within the period where precipitation
    is below a certain threshold.

    Parameters
    ----------
    pr : xarray.DataArray
      Mean daily precipitation flux [mm]
    thresh : str
      Threshold precipitation on which to base evaluation [mm]. Default : '1 mm/day'
    freq : str, optional
      Resampling frequency

    Returns
    -------
    xarray.DataArray
      The maximum number of consecutive dry days.

    Notes
    -----
    Let :math:`\mathbf{p}=p_0, p_1, \ldots, p_n` be a daily precipitation series and :math:`thresh` the threshold
    under which a day is considered dry. Then let :math:`\mathbf{s}` be the sorted vector of indices :math:`i` where
    :math:`[p_i < thresh] \neq [p_{i+1} < thresh]`, that is, the days when the temperature crosses the threshold.
    Then the maximum number of consecutive dry days is given by

    .. math::

       \max(\mathbf{d}) \quad \mathrm{where} \quad d_j = (s_j - s_{j-1}) [p_{s_j} > thresh]

    where :math:`[P]` is 1 if :math:`P` is true, and 0 if false. Note that this formula does not handle sequences at
    the start and end of the series, but the numerical algorithm does.
    """
    t = utils.convert_units_to(thresh, pr, 'hydro')
    group = (pr < t).resample(time=freq)

    return group.apply(rl.longest_run, dim='time')


@declare_units('mm', pr='[precipitation]')
def max_n_day_precipitation_amount(pr, window=1, freq='YS'):
    r"""Highest precipitation amount cumulated over a n-day moving window.

    Calculate the n-day rolling sum of the original daily total precipitation series
    and determine the maximum value over each period.

    Parameters
    ----------
    da : xarray.DataArray
      Daily precipitation values [Kg m-2 s-1] or [mm]
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
    out = arr.resample(time=freq).max(dim='time', keep_attrs=True)

    out.attrs['units'] = pr.units
    # Adjust values and units to make sure they are daily
    return utils.pint_multiply(out, 1 * units.day, 'mm')


@declare_units('days', tasmin='[temperature]', thresh='[temperature]')
def tropical_nights(tasmin, thresh='20.0 degC', freq='YS'):
    r"""Tropical nights

    The number of days with minimum daily temperature above threshold.

    Parameters
    ----------
    tasmin : xarray.DataArray
      Minimum daily temperature [℃] or [K]
    thresh : str
      Threshold temperature on which to base evaluation [℃] or [K]. Default: '20 degC'.
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

        TN_{ij} > Threshold [℃]
    """
    thresh = utils.convert_units_to(thresh, tasmin)
    return tasmin.pipe(lambda x: (tasmin > thresh) * 1) \
        .resample(time=freq) \
        .sum(dim='time')

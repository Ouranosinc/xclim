# noqa: D100
import warnings
from typing import Optional

import numpy as np
import xarray

from xclim.core.calendar import get_calendar
from xclim.core.units import (
    convert_units_to,
    declare_units,
    rate2amount,
    str2pint,
    to_agg_units,
)
from xclim.core.utils import DayOfYearStr

from . import run_length as rl
from .generic import domain_count, threshold_count

# Frequencies : YS: year start, QS-DEC: seasons starting in december, MS: month start
# See http://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases

# -------------------------------------------------- #
# ATTENTION: ASSUME ALL INDICES WRONG UNTIL TESTED ! #
# -------------------------------------------------- #

__all__ = [
    "calm_days",
    "cold_spell_days",
    "cold_spell_frequency",
    "daily_pr_intensity",
    "degree_days_exceedance_date",
    "cooling_degree_days",
    "continuous_snow_cover_end",
    "continuous_snow_cover_start",
    "days_with_snow",
    "freshet_start",
    "growing_degree_days",
    "growing_season_end",
    "growing_season_length",
    "last_spring_frost",
    "frost_season_length",
    "first_day_below",
    "first_day_above",
    "first_snowfall",
    "last_snowfall",
    "heat_wave_index",
    "heating_degree_days",
    "hot_spell_frequency",
    "hot_spell_max_length",
    "snow_cover_duration",
    "tn_days_above",
    "tn_days_below",
    "tg_days_above",
    "tg_days_below",
    "tx_days_above",
    "tx_days_below",
    "tropical_nights",
    "warm_day_frequency",
    "warm_night_frequency",
    "wetdays",
    "winter_storm",
    "dry_days",
    "maximum_consecutive_dry_days",
    "maximum_consecutive_frost_days",
    "maximum_consecutive_frost_free_days",
    "maximum_consecutive_tx_days",
    "maximum_consecutive_wet_days",
    "sea_ice_area",
    "sea_ice_extent",
    "windy_days",
]


@declare_units(sfcWind="[speed]", thresh="[speed]")
def calm_days(
    sfcWind: xarray.DataArray, thresh: str = "2 m s-1", freq: str = "MS"
) -> xarray.DataArray:
    r"""Calm days.

    The number of days with average near-surface wind speed below threshold.

    Parameters
    ----------
    sfcWind : xarray.DataArray
      Daily windspeed.
    thresh : str
      Threshold average near-surface wind speed on which to base evaluation.
    freq : str
      Resampling frequency.

    Returns
    -------
    xarray.DataArray, [time]
      Number of days with average near-surface wind speed below threshold.

    Notes
    -----
    Let :math:`WS_{ij}` be the windspeed at day :math:`i` of period :math:`j`. Then
    counted is the number of days where:

    .. math::

        WS_{ij} < Threshold [m s-1]
    """
    thresh = convert_units_to(thresh, sfcWind)
    out = threshold_count(sfcWind, "<", thresh, freq)
    out = to_agg_units(out, sfcWind, "count")
    return out


@declare_units(tas="[temperature]", thresh="[temperature]")
def cold_spell_days(
    tas: xarray.DataArray,
    thresh: str = "-10 degC",
    window: int = 5,
    freq: str = "AS-JUL",
) -> xarray.DataArray:
    r"""Cold spell days.

    The number of days that are part of cold spell events, defined as a sequence of consecutive days with mean daily
    temperature below a threshold in °C.

    Parameters
    ----------
    tas : xarray.DataArray
      Mean daily temperature.
    thresh : str
      Threshold temperature below which a cold spell begins.
    window : int
      Minimum number of days with temperature below threshold to qualify as a cold spell.
    freq : str
      Resampling frequency.

    Returns
    -------
    xarray.DataArray, [time]
      Cold spell days.

    Notes
    -----
    Let :math:`T_i` be the mean daily temperature on day :math:`i`, the number of cold spell days during
    period :math:`\phi` is given by

    .. math::

       \sum_{i \in \phi} \prod_{j=i}^{i+5} [T_j < thresh]

    where :math:`[P]` is 1 if :math:`P` is true, and 0 if false.

    """
    t = convert_units_to(thresh, tas)
    over = tas < t
    group = over.resample(time=freq)

    out = group.map(rl.windowed_run_count, window=window, dim="time")
    return to_agg_units(out, tas, "count")


@declare_units(tas="[temperature]", thresh="[temperature]")
def cold_spell_frequency(
    tas: xarray.DataArray,
    thresh: str = "-10 degC",
    window: int = 5,
    freq: str = "AS-JUL",
) -> xarray.DataArray:
    r"""Cold spell frequency.

    The number of cold spell events, defined as a sequence of consecutive days with mean daily
    temperature below a threshold.

    Parameters
    ----------
    tas : xarray.DataArray
      Mean daily temperature.
    thresh : str
      Threshold temperature below which a cold spell begins.
    window : int
      Minimum number of days with temperature below threshold to qualify as a cold spell.
    freq : str
      Resampling frequency.

    Returns
    -------
    xarray.DataArray, [dimensionless]
      Cold spell frequency.


    """
    t = convert_units_to(thresh, tas)
    over = tas < t
    group = over.resample(time=freq)

    out = group.map(rl.windowed_run_events, window=window, dim="time")
    out.attrs["units"] = ""
    return out


@declare_units(snd="[length]", thresh="[length]")
def continuous_snow_cover_end(
    snd: xarray.DataArray, thresh: str = "2 cm", window: int = 14, freq: str = "AS-JUL"
) -> xarray.DataArray:
    r"""End date of continuous snow cover.

    First day after the start of the continuous snow cover when snow depth is below `threshold` for at least
    `window` consecutive days.
    WARNING: The default `freq` is valid for the northern hemisphere.

    Parameters
    ----------
    snd : xarray.DataArray
      Surface snow thickness.
    thresh : str
      Threshold snow thickness.
    window : int
      Minimum number of days with snow depth below threshold.
    freq : str
      Resampling frequency.

    Returns
    -------
    xarray.DataArray, [dimensionless]
      First day after the start of the continuous snow cover when the snow depth goes below a threshold
      for a minimum duration.
      If there is no such day, return np.nan.

    References
    ----------
    Chaumont D., Mailhot A., Diaconescu E.P., Fournier É., Logan T. 2017: Élaboration du portrait bioclimatique futur
    du Nunavik – Tome II. [Rapport présenté au Ministère de la forêt, de la faune et des parcs], Ouranos.
    """
    thresh = convert_units_to(thresh, snd)
    cond = snd >= thresh

    out = (
        cond.resample(time=freq)
        .map(rl.season, window=window, dim="time", coord="dayofyear")
        .end
    )
    out.attrs.update(units="", is_dayofyear=1, calendar=get_calendar(snd))
    return out


@declare_units(snd="[length]", thresh="[length]")
def continuous_snow_cover_start(
    snd: xarray.DataArray, thresh: str = "2 cm", window: int = 14, freq: str = "AS-JUL"
) -> xarray.DataArray:
    r"""Start date of continuous snow cover.

    Day of year when snow depth is above `threshold` for at least `window` consecutive days.
    WARNING: The default `freq` is valid for the northern hemisphere.

    Parameters
    ----------
    snd : xarray.DataArray
      Surface snow thickness.
    thresh : str
      Threshold snow thickness.
    window : int
      Minimum number of days with snow depth above or equal to threshold.
    freq : str
      Resampling frequency.

    Returns
    -------
    xarray.DataArray, [dimensionless]
      First day of the year when the snow depth is superior to a threshold for a minimum duration.
      If there is no such day, return np.nan.

    References
    ----------
    Chaumont D., Mailhot A., Diaconescu E.P., Fournier É., Logan T. 2017: Élaboration du portrait bioclimatique futur
    du Nunavik – Tome II. [Rapport présenté au Ministère de la forêt, de la faune et des parcs], Ouranos.
    """
    thresh = convert_units_to(thresh, snd)
    cond = snd >= thresh

    out = (
        cond.resample(time=freq)
        .map(
            rl.season,
            window=window,
            dim="time",
            coord="dayofyear",
        )
        .start
    )
    out.attrs.update(units="", is_dayofyear=1, calendar=get_calendar(snd))
    return out


@declare_units(pr="[precipitation]", thresh="[precipitation]")
def daily_pr_intensity(
    pr: xarray.DataArray, thresh: str = "1 mm/day", freq: str = "YS"
) -> xarray.DataArray:
    r"""Average daily precipitation intensity.

    Return the average precipitation over wet days.

    Parameters
    ----------
    pr : xarray.DataArray
      Daily precipitation.
    thresh : str
      Precipitation value over which a day is considered wet.
    freq : str
      Resampling frequency.

    Returns
    -------
    xarray.DataArray, [precipitation]
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

    >>> from xclim.indices import daily_pr_intensity
    >>> pr = xr.open_dataset(path_to_pr_file).pr
    >>> daily_int = daily_pr_intensity(pr, thresh='5 mm/day', freq="QS-DEC")
    """
    t = convert_units_to(thresh, pr, "hydro")

    # Get amount of rain (not rate)
    pram = rate2amount(pr)

    # put pram = 0 for non wet-days
    pram_wd = xarray.where(pr >= t, pram, 0)
    pram_wd.attrs["units"] = pram.units

    # sum over wanted period
    s = pram_wd.resample(time=freq).sum(dim="time", keep_attrs=True)

    # get number of wetdays over period
    wd = wetdays(pr, thresh=thresh, freq=freq)
    out = s / wd
    out.attrs["units"] = f"{str2pint(s.units) / str2pint(wd.units):~}"
    return out


@declare_units(pr="[precipitation]", thresh="[precipitation]")
def dry_days(
    pr: xarray.DataArray, thresh: str = "0.2 mm/d", freq: str = "YS"
) -> xarray.DataArray:
    r"""Dry days.

    The number of days with daily precipitation below threshold.

    Parameters
    ----------
    pr : xarray.DataArray
      Daily precipitation.
    thresh : str
      Threshold temperature on which to base evaluation.
    freq : str
      Resampling frequency.

    Returns
    -------
    xarray.DataArray, [time]
      Number of days with daily precipitation below threshold.

    Notes
    -----
    Let :math:`PR_{ij}` be the daily precipitation at day :math:`i` of period :math:`j`. Then
    counted is the number of days where:

    .. math::

        \sum PR_{ij} < Threshold [mm/day]
    """
    thresh = convert_units_to(thresh, pr)
    out = threshold_count(pr, "<", thresh, freq)
    out = to_agg_units(out, pr, "count")
    return out


@declare_units(pr="[precipitation]", thresh="[precipitation]")
def maximum_consecutive_wet_days(
    pr: xarray.DataArray, thresh: str = "1 mm/day", freq: str = "YS"
) -> xarray.DataArray:
    r"""Consecutive wet days.

    Returns the maximum number of consecutive wet days.

    Parameters
    ----------
    pr : xarray.DataArray
      Mean daily precipitation flux.
    thresh : str
      Threshold precipitation on which to base evaluation.
    freq : str
      Resampling frequency.

    Returns
    -------
    xarray.DataArray, [time]
      The maximum number of consecutive wet days.

    Notes
    -----
    Let :math:`\mathbf{x}=x_0, x_1, \ldots, x_n` be a daily precipitation series and
    :math:`\mathbf{s}` be the sorted vector of indices :math:`i` where :math:`[p_i > thresh] \neq [p_{i+1} >
    thresh]`, that is, the days when the precipitation crosses the *wet day* threshold.
    Then the maximum number of consecutive wet days is given by

    .. math::


       \max(\mathbf{d}) \quad \mathrm{where} \quad d_j = (s_j - s_{j-1}) [x_{s_j} > 0^\circ C]

    where :math:`[P]` is 1 if :math:`P` is true, and 0 if false. Note that this formula does not handle sequences at
    the start and end of the series, but the numerical algorithm does.
    """
    thresh = convert_units_to(thresh, pr, "hydro")

    group = (pr > thresh).resample(time=freq)
    out = group.map(rl.longest_run, dim="time")
    out = to_agg_units(out, pr, "count")
    return out


@declare_units(tas="[temperature]", thresh="[temperature]")
def cooling_degree_days(
    tas: xarray.DataArray, thresh: str = "18 degC", freq: str = "YS"
) -> xarray.DataArray:
    r"""Cooling degree days.

    Sum of degree days above the temperature threshold at which spaces are cooled.

    Parameters
    ----------
    tas : xarray.DataArray
      Mean daily temperature.
    thresh : str
      Temperature threshold above which air is cooled.
    freq : str
      Resampling frequency.

    Returns
    -------
    xarray.DataArray, [time][temperature]
      Cooling degree days

    Notes
    -----
    Let :math:`x_i` be the daily mean temperature at day :math:`i`. Then the cooling degree days above
    temperature threshold :math:`thresh` over period :math:`\phi` is given by:

    .. math::

        \sum_{i \in \phi} (x_{i}-{thresh} [x_i > thresh]

    where :math:`[P]` is 1 if :math:`P` is true, and 0 if false.
    """
    thresh = convert_units_to(thresh, tas)

    out = (tas - thresh).clip(min=0).resample(time=freq).sum(dim="time")
    out = to_agg_units(out, tas, "delta_prod")
    return out


@declare_units(tas="[temperature]", thresh="[temperature]")
def freshet_start(
    tas: xarray.DataArray, thresh: str = "0 degC", window: int = 5, freq: str = "YS"
) -> xarray.DataArray:
    r"""First day consistently exceeding threshold temperature.

    Returns first day of period where a temperature threshold is exceeded
    over a given number of days.

    Parameters
    ----------
    tas : xarray.DataArray
      Mean daily temperature.
    thresh : str
      Threshold temperature on which to base evaluation.
    window : int
      Minimum number of days with temperature above threshold needed for evaluation.
    freq : str
      Resampling frequency.

    Returns
    -------
    xarray.DataArray, [dimensionless]
      Day of the year when temperature exceeds threshold over a given number of days for the first time. If there is
      no such day, return np.nan.

    Notes
    -----
    Let :math:`x_i` be the daily mean temperature at day of the year :math:`i` for values of :math:`i` going from 1
    to 365 or 366. The start date of the freshet is given by the smallest index :math:`i` for which

    .. math::

       \prod_{j=i}^{i+w} [x_j > thresh]

    is true, where :math:`w` is the number of days the temperature threshold should be exceeded, and :math:`[P]` is
    1 if :math:`P` is true, and 0 if false.
    """
    thresh = convert_units_to(thresh, tas)
    over = tas > thresh
    out = over.resample(time=freq).map(rl.first_run, window=window, coord="dayofyear")
    out.attrs.update(units="", is_dayofyear=1, calendar=get_calendar(tas))
    return out


@declare_units(tas="[temperature]", thresh="[temperature]")
def growing_degree_days(
    tas: xarray.DataArray, thresh: str = "4.0 degC", freq: str = "YS"
) -> xarray.DataArray:
    r"""Growing degree-days over threshold temperature value.

    The sum of degree-days over the threshold temperature.

    Parameters
    ----------
    tas : xarray.DataArray
      Mean daily temperature.
    thresh : str
      Threshold temperature on which to base evaluation.
    freq : str
      Resampling frequency.

    Returns
    -------
    xarray.DataArray, [time][temperature]
      The sum of growing degree-days above a given threshold.

    Notes
    -----
    Let :math:`TG_{ij}` be the daily mean temperature at day :math:`i` of period :math:`j`. Then the
    growing degree days are:

    .. math::

        GD4_j = \sum_{i=1}^I (TG_{ij}-{4} | TG_{ij} > {4}℃)
    """
    thresh = convert_units_to(thresh, tas)
    out = (tas - thresh).clip(min=0).resample(time=freq).sum(dim="time")
    return to_agg_units(out, tas, "delta_prod")


@declare_units(tas="[temperature]", thresh="[temperature]")
def growing_season_end(
    tas: xarray.DataArray,
    thresh: str = "5.0 degC",
    mid_date: DayOfYearStr = "07-01",
    window: int = 5,
    freq: str = "YS",
) -> xarray.DataArray:
    r"""End of the growing season.

    Day of the year of the start of a sequence of days with a temperature consistently
    below a threshold, after a period with temperatures consistently above the same threshold.

    Parameters
    ----------
    tas : xarray.DataArray
      Mean daily temperature.
    thresh : str
      Threshold temperature on which to base evaluation.
    mid_date : str
      Date of the year after which to look for the end of the season. Should have the format '%m-%d'.
    window : int
      Minimum number of days with temperature below threshold needed for evaluation.
    freq : str
      Resampling frequency.

    Returns
    -------
    xarray.DataArray, [dimensionless]
      Day of the year when temperature is inferior to a threshold over a given number of days for the first time.
      If there is no such day or if a growing season is not detected, returns np.nan.
      If the growing season does not end within the time period, returns the last day of the period.
    """
    thresh = convert_units_to(thresh, tas)
    cond = tas >= thresh

    out = cond.resample(time=freq).map(
        rl.run_end_after_date,
        window=window,
        date=mid_date,
        dim="time",
        coord="dayofyear",
    )
    out.attrs.update(units="", is_dayofyear=1, calendar=get_calendar(tas))
    return out


@declare_units(tas="[temperature]", thresh="[temperature]")
def growing_season_length(
    tas: xarray.DataArray,
    thresh: str = "5.0 degC",
    window: int = 6,
    mid_date: DayOfYearStr = "07-01",
    freq: str = "YS",
) -> xarray.DataArray:
    r"""Growing season length.

    The number of days between the first occurrence of at least six consecutive days
    with mean daily temperature over a threshold (default: 5℃) and the first occurrence
    of at least six consecutive days with mean daily temperature below the same threshold
    after a certain date.
    (Usually July 1st in the northern emisphere and January 1st in the southern hemisphere.)

    WARNING: The default calendar values are only valid for the northern hemisphere.

    Parameters
    ----------
    tas : xarray.DataArray
      Mean daily temperature.
    thresh : str
      Threshold temperature on which to base evaluation.
    window : int
      Minimum number of days with temperature above threshold to mark the beginning and end of growing season.
    mid_date : str
      Date of the year after which to look for the end of the season. Should have the format '%m-%d'.
    freq : str
      Resampling frequency.

    Returns
    -------
    xarray.DataArray, [time]
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

    Examples
    --------
    >>> from xclim.indices import growing_season_length
    >>> tas = xr.open_dataset(path_to_tas_file).tas

    # For the Northern Hemisphere:
    >>> gsl_nh = growing_season_length(tas, mid_date='07-01', freq='AS')

    # If working in the Southern Hemisphere, one can use:
    >>> gsl_sh = growing_season_length(tas, mid_date='01-01', freq='AS-JUL')
    """
    thresh = convert_units_to(thresh, tas)
    cond = tas >= thresh

    out = cond.resample(time=freq).map(
        rl.season_length,
        window=window,
        date=mid_date,
        dim="time",
    )
    return to_agg_units(out, tas, "count")


@declare_units(tasmin="[temperature]", thresh="[temperature]")
def frost_season_length(
    tasmin: xarray.DataArray,
    window: int = 5,
    mid_date: Optional[DayOfYearStr] = "01-01",
    thresh: str = "0.0 degC",
    freq: str = "AS-JUL",
) -> xarray.DataArray:
    r"""Frost season length.

    The number of days between the first occurrence of at least N (def: 5) consecutive days
    with minimum daily temperature under a threshold (default: 0℃) and the first occurrence
    of at least N (def 5) consecutive days with minimum daily temperature above the same threshold
    A mid date can be given to limit the earliest day the end of season can take.
    WARNING: The default freq and mid_date values are valid for the northern hemisphere.

    Parameters
    ----------
    tasmin : xarray.DataArray
      Minimum daily temperature.
    window : int
      Minimum number of days with temperature below threshold to mark the beginning and end of frost season.
    mid_date : str, optional
      Date the must be included in the season. It is the earliest the end of the season can be.
      If None, there is no limit.
    thresh : str
      Threshold temperature on which to base evaluation.
    freq : str
      Resampling frequency.

    Returns
    -------
    xarray.DataArray, [time]
      Frost season length.

    Notes
    -----
    Let :math:`TN_{ij}` be the minimum temperature at day :math:`i` of period :math:`j`. Then counted is
    the number of days between the first occurrence of at least N consecutive days with:

    .. math::

        TN_{ij} > 0 ℃

    and the first subsequent occurrence of at least N consecutive days with:

    .. math::

        TN_{ij} < 0 ℃

    Examples
    --------
    >>> from xclim.indices import frost_season_length
    >>> tasmin = xr.open_dataset(path_to_tasmin_file).tasmin

    # For the Northern Hemisphere:
    >>> fsl_nh = frost_season_length(tasmin, freq='AS-JUL')

    # If working in the Southern Hemisphere, one can use:
    >>> dsl_sh = frost_season_length(tasmin, freq='YS')
    """
    thresh = convert_units_to(thresh, tasmin)
    cond = tasmin < thresh

    out = cond.resample(time=freq).map(
        rl.season_length,
        window=window,
        date=mid_date,
        dim="time",
    )
    return to_agg_units(out, tasmin, "count")


@declare_units(tas="[temperature]", thresh="[temperature]")
def last_spring_frost(
    tas: xarray.DataArray,
    thresh: str = "0 degC",
    before_date: DayOfYearStr = "07-01",
    window: int = 1,
    freq: str = "YS",
) -> xarray.DataArray:
    r"""Last day of temperatures inferior to a threshold temperature.

    Returns last day of period where a temperature is inferior to a threshold
    over a given number of days and limited to a final calendar date.

    Parameters
    ----------
    tas : xarray.DataArray
      Mean daily temperature.
    thresh : str
      Threshold temperature on which to base evaluation.
    before_date : str,
      Date of the year before which to look for the final frost event. Should have the format '%m-%d'.
    window : int
      Minimum number of days with temperature below threshold needed for evaluation.
    freq : str
      Resampling frequency.

    Returns
    -------
    xarray.DataArray, [dimensionless]
      Day of the year when temperature is inferior to a threshold over a given number of days for the first time.
      If there is no such day, returns np.nan.
    """
    thresh = convert_units_to(thresh, tas)
    cond = tas < thresh

    out = cond.resample(time=freq).map(
        rl.last_run_before_date,
        window=window,
        date=before_date,
        dim="time",
        coord="dayofyear",
    )
    out.attrs.update(units="", is_dayofyear=1, calendar=get_calendar(tas))
    return out


@declare_units(tasmin="[temperature]", thresh="[temperature]")
def first_day_below(
    tasmin: xarray.DataArray,
    thresh: str = "0 degC",
    after_date: DayOfYearStr = "07-01",
    window: int = 1,
    freq: str = "YS",
) -> xarray.DataArray:
    r"""First day of temperatures inferior to a threshold temperature.

    Returns first day of period where a temperature is inferior to a threshold
    over a given number of days, limited to a starting calendar date.

    WARNING: The default date and freq are valid for the northern hemisphere.

    Parameters
    ----------
    tasmin : xarray.DataArray
      Minimum daily temperature.
    thresh : str
      Threshold temperature on which to base evaluation.
    after_date : str
      Date of the year after which to look for the first frost event. Should have the format '%m-%d'.
    window : int
      Minimum number of days with temperature below threshold needed for evaluation.
    freq : str
      Resampling frequency.

    Returns
    -------
    xarray.DataArray, [dimensionless]
      Day of the year when minimum temperature is inferior to a threshold over a given number of days for the first time.
      If there is no such day, returns np.nan.
    """
    thresh = convert_units_to(thresh, tasmin)
    cond = tasmin < thresh

    out = cond.resample(time=freq).map(
        rl.first_run_after_date,
        window=window,
        date=after_date,
        dim="time",
        coord="dayofyear",
    )
    out.attrs.update(units="", is_dayofyear=1, calendar=get_calendar(tasmin))
    return out


@declare_units(tasmin="[temperature]", thresh="[temperature]")
def first_day_above(
    tasmin: xarray.DataArray,
    thresh: str = "0 degC",
    after_date: DayOfYearStr = "01-01",
    window: int = 1,
    freq: str = "YS",
) -> xarray.DataArray:
    r"""First day of temperatures superior to a threshold temperature.

    Returns first day of period where a temperature is superior to a threshold
    over a given number of days, limited to a starting calendar date.

    WARNING: The default date and freq are valid for the northern hemisphere.

    Parameters
    ----------
    tasmin : xarray.DataArray
      Minimum daily temperature.
    thresh : str
      Threshold temperature on which to base evaluation.
    after_date : str
      Date of the year after which to look for the first event. Should have the format '%m-%d'.
    window : int
      Minimum number of days with temperature above threshold needed for evaluation.
    freq : str
      Resampling frequency.

    Returns
    -------
    xarray.DataArray, [dimensionless]
      Day of the year when minimum temperature is superior to a threshold over a given number of days for the first time.
      If there is no such day, returns np.nan.
    """
    thresh = convert_units_to(thresh, tasmin)
    cond = tasmin > thresh

    out = cond.resample(time=freq).map(
        rl.first_run_after_date,
        window=window,
        date=after_date,
        dim="time",
        coord="dayofyear",
    )
    out.attrs.update(units="", is_dayofyear=1, calendar=get_calendar(tasmin))
    return out


@declare_units(prsn="[precipitation]", thresh="[precipitation]")
def first_snowfall(
    prsn: xarray.DataArray,
    thresh: str = "0.5 mm/day",
    freq: str = "AS-JUL",
) -> xarray.DataArray:
    r"""First day with solid precipitation above a threshold.

    Returns the first day of a period where the solid precipitation exceeds a threshold.
    WARNING: The default `freq` is valid for the northern hemisphere.

    Parameters
    ----------
    prsn : xarray.DataArray
      Solid precipitation flux.
    thresh : str
      Threshold precipitation flux on which to base evaluation.
    freq : str
      Resampling frequency.

    Returns
    -------
    xarray.DataArray, [dimensionless]
      First day of the year when the solid precipitation is superior to a threshold.
      If there is no such day, returns np.nan.

    References
    ----------
    Climate Projections for the National Capital Region (2020), Volume 1: Results and Interpretation for Key Climate
    Indices, Report 193600.00, Prepared for Ottawa by CBCL.
    """
    thresh = convert_units_to(thresh, prsn)
    cond = prsn >= thresh

    out = cond.resample(time=freq).map(
        rl.first_run,
        window=1,
        dim="time",
        coord="dayofyear",
    )
    out.attrs.update(units="", is_dayofyear=1, calendar=get_calendar(prsn))
    return out


@declare_units(prsn="[precipitation]", thresh="[precipitation]")
def last_snowfall(
    prsn: xarray.DataArray,
    thresh: str = "0.5 mm/day",
    freq: str = "AS-JUL",
) -> xarray.DataArray:
    r"""Last day with solid precipitation above a threshold.

    Returns the last day of a period where the solid precipitation exceeds a threshold.
    WARNING: The default freq is valid for the northern hemisphere.

    Parameters
    ----------
    prsn : xarray.DataArray
      Solid precipitation flux.
    thresh : str
      Threshold precipitation flux on which to base evaluation.
    freq : str
      Resampling frequency.

    Returns
    -------
    xarray.DataArray, [dimensionless]
      Last day of the year when the solid precipitation is superior to a threshold.
      If there is no such day, returns np.nan.

    References
    ----------
    Climate Projections for the National Capital Region (2020), Volume 1: Results and Interpretation for Key Climate
    Indices, Report 193600.00, Prepared for Ottawa by CBCL.
    """
    thresh = convert_units_to(thresh, prsn)
    cond = prsn >= thresh

    out = cond.resample(time=freq).map(
        rl.last_run,
        window=1,
        dim="time",
        coord="dayofyear",
    )
    out.attrs["units"] = ""
    return out


@declare_units(prsn="[precipitation]", low="[precipitation]", high="[precipitation]")
def days_with_snow(
    prsn: xarray.DataArray,  # noqa
    low: str = "0 kg m-2 s-1",
    high: str = "1E6 kg m-2 s-1",
    freq: str = "AS-JUL",
) -> xarray.DataArray:
    r"""Days with snow.

    Return the number of days where snowfall is within low and high thresholds.

    Parameters
    ----------
    prsn : xr.DataArray
      Solid precipitation flux.
    low : float
      Minimum threshold solid precipitation flux.
    high : float
      Maximum threshold solid precipitation flux.
    freq : str
      Resampling frequency defining the periods as defined in
      https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#resampling.

    Returns
    -------
    xarray.DataArray, [time]
      Number of days where snowfall is between low and high thresholds.

    References
    ----------
    Matthews, L., Andrey, J., & Picketts, I. (2017). Planning for Winter Road Maintenance in the Context of Climate
    Change, Weather, Climate, and Society, 9(3), 521-532, https://doi.org/10.1175/WCAS-D-16-0103.1
    """
    low = convert_units_to(low, prsn)
    high = convert_units_to(high, prsn)
    out = domain_count(prsn, low, high, freq)
    return to_agg_units(out, prsn, "count")


@declare_units(tasmax="[temperature]", thresh="[temperature]")
def heat_wave_index(
    tasmax: xarray.DataArray,
    thresh: str = "25.0 degC",
    window: int = 5,
    freq: str = "YS",
) -> xarray.DataArray:
    """Heat wave index.

    Number of days that are part of a heatwave, defined as five or more consecutive days over 25℃.

    Parameters
    ----------
    tasmax : xarray.DataArray
      Maximum daily temperature.
    thresh : str
      Threshold temperature on which to designate a heatwave.
    window : int
      Minimum number of days with temperature above threshold to qualify as a heatwave.
    freq : str
      Resampling frequency.

    Returns
    -------
    DataArray, [time]
      Heat wave index.
    """
    thresh = convert_units_to(thresh, tasmax)
    over = tasmax > thresh
    group = over.resample(time=freq)

    out = group.map(rl.windowed_run_count, window=window, dim="time")
    return to_agg_units(out, tasmax, "count")


@declare_units(tas="[temperature]", thresh="[temperature]")
def heating_degree_days(
    tas: xarray.DataArray, thresh: str = "17.0 degC", freq: str = "YS"
) -> xarray.DataArray:
    r"""Heating degree days.

    Sum of degree days below the temperature threshold at which spaces are heated.

    Parameters
    ----------
    tas : xarray.DataArray
      Mean daily temperature.
    thresh : str
      Threshold temperature on which to base evaluation.
    freq : str
      Resampling frequency.

    Returns
    -------
    xarray.DataArray, [time][temperature]
      Heating degree days index.

    Notes
    -----
    Let :math:`TG_{ij}` be the daily mean temperature at day :math:`i` of period :math:`j`. Then the
    heating degree days are:

    .. math::

        HD17_j = \sum_{i=1}^{I} (17℃ - TG_{ij})
    """
    thresh = convert_units_to(thresh, tas)

    out = (thresh - tas).clip(0).resample(time=freq).sum(dim="time")
    return to_agg_units(out, tas, "delta_prod")


@declare_units(tasmax="[temperature]", thresh_tasmax="[temperature]")
def hot_spell_max_length(
    tasmax: xarray.DataArray,
    thresh_tasmax: str = "30 degC",
    window: int = 1,
    freq: str = "YS",
) -> xarray.DataArray:
    """Longest hot spell.

    Longest spell of high temperatures over a given period.

    The longest series of consecutive days with tasmax ≥ 30 °C. Here, there is no minimum threshold for number of
    days in a row that must be reached or exceeded to count as a spell. A year with zero +30 °C days will return a
    longest spell value of zero.

    Parameters
    ----------
    tasmax : xarray.DataArray
      Maximum daily temperature.
    thresh_tasmax : str
      The maximum temperature threshold needed to trigger a heatwave event.
    window : int
      Minimum number of days with temperatures above thresholds to qualify as a heatwave.
    freq : str
      Resampling frequency.

    Returns
    -------
    xarray.DataArray, [time]
      Maximum length of continuous hot days at the wanted frequency.

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
    thresh_tasmax = convert_units_to(thresh_tasmax, tasmax)

    cond = tasmax > thresh_tasmax
    group = cond.resample(time=freq)
    max_l = group.map(rl.longest_run, dim="time")
    out = max_l.where(max_l >= window, 0)
    return to_agg_units(out, tasmax, "count")


@declare_units(tasmax="[temperature]", thresh_tasmax="[temperature]")
def hot_spell_frequency(
    tasmax: xarray.DataArray,
    thresh_tasmax: str = "30 degC",
    window: int = 3,
    freq: str = "YS",
) -> xarray.DataArray:
    """Hot spell frequency.

    Number of hot spells over a given period. A hot spell is defined as an event
    where the maximum daily temperature exceeds a specific threshold
    over a minimum number of days.

    Parameters
    ----------
    tasmax : xarray.DataArray
      Maximum daily temperature.
    thresh_tasmax : str
      The maximum temperature threshold needed to trigger a heatwave event.
    window : int
      Minimum number of days with temperatures above thresholds to qualify as a heatwave.
    freq : str
      Resampling frequency.

    Returns
    -------
    xarray.DataArray, [dimensionless]
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
    thresh_tasmax = convert_units_to(thresh_tasmax, tasmax)

    cond = tasmax > thresh_tasmax
    group = cond.resample(time=freq)
    out = group.map(rl.windowed_run_events, window=window, dim="time")
    out.attrs["units"] = ""
    return out


@declare_units(snd="[length]", thresh="[length]")
def snow_cover_duration(
    snd: xarray.DataArray, thresh: str = "2 cm", freq: str = "AS-JUL"
) -> xarray.DataArray:
    """Number of days with snow depth above a threshold.

    Number of days where surface snow depth is greater or equal to given threshold.
    WARNING: The default `freq` is valid for the northern hemisphere.

    Parameters
    ----------
    snd : xarray.DataArray
      Surface snow thickness.
    thresh : str
      Threshold snow thickness.
    freq : str
      Resampling frequency.

    Returns
    -------
    xarray.DataArray, [time]
      Number of days where snow depth is greater or equal to threshold.
    """
    thresh = convert_units_to(thresh, snd)
    out = threshold_count(snd, ">=", thresh, freq)
    return to_agg_units(out, snd, "count")


@declare_units(tasmin="[temperature]", thresh="[temperature]")
def tn_days_above(
    tasmin: xarray.DataArray, thresh: str = "20.0 degC", freq: str = "YS"
):  # noqa: D401
    """Number of days with tasmin above a threshold (number of tropical nights).

    Number of days where daily minimum temperature exceeds a threshold.

    Parameters
    ----------
    tasmin : xarray.DataArray
      Minimum daily temperature.
    thresh : str
      Threshold temperature on which to base evaluation.
    freq : str
      Resampling frequency.

    Returns
    -------
    xarray.DataArray, [time]
      Number of days where tasmin > threshold.

    Notes
    -----
    Let :math:`TN_{ij}` be the daily minimum temperature at day :math:`i` of period :math:`j`. Then
    counted is the number of days where:

    .. math::

        TN_{ij} > Threshold [℃]
    """
    thresh = convert_units_to(thresh, tasmin)
    f = threshold_count(tasmin, ">", thresh, freq)
    return to_agg_units(f, tasmin, "count")


@declare_units(tasmin="[temperature]", thresh="[temperature]")
def tn_days_below(
    tasmin: xarray.DataArray, thresh: str = "-10.0 degC", freq: str = "YS"
) -> xarray.DataArray:  # noqa: D401
    """Number of days with tasmin below a threshold.

    Number of days where daily minimum temperature is below a threshold.

    Parameters
    ----------
    tasmin : xarray.DataArray
      Minimum daily temperature.
    thresh : str
      Threshold temperature on which to base evaluation.
    freq : str
      Resampling frequency.

    Returns
    -------
    xarray.DataArray, [time]
      Number of days where tasmin < threshold.

    Notes
    -----
    Let :math:`TN_{ij}` be the daily minimum temperature at day :math:`i` of period :math:`j`. Then
    counted is the number of days where:

    .. math::

        TN_{ij} < Threshold [℃]
    """
    thresh = convert_units_to(thresh, tasmin)
    f1 = threshold_count(tasmin, "<", thresh, freq)
    return to_agg_units(f1, tasmin, "count")


@declare_units(tas="[temperature]", thresh="[temperature]")
def tg_days_above(
    tas: xarray.DataArray, thresh: str = "10.0 degC", freq: str = "YS"
):  # noqa: D401
    """Number of days with tas above a threshold.

    Number of days where daily mean temperature exceeds a threshold.

    Parameters
    ----------
    tas : xarray.DataArray
      Mean daily temperature.
    thresh : str
      Threshold temperature on which to base evaluation.
    freq : str
      Resampling frequency.

    Returns
    -------
    xarray.DataArray, [time]
      Number of days where tas > threshold.

    Notes
    -----
    Let :math:`TG_{ij}` be the daily mean temperature at day :math:`i` of period :math:`j`. Then
    counted is the number of days where:

    .. math::

        TG_{ij} > Threshold [℃]
    """
    thresh = convert_units_to(thresh, tas)
    f = threshold_count(tas, ">", thresh, freq)
    return to_agg_units(f, tas, "count")


@declare_units(tas="[temperature]", thresh="[temperature]")
def tg_days_below(
    tas: xarray.DataArray, thresh: str = "10.0 degC", freq: str = "YS"
):  # noqa: D401
    """Number of days with tas below a threshold.

    Number of days where daily mean temperature is below a threshold.

    Parameters
    ----------
    tas : xarray.DataArray
      Mean daily temperature.
    thresh : str
      Threshold temperature on which to base evaluation.
    freq : str
      Resampling frequency.

    Returns
    -------
    xarray.DataArray, [time]
      Number of days where tas < threshold.

    Notes
    -----
    Let :math:`TG_{ij}` be the daily mean temperature at day :math:`i` of period :math:`j`. Then
    counted is the number of days where:

    .. math::

        TG_{ij} < Threshold [℃]
    """
    thresh = convert_units_to(thresh, tas)
    f1 = threshold_count(tas, "<", thresh, freq)
    return to_agg_units(f1, tas, "count")


@declare_units(tasmax="[temperature]", thresh="[temperature]")
def tx_days_above(
    tasmax: xarray.DataArray, thresh: str = "25.0 degC", freq: str = "YS"
) -> xarray.DataArray:  # noqa: D401
    """Number of days with tasmax above a threshold (number of summer days).

    Number of days where daily maximum temperature exceeds a threshold.

    Parameters
    ----------
    tasmax : xarray.DataArray
      Maximum daily temperature.
    thresh : str
      Threshold temperature on which to base evaluation.
    freq : str
      Resampling frequency.

    Returns
    -------
    xarray.DataArray, [time]
      Number of days where tasmax > threshold (number of summer days).

    Notes
    -----
    Let :math:`TX_{ij}` be the daily maximum temperature at day :math:`i` of period :math:`j`. Then
    counted is the number of days where:

    .. math::

        TX_{ij} > Threshold [℃]
    """
    thresh = convert_units_to(thresh, tasmax)
    f = threshold_count(tasmax, ">", thresh, freq)
    return to_agg_units(f, tasmax, "count")


@declare_units(tasmax="[temperature]", thresh="[temperature]")
def tx_days_below(
    tasmax: xarray.DataArray, thresh: str = "25.0 degC", freq: str = "YS"
):  # noqa: D401
    """Number of days with tmax below a threshold.

    Number of days where daily maximum temperature is below a threshold.

    Parameters
    ----------
    tasmax : xarray.DataArray
      Maximum daily temperature.
    thresh : str
      Threshold temperature on which to base evaluation.
    freq : str
      Resampling frequency.

    Returns
    -------
    xarray.DataArray, [time]
      Number of days where tasmin < threshold.

    Notes
    -----
    Let :math:`TN_{ij}` be the daily minimum temperature at day :math:`i` of period :math:`j`. Then
    counted is the number of days where:

    .. math::

        TN_{ij} < Threshold [℃]
    """
    thresh = convert_units_to(thresh, tasmax)
    f1 = threshold_count(tasmax, "<", thresh, freq)
    return to_agg_units(f1, tasmax, "count")


@declare_units(tasmax="[temperature]", thresh="[temperature]")
def warm_day_frequency(
    tasmax: xarray.DataArray, thresh: str = "30 degC", freq: str = "YS"
) -> xarray.DataArray:
    """Frequency of extreme warm days.

    Return the number of days with tasmax > thresh per period

    Parameters
    ----------
    tasmax : xarray.DataArray
      Mean daily temperature.
    thresh : str
      Threshold temperature on which to base evaluation.
    freq : str
      Resampling frequency.

    Returns
    -------
    xarray.DataArray, [time]
      Number of days with tasmax > threshold per period.

    Notes
    -----
    Let :math:`TX_{ij}` be the daily maximum temperature at day :math:`i` of period :math:`j`. Then
    counted is the number of days where:

    .. math::

        TN_{ij} > Threshold [℃]

    """
    thresh = convert_units_to(thresh, tasmax)
    events = threshold_count(tasmax, ">", thresh, freq)
    return to_agg_units(events, tasmax, "count")


@declare_units(tasmin="[temperature]", thresh="[temperature]")
def warm_night_frequency(
    tasmin: xarray.DataArray, thresh: str = "22 degC", freq: str = "YS"
) -> xarray.DataArray:
    """Frequency of extreme warm nights.

    Return the number of days with tasmin > thresh per period

    Parameters
    ----------
    tasmin : xarray.DataArray
      Minimum daily temperature.
    thresh : str
      Threshold temperature on which to base evaluation.
    freq : str
      Resampling frequency.

    Returns
    -------
    xarray.DataArray, [time]
      Number of days with tasmin > threshold per period.
    """
    thresh = convert_units_to(thresh, tasmin)
    events = threshold_count(tasmin, ">", thresh, freq)
    return to_agg_units(events, tasmin, "count")


@declare_units(pr="[precipitation]", thresh="[precipitation]")
def wetdays(
    pr: xarray.DataArray, thresh: str = "1.0 mm/day", freq: str = "YS"
) -> xarray.DataArray:
    """Wet days.

    Return the total number of days during period with precipitation over threshold.

    Parameters
    ----------
    pr : xarray.DataArray
      Daily precipitation.
    thresh : str
      Precipitation value over which a day is considered wet.
    freq : str
      Resampling frequency.

    Returns
    -------
    xarray.DataArray, [time]
      The number of wet days for each period [day].

    Examples
    --------
    The following would compute for each grid cell of file `pr.day.nc` the number days
    with precipitation over 5 mm at the seasonal frequency, ie DJF, MAM, JJA, SON, DJF, etc.:

    >>> from xclim.indices import wetdays
    >>> pr = xr.open_dataset(path_to_pr_file).pr
    >>> wd = wetdays(pr, thresh="5 mm/day", freq="QS-DEC")
    """
    thresh = convert_units_to(thresh, pr, "hydro")

    wd = threshold_count(pr, ">=", thresh, freq)
    return to_agg_units(wd, pr, "count")


@declare_units(tasmin="[temperature]", thresh="[temperature]")
def maximum_consecutive_frost_days(
    tasmin: xarray.DataArray,
    thresh: str = "0.0 degC",
    freq: str = "AS-JUL",
) -> xarray.DataArray:
    r"""Maximum number of consecutive frost days (Tn < 0℃).

    The maximum number of consecutive days within the period where the
    temperature is under a certain threshold (default: 0°C).
    WARNING: The default freq value is valid for the northern hemisphere.

    Parameters
    ----------
    tasmin : xarray.DataArray
      Minimum daily temperature.
    thresh : str
      Threshold temperature.
    freq : str
      Resampling frequency.

    Returns
    -------
    xarray.DataArray, [time]
      The maximum number of consecutive frost days (tasmin < threshold per period).

    Notes
    -----
    Let :math:`\mathbf{t}=t_0, t_1, \ldots, t_n` be a daily minimum temperature series and :math:`thresh` the threshold
    below which a day is considered a frost day. Let :math:`\mathbf{s}` be the sorted vector of indices :math:`i`
    where :math:`[t_i < thresh] \neq [t_{i+1} < thresh]`, that is, the days when the temperature crosses the threshold.
    Then the maximum number of consecutive frost free days is given by

    .. math::

       \max(\mathbf{d}) \quad \mathrm{where} \quad d_j = (s_j - s_{j-1}) [t_{s_j} > thresh]

    where :math:`[P]` is 1 if :math:`P` is true, and 0 if false. Note that this formula does not handle sequences at
    the start and end of the series, but the numerical algorithm does.
    """
    t = convert_units_to(thresh, tasmin)
    group = (tasmin < t).resample(time=freq)
    out = group.map(rl.longest_run, dim="time")
    return to_agg_units(out, tasmin, "count")


@declare_units(pr="[precipitation]", thresh="[precipitation]")
def maximum_consecutive_dry_days(
    pr: xarray.DataArray, thresh: str = "1 mm/day", freq: str = "YS"
) -> xarray.DataArray:
    r"""Maximum number of consecutive dry days.

    Return the maximum number of consecutive days within the period where precipitation
    is below a certain threshold.

    Parameters
    ----------
    pr : xarray.DataArray
      Mean daily precipitation flux.
    thresh : str
      Threshold precipitation on which to base evaluation.
    freq : str
      Resampling frequency.

    Returns
    -------
    xarray.DataArray, [time]
      The maximum number of consecutive dry days (precipitation < threshold per period).

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
    t = convert_units_to(thresh, pr, "hydro")
    group = (pr < t).resample(time=freq)
    out = group.map(rl.longest_run, dim="time")
    return to_agg_units(out, pr, "count")


@declare_units(tasmin="[temperature]", thresh="[temperature]")
def maximum_consecutive_frost_free_days(
    tasmin: xarray.DataArray, thresh: str = "0 degC", freq: str = "YS"
) -> xarray.DataArray:
    r"""Maximum number of consecutive frost free days (Tn > 0℃).

    Return the maximum number of consecutive days within the period where the
    minimum temperature is above a certain threshold.

    Parameters
    ----------
    tasmin : xarray.DataArray
      Max daily temperature.
    thresh : str
      Threshold temperature.
    freq : str
      Resampling frequency.

    Returns
    -------
    xarray.DataArray, [time]
      The maximum number of consecutive frost free days (tasmin > threshold per period).

    Notes
    -----
    Let :math:`\mathbf{t}=t_0, t_1, \ldots, t_n` be a daily minimum temperature series and :math:`thresh` the threshold
    above which a day is considered a frost free day. Let :math:`\mathbf{s}` be the sorted vector of indices :math:`i`
    where :math:`[t_i < thresh] \neq [t_{i+1} < thresh]`, that is, the days when the temperature crosses the threshold.
    Then the maximum number of consecutive frost free days is given by

    .. math::

       \max(\mathbf{d}) \quad \mathrm{where} \quad d_j = (s_j - s_{j-1}) [t_{s_j} > thresh]

    where :math:`[P]` is 1 if :math:`P` is true, and 0 if false. Note that this formula does not handle sequences at
    the start and end of the series, but the numerical algorithm does.
    """
    t = convert_units_to(thresh, tasmin)
    group = (tasmin > t).resample(time=freq)
    out = group.map(rl.longest_run, dim="time")
    return to_agg_units(out, tasmin, "count")


@declare_units(tasmax="[temperature]", thresh="[temperature]")
def maximum_consecutive_tx_days(
    tasmax: xarray.DataArray, thresh: str = "25 degC", freq: str = "YS"
) -> xarray.DataArray:
    r"""Maximum number of consecutive days with tasmax above a threshold (summer days).

    Return the maximum number of consecutive days within the period where
    the maximum temperature is above a certain threshold.

    Parameters
    ----------
    tasmax : xarray.DataArray
      Max daily temperature.
    thresh : str
      Threshold temperature.
    freq : str
      Resampling frequency.

    Returns
    -------
    xarray.DataArray, [time]
      The maximum number of days with tasmax > thresh per periods (summer days).

    Notes
    -----
    Let :math:`\mathbf{t}=t_0, t_1, \ldots, t_n` be a daily maximum temperature series and :math:`thresh` the threshold
    above which a day is considered a summer day. Let :math:`\mathbf{s}` be the sorted vector of indices :math:`i`
    where :math:`[t_i < thresh] \neq [t_{i+1} < thresh]`, that is, the days when the temperature crosses the threshold.
    Then the maximum number of consecutive dry days is given by

    .. math::

       \max(\mathbf{d}) \quad \mathrm{where} \quad d_j = (s_j - s_{j-1}) [t_{s_j} > thresh]

    where :math:`[P]` is 1 if :math:`P` is true, and 0 if false. Note that this formula does not handle sequences at
    the start and end of the series, but the numerical algorithm does.
    """
    t = convert_units_to(thresh, tasmax)
    group = (tasmax > t).resample(time=freq)
    out = group.map(rl.longest_run, dim="time")
    return to_agg_units(out, tasmax, "count")


@declare_units(siconc="[]", areacello="[area]", thresh="[]")
def sea_ice_area(
    siconc: xarray.DataArray, areacello: xarray.DataArray, thresh: str = "15 pct"
) -> xarray.DataArray:
    """Total sea ice area.

    Sea ice area measures the total sea ice covered area where sea ice concentration is above a threshold,
    usually set to 15%.

    Parameters
    ----------
    siconc : xarray.DataArray
      Sea ice concentration (area fraction).
    areacello : xarray.DataArray
      Grid cell area (usually over the ocean).
    thresh : str
      Minimum sea ice concentration for a grid cell to contribute to the sea ice extent.

    Returns
    -------
    xarray.DataArray, [length]^2
      Sea ice area.

    Notes
    -----
    To compute sea ice area over a subregion, first mask or subset the input sea ice concentration data.

    References
    ----------
    `What is the difference between sea ice area and extent
    <https://nsidc.org/arcticseaicenews/faq/#area_extent>`_

    """
    t = convert_units_to(thresh, siconc)
    factor = convert_units_to("100 pct", siconc)
    out = xarray.dot(siconc.where(siconc >= t, 0), areacello) / factor
    out.attrs["units"] = areacello.units
    return out


@declare_units(siconc="[]", areacello="[area]", thresh="[]")
def sea_ice_extent(
    siconc: xarray.DataArray, areacello: xarray.DataArray, thresh: str = "15 pct"
) -> xarray.DataArray:
    """Total sea ice extent.

    Sea ice extent measures the *ice-covered* area, where a region is considered ice-covered if its sea ice
    concentration is above a threshold usually set to 15%.

    Parameters
    ----------
    siconc : xarray.DataArray
      Sea ice concentration (area fraction).
    areacello : xarray.DataArray
      Grid cell area.
    thresh : str
      Minimum sea ice concentration for a grid cell to contribute to the sea ice extent.

    Returns
    -------
    xarray.DataArray, [length]^2
      Sea ice extent.

    Notes
    -----
    To compute sea ice area over a subregion, first mask or subset the input sea ice concentration data.

    References
    ----------
    `What is the difference between sea ice area and extent
    <https://nsidc.org/arcticseaicenews/faq/#area_extent>`_
    """
    t = convert_units_to(thresh, siconc)
    out = xarray.dot(siconc >= t, areacello)
    out.attrs["units"] = areacello.units
    return out


@declare_units(sfcWind="[speed]", thresh="[speed]")
def windy_days(
    sfcWind: xarray.DataArray, thresh: str = "10.8 m s-1", freq: str = "MS"
) -> xarray.DataArray:
    r"""Windy days.

    The number of days with average near-surface wind speed above threshold.

    Parameters
    ----------
    sfcWind : xarray.DataArray
      Daily average near-surface wind speed.
    thresh : str
      Threshold average near-surface wind speed on which to base evaluation.
    freq : str
      Resampling frequency.

    Returns
    -------
    xarray.DataArray, [time]
      Number of days with average near-surface wind speed above threshold.

    Notes
    -----
    Let :math:`WS_{ij}` be the windspeed at day :math:`i` of period :math:`j`. Then
    counted is the number of days where:

    .. math::

        WS_{ij} >= Threshold [m s-1]
    """
    thresh = convert_units_to(thresh, sfcWind)
    out = threshold_count(sfcWind, ">=", thresh, freq)
    out = to_agg_units(out, sfcWind, "count")
    return out


@declare_units(tasmin="[temperature]", thresh="[temperature]")
def tropical_nights(
    tasmin: xarray.DataArray,
    thresh: str = "20.0 degC",
    freq: str = "YS",
) -> xarray.DataArray:
    """Tropical nights.

    The number of days with minimum daily temperature above threshold.

    Parameters
    ----------
    tasmin : xarray.DataArray
      Minimum daily temperature.
    thresh : str
      Threshold temperature on which to base evaluation.
    freq : str
      Resampling frequency.

    Returns
    -------
    xarray.DataArray, [time]
      Number of days with minimum daily temperature above threshold.

    Notes
    -----
    Let :math:`TN_{ij}` be the daily minimum temperature at day :math:`i` of period :math:`j`. Then
    counted is the number of days where:

    .. math::

        TN_{ij} > Threshold [℃]

    Warnings
    --------
    The `tropical_nights` indice is being deprecated in favour of `tn_days_above` with `thresh="20 degC"` by default.
    The indicator reflects this change. This indice will be removed in a future version of xclim.
    """
    warnings.warn(
        "The `tropical_nights` indice is being deprecated in favour of `tn_days_above` with `thresh='20 degC'`. "
        "This indice will be removed in `xclim>=0.28.0`. Please update your scripts accordingly.",
        UserWarning,
        stacklevel=3,
    )

    return tn_days_above(tasmin, thresh=thresh, freq=freq)


@declare_units(tas="[temperature]", thresh="[temperature]", sum_thresh="K days")
def degree_days_exceedance_date(
    tas: xarray.DataArray,
    thresh: str = "0 degC",
    sum_thresh: str = "25 K days",
    op: str = ">",
    after_date: DayOfYearStr = None,
    freq: str = "YS",
) -> xarray.DataArray:
    r"""Degree days exceedance date.

    Day of year when the sum of degree days exceeds a threshold. Degree days are
    computed above or below a given temperature threshold.

    Parameters
    ----------
    tas : xarray.DataArray
      Mean daily temperature.
    thresh : str
      Threshold temperature on which to base degree days evaluation.
    sum_thresh : str
      Threshold of the degree days sum.
    op : {">", "gt", "<", "lt", ">=", "ge", "<=", "le"}
      If equivalent to '>', degree days are computed as `tas - thresh` and if
      equivalent to '<', they are computed as `thresh - tas`.
    after_date: str, optional
      Date at which to start the cumulative sum. In "mm-dd" format, defaults to the
      start of the sampling period.
    freq : str
      Resampling frequency. If `after_date` is given, `freq` should be annual.

    Returns
    -------
    xarray.DataArray, [dimensionless]
      Degree-days exceedance date.

    Notes
    -----
    Let :math:`TG_{ij}` be the daily mean temperature at day :math:`i` of period :math:`j`,
    :math:`T` is the reference threshold and :math:`ST` is the sum threshold. Then, starting
    at day :math:i_0:, the degree days exceedance date is the first day :math:`k` such that

    .. math::

        \begin{cases}
        ST < \sum_{i=i_0}^{k} \max(TG_{ij} - T, 0) & \text{if $op$ is '>'} \\
        ST < \sum_{i=i_0}^{k} \max(T - TG_{ij}, 0) & \text{if $op$ is '<'}
        \end{cases}

    The resulting :math:`k` is expressed as a day of year.

    Cumulated degree days have numerous applications including plant and insect phenology.
    See https://en.wikipedia.org/wiki/Growing_degree-day for examples.
    """
    thresh = convert_units_to(thresh, "K")
    tas = convert_units_to(tas, "K")
    sum_thresh = convert_units_to(sum_thresh, "K days")

    if op in ["<", "<=", "lt", "le"]:
        c = thresh - tas
    elif op in [">", ">=", "gt", "ge"]:
        c = tas - thresh
    else:
        raise NotImplementedError(f"op: '{op}'.")

    def _exceedance_date(grp):
        strt_idx = rl.index_of_date(grp.time, after_date, max_idxs=1, default=0)
        if (
            strt_idx.size == 0
        ):  # The date is not within the group. Happens at boundaries.
            return xarray.full_like(grp.isel(time=0), np.nan, float).drop_vars("time")  # type: ignore

        return rl.first_run_after_date(
            grp.where(grp.time >= grp.time[strt_idx][0]).cumsum("time") > sum_thresh,
            window=1,
            date=None,
        )

    out = c.clip(0).resample(time=freq).map(_exceedance_date)
    out.attrs.update(units="", is_dayofyear=1, calendar=get_calendar(tas))
    return out


@declare_units(snd="[length]", thresh="[length]")
def winter_storm(
    snd: xarray.DataArray, thresh: str = "25 cm", freq: str = "AS-JUL"
) -> xarray.DataArray:
    """Days with snowfall over threshold.

    Number of days with snowfall accumulation greater or equal to threshold.

    Parameters
    ----------
    snd : xarray.DataArray
      Surface snow depth.
    thresh : str
      Threshold on snowfall accumulation require to label an event a `winter storm`.
    freq : str
      Resampling frequency.

    Returns
    -------
    xarray.DataArray
      Number of days per period identified as winter storms.

    Notes
    -----
    Snowfall accumulation is estimated by the change in snow depth.
    """
    thresh = convert_units_to(thresh, snd)

    # Compute daily accumulation
    acc = snd.diff(dim="time")

    # Winter storm condition
    out = threshold_count(acc, ">=", thresh, freq)

    out.attrs["units"] = to_agg_units(out, snd, "count")
    return out

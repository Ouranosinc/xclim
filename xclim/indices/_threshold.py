# noqa: D100
from __future__ import annotations

import warnings

import numpy as np
import xarray

from xclim.core.calendar import get_calendar, select_time
from xclim.core.missing import at_least_n_valid
from xclim.core.units import (
    convert_units_to,
    declare_units,
    pint2cfunits,
    rate2amount,
    str2pint,
    to_agg_units,
)
from xclim.core.utils import DayOfYearStr, Quantified
from xclim.indices.helpers import _gather_lat

from . import run_length as rl
from ._simple import frost_days
from .generic import (
    compare,
    cumulative_difference,
    domain_count,
    first_day_threshold_reached,
    threshold_count,
)

# Frequencies : YS: year start, QS-DEC: seasons starting in december, MS: month start
# See http://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases

# -------------------------------------------------- #
# ATTENTION: ASSUME ALL INDICES WRONG UNTIL TESTED ! #
# -------------------------------------------------- #

__all__ = [
    "calm_days",
    "cold_spell_days",
    "cold_spell_frequency",
    "cold_spell_max_length",
    "cold_spell_total_length",
    "cooling_degree_days",
    "daily_pr_intensity",
    "days_with_snow",
    "degree_days_exceedance_date",
    "dry_days",
    "dry_spell_frequency",
    "dry_spell_max_length",
    "dry_spell_total_length",
    "first_day_temperature_above",
    "first_day_temperature_below",
    "first_snowfall",
    "frost_free_season_end",
    "frost_free_season_length",
    "frost_free_season_start",
    "frost_season_length",
    "growing_degree_days",
    "growing_season_end",
    "growing_season_length",
    "growing_season_start",
    "heat_wave_index",
    "heating_degree_days",
    "hot_spell_frequency",
    "hot_spell_max_length",
    "hot_spell_total_length",
    "last_snowfall",
    "last_spring_frost",
    "maximum_consecutive_dry_days",
    "maximum_consecutive_frost_days",
    "maximum_consecutive_frost_free_days",
    "maximum_consecutive_tx_days",
    "maximum_consecutive_wet_days",
    "rprctot",
    "sea_ice_area",
    "sea_ice_extent",
    "snd_season_end",
    "snd_season_length",
    "snd_season_start",
    "snowfall_frequency",
    "snowfall_intensity",
    "snw_season_end",
    "snw_season_length",
    "snw_season_start",
    "tg_days_above",
    "tg_days_below",
    "tn_days_above",
    "tn_days_below",
    "tx_days_above",
    "tx_days_below",
    "warm_day_frequency",
    "warm_night_frequency",
    "wet_spell_frequency",
    "wet_spell_max_length",
    "wet_spell_total_length",
    "wetdays",
    "wetdays_prop",
    "windy_days",
    "winter_storm",
]


@declare_units(sfcWind="[speed]", thresh="[speed]")
def calm_days(
    sfcWind: xarray.DataArray, thresh: Quantified = "2 m s-1", freq: str = "MS"
) -> xarray.DataArray:
    r"""Calm days.

    The number of days with average near-surface wind speed below threshold (default: 2 m/s).

    Parameters
    ----------
    sfcWind : xarray.DataArray
        Daily windspeed.
    thresh : Quantified
        Threshold average near-surface wind speed on which to base evaluation.
    freq : str
        Resampling frequency.

    Returns
    -------
    xarray.DataArray, [time]
        Number of days with average near-surface wind speed below threshold.

    Notes
    -----
    Let :math:`WS_{ij}` be the windspeed at day :math:`i` of period :math:`j`. Then counted is the number of days where:

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
    thresh: Quantified = "-10 degC",
    window: int = 5,
    freq: str = "AS-JUL",
    op: str = "<",
    resample_before_rl: bool = True,
) -> xarray.DataArray:
    r"""Cold spell days.

    The number of days that are part of cold spell events, defined as a sequence of consecutive days with mean daily
    temperature below a threshold (default: -10°C).

    Parameters
    ----------
    tas : xarray.DataArray
        Mean daily temperature.
    thresh : Quantified
        Threshold temperature below which a cold spell begins.
    window : int
        Minimum number of days with temperature below threshold to qualify as a cold spell.
    freq : str
        Resampling frequency.
    op : {"<", "<=", "lt", "le"}
        Comparison operation. Default: "<".
    resample_before_rl : bool
        Determines if the resampling should take place before or after the run
        length encoding (or a similar algorithm) is applied to runs.

    Returns
    -------
    xarray.DataArray, [time]
        Cold spell days.

    Notes
    -----
    Let :math:`T_i` be the mean daily temperature on day :math:`i`, the number of cold spell days during
    period :math:`\phi` is given by:

    .. math::

       \sum_{i \in \phi} \prod_{j=i}^{i+5} [T_j < thresh]

    where :math:`[P]` is 1 if :math:`P` is true, and 0 if false.
    """
    t = convert_units_to(thresh, tas)
    over = compare(tas, op, t, constrain=("<", "<="))

    out = rl.resample_and_rl(
        over,
        resample_before_rl,
        rl.windowed_run_count,
        window=window,
        freq=freq,
    )
    return to_agg_units(out, tas, "count")


@declare_units(tas="[temperature]", thresh="[temperature]")
def cold_spell_frequency(
    tas: xarray.DataArray,
    thresh: Quantified = "-10 degC",
    window: int = 5,
    freq: str = "AS-JUL",
    op: str = "<",
    resample_before_rl: bool = True,
) -> xarray.DataArray:
    r"""Cold spell frequency.

    The number of cold spell events, defined as a sequence of consecutive {window} days
    with mean daily temperature below a {thresh}.

    Parameters
    ----------
    tas : xarray.DataArray
        Mean daily temperature.
    thresh : Quantified
        Threshold temperature below which a cold spell begins.
    window : int
        Minimum number of days with temperature below threshold to qualify as a cold spell.
    freq : str
        Resampling frequency.
    op : {"<", "<=", "lt", "le"}
        Comparison operation. Default: "<".
    resample_before_rl : bool
        Determines if the resampling should take place before or after the run

    Returns
    -------
    xarray.DataArray, [unitless]
        The {freq} number of cold periods of minimum {window} days.

    """
    t = convert_units_to(thresh, tas)
    over = compare(tas, op, t, constrain=("<", "<="))

    out = rl.resample_and_rl(
        over,
        resample_before_rl,
        rl.windowed_run_events,
        window=window,
        freq=freq,
    )
    out.attrs["units"] = ""
    return out


@declare_units(tas="[temperature]", thresh="[temperature]")
def cold_spell_max_length(
    tas: xarray.DataArray,
    thresh: Quantified = "-10 degC",
    window: int = 1,
    freq: str = "AS-JUL",
    op: str = "<",
    resample_before_rl: bool = True,
) -> xarray.DataArray:
    r"""Longest cold spell.

    Longest spell of low temperatures over a given period.
    Longest series of at least {window} consecutive days with temperature at or below {thresh}.

    Parameters
    ----------
    tas : xarray.DataArray
        Mean daily temperature.
    thresh : Quantified
        The temperature threshold needed to trigger a cold spell.
    window : int
        Minimum number of days with temperatures below thresholds to qualify as a cold spell.
    freq : str
        Resampling frequency.
    op : {"<", "<=", "lt", "le"}
        Comparison operation. Default: "<".
    resample_before_rl : bool
        Determines if the resampling should take place before or after the run
        length encoding (or a similar algorithm) is applied to runs.

    Returns
    -------
    xarray.DataArray, [days]
        The {freq} longest spell in cold periods of minimum {window} days.
    """
    thresh = convert_units_to(thresh, tas)

    cond = compare(tas, op, thresh, constrain=("<", "<="))
    max_l = rl.resample_and_rl(
        cond,
        resample_before_rl,
        rl.longest_run,
        freq=freq,
    )
    out = max_l.where(max_l >= window, 0)
    return to_agg_units(out, tas, "count")


@declare_units(tas="[temperature]", thresh="[temperature]")
def cold_spell_total_length(
    tas: xarray.DataArray,
    thresh: Quantified = "-10 degC",
    window: int = 3,
    freq: str = "AS-JUL",
    op: str = "<",
    resample_before_rl: bool = True,
) -> xarray.DataArray:
    r"""Total length of cold spells.

    Total length of spells of low temperatures over a given period.
    Total length of series of at least {window} consecutive days with temperature at or below {thresh}.

    Parameters
    ----------
    tas : xarray.DataArray
        Mean daily temperature.
    thresh : Quantified
        The temperature threshold needed to trigger a cold spell.
    window : int
        Minimum number of days with temperatures below thresholds to qualify as a cold spell.
    freq : str
        Resampling frequency.
    op : {"<", "<=", "lt", "le"}
        Comparison operation. Default: "<".
    resample_before_rl : bool
        Determines if the resampling should take place before or after the run
        length encoding (or a similar algorithm) is applied to runs.

    Returns
    -------
    xarray.DataArray, [days]
        The {freq} total number of days in cold periods of minimum {window} days.
    """
    thresh = convert_units_to(thresh, tas)

    cond = compare(tas, op, thresh, constrain=("<", "<="))
    max_l = rl.resample_and_rl(
        cond,
        resample_before_rl,
        rl.windowed_run_count,
        window=1,
        freq=freq,
    )
    out = max_l.where(max_l >= window, 0)
    return to_agg_units(out, tas, "count")


@declare_units(snd="[length]", thresh="[length]")
def snd_season_end(
    snd: xarray.DataArray,
    thresh: Quantified = "2 cm",
    window: int = 14,
    freq: str = "AS-JUL",
) -> xarray.DataArray:
    r"""End date of continuous snow depth cover.

    First day after the start of the continuous snow depth cover when snow depth is below a threshold (default: 2 cm)
    for at least `N` (default: 14) consecutive days.

    Warnings
    --------
    The default `freq` is valid for the northern hemisphere.

    Parameters
    ----------
    snd : xarray.DataArray
        Surface snow thickness.
    thresh : Quantified
        Threshold snow thickness.
    window : int
        Minimum number of days with snow depth below threshold.
    freq : str
        Resampling frequency.

    Returns
    -------
    xarray.DataArray, [dimensionless]
        First day after the start of the continuous snow depth cover when the snow depth
        goes below a threshold for a minimum duration.
        If there is no such day, returns np.nan.

    References
    ----------
    :cite:cts:`chaumont_elaboration_2017`
    """
    valid = at_least_n_valid(snd.where(snd > 0), n=1, freq=freq)

    thresh = convert_units_to(thresh, snd)
    cond = snd >= thresh

    out = (
        cond.resample(time=freq)
        .map(rl.season, window=window, dim="time", coord="dayofyear")
        .end
    )
    out.attrs.update(units="", is_dayofyear=np.int32(1), calendar=get_calendar(snd))
    return out.where(~valid)


@declare_units(snw="[mass]/[area]", thresh="[mass]/[area]")
def snw_season_end(
    snw: xarray.DataArray,
    thresh: Quantified = "20 kg m-2",
    window: int = 14,
    freq: str = "AS-JUL",
) -> xarray.DataArray:
    r"""End date of continuous snow water cover.

    First day after the start of the continuous snow water cover when snow water is below a threshold (default: 2 cm)
    for at least `N` (default: 14) consecutive days.

    Warnings
    --------
    The default `freq` is valid for the northern hemisphere.

    Parameters
    ----------
    snw : xarray.DataArray
        Surface snow amount.
    thresh : str
        Threshold snow amount.
    window : int
        Minimum number of days with snow water below threshold.
    freq : str
        Resampling frequency.

    Returns
    -------
    xarray.DataArray, [dimensionless]
        First day after the start of the continuous snow water cover when the snow water
        goes below a threshold for a minimum duration.
        If there is no such day, returns np.nan.

    References
    ----------
    :cite:cts:`chaumont_elaboration_2017`
    """
    valid = at_least_n_valid(snw.where(snw > 0), n=1, freq=freq)

    thresh = convert_units_to(thresh, snw)
    cond = snw >= thresh

    out = (
        cond.resample(time=freq)
        .map(rl.season, window=window, dim="time", coord="dayofyear")
        .end
    )
    out.attrs.update(units="", is_dayofyear=np.int32(1), calendar=get_calendar(snw))
    return out.where(~valid)


@declare_units(snd="[length]", thresh="[length]")
def snd_season_start(
    snd: xarray.DataArray,
    thresh: Quantified = "2 cm",
    window: int = 14,
    freq: str = "AS-JUL",
) -> xarray.DataArray:
    r"""Start date of continuous snow depth cover.

    Day of year when snow depth is above or equal to a threshold (default: 2 cm)
    for at least `N` (default: 14) consecutive days.

    Warnings
    --------
    The default `freq` is valid for the northern hemisphere.

    Parameters
    ----------
    snd : xarray.DataArray
        Surface snow thickness.
    thresh : Quantified
        Threshold snow thickness.
    window : int
        Minimum number of days with snow depth above or equal to threshold.
    freq : str
        Resampling frequency.

    Returns
    -------
    xarray.DataArray, [dimensionless]
        First day of the year when the snow depth is superior to a threshold for a minimum duration.
        If there is no such day, returns np.nan.

    References
    ----------
    :cite:cts:`chaumont_elaboration_2017`
    """
    valid = at_least_n_valid(snd.where(snd > 0), n=1, freq=freq)

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
    out.attrs.update(units="", is_dayofyear=np.int32(1), calendar=get_calendar(snd))
    return out.where(~valid)


@declare_units(snw="[mass]/[area]", thresh="[mass]/[area]")
def snw_season_start(
    snw: xarray.DataArray,
    thresh: Quantified = "20 kg m-2",
    window: int = 14,
    freq: str = "AS-JUL",
) -> xarray.DataArray:
    r"""Start date of continuous snow water cover.

    Day of year when snow water is above or equal to a threshold (default: 2 cm)
    for at least `N` (default: 14) consecutive days.

    Warnings
    --------
    The default `freq` is valid for the northern hemisphere.

    Parameters
    ----------
    snw : xarray.DataArray
        Surface snow amount.
    thresh : str
        Threshold snow amount.
    window : int
        Minimum number of days with snow water above or equal to threshold.
    freq : str
        Resampling frequency.

    Returns
    -------
    xarray.DataArray, [dimensionless]
        First day of the year when the snow water is superior to a threshold for a minimum duration.
        If there is no such day, returns np.nan.

    References
    ----------
    :cite:cts:`chaumont_elaboration_2017`
    """
    valid = at_least_n_valid(snw.where(snw > 0), n=1, freq=freq)

    thresh = convert_units_to(thresh, snw)
    cond = snw >= thresh

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
    out.attrs.update(units="", is_dayofyear=np.int32(1), calendar=get_calendar(snw))
    return out.where(~valid)


@declare_units(pr="[precipitation]", thresh="[precipitation]")
def daily_pr_intensity(
    pr: xarray.DataArray,
    thresh: Quantified = "1 mm/day",
    freq: str = "YS",
    op: str = ">=",
) -> xarray.DataArray:
    r"""Average daily precipitation intensity.

    Return the average precipitation over wet days.
    Wet days are those with precipitation over a given threshold (default: 1 mm/day).

    Parameters
    ----------
    pr : xarray.DataArray
        Daily precipitation.
    thresh : Quantified
        Precipitation value over which a day is considered wet.
    freq : str
        Resampling frequency.
    op : {">", ">=", "gt", "ge"}
        Comparison operation. Default: ">=".

    Returns
    -------
    xarray.DataArray, [precipitation]
        The average precipitation over wet days for each period.

    Notes
    -----
    Let :math:`\mathbf{p} = p_0, p_1, \ldots, p_n` be the daily precipitation and :math:`thresh` be the precipitation
    threshold defining wet days. Then the daily precipitation intensity is defined as:

    .. math::

       \frac{\sum_{i=0}^n p_i [p_i \leq thresh]}{\sum_{i=0}^n [p_i \leq thresh]}

    where :math:`[P]` is 1 if :math:`P` is true, and 0 if false.

    Examples
    --------
    The following would compute for each grid cell of file `pr.day.nc` the average precipitation fallen over days with
    precipitation >= 5 mm at seasonal frequency, i.e. DJF, MAM, JJA, SON, DJF, etc.:

    >>> from xclim.indices import daily_pr_intensity
    >>> pr = xr.open_dataset(path_to_pr_file).pr
    >>> daily_int = daily_pr_intensity(pr, thresh="5 mm/day", freq="QS-DEC")
    """
    t = convert_units_to(thresh, pr, "hydro")

    # Get amount of rain (not rate)
    pram = rate2amount(pr)

    # Comparison
    comparison = compare(pr, op, t, constrain=(">", ">="))

    # put pram = 0 for non wet-days
    pram_wd = xarray.where(comparison, pram, 0)

    # sum over wanted period
    s = pram_wd.resample(time=freq).sum(dim="time")

    # get number of wetdays over period
    wd = wetdays(pr, thresh=thresh, freq=freq)
    out = s / wd
    out.attrs["units"] = f"{str2pint(pram.units) / str2pint(wd.units):~}"
    return out


@declare_units(pr="[precipitation]", thresh="[precipitation]")
def dry_days(
    pr: xarray.DataArray,
    thresh: Quantified = "0.2 mm/d",
    freq: str = "YS",
    op: str = "<",
) -> xarray.DataArray:
    r"""Dry days.

    The number of days with daily precipitation below threshold.

    Parameters
    ----------
    pr : xarray.DataArray
        Daily precipitation.
    thresh : Quantified
        Threshold temperature on which to base evaluation.
    freq : str
        Resampling frequency.
    op : {"<", "<=", "lt", "le"}
        Comparison operation. Default: "<".

    Returns
    -------
    xarray.DataArray, [time]
         Number of days with daily precipitation {op} threshold.

    Notes
    -----
    Let :math:`PR_{ij}` be the daily precipitation at day :math:`i` of period :math:`j`. Then counted is the number
    of days where:

    .. math::

        \sum PR_{ij} < Threshold [mm/day]
    """
    thresh = convert_units_to(thresh, pr, context="hydro")
    out = threshold_count(pr, op, thresh, freq, constrain=("<", "<="))
    out = to_agg_units(out, pr, "count")
    return out


# NOTE : A spell index could be used below
@declare_units(pr="[precipitation]", thresh="[precipitation]")
def maximum_consecutive_wet_days(
    pr: xarray.DataArray,
    thresh: Quantified = "1 mm/day",
    freq: str = "YS",
    resample_before_rl: bool = True,
) -> xarray.DataArray:
    r"""Consecutive wet days.

    Returns the maximum number of consecutive days with precipitation above a given threshold (default: 1 mm/day).

    Parameters
    ----------
    pr : xarray.DataArray
        Mean daily precipitation flux.
    thresh : Quantified
        Threshold precipitation on which to base evaluation.
    freq : str
      Resampling frequency.
    resample_before_rl : bool
      Determines if the resampling should take place before or after the run
      length encoding (or a similar algorithm) is applied to runs.

    Returns
    -------
    xarray.DataArray, [time]
        The maximum number of consecutive wet days.

    Notes
    -----
    Let :math:`\mathbf{x}=x_0, x_1, \ldots, x_n` be a daily precipitation series and :math:`\mathbf{s}` be the sorted
    vector of indices :math:`i` where :math:`[p_i > thresh] \neq [p_{i+1} > thresh]`, that is, the days where the
    precipitation crosses the *wet day* threshold. Then the maximum number of consecutive wet days is given by:

    .. math::

       \max(\mathbf{d}) \quad \mathrm{where} \quad d_j = (s_j - s_{j-1}) [x_{s_j} > 0^\circ C]

    where :math:`[P]` is 1 if :math:`P` is true, and 0 if false. Note that this formula does not handle sequences at
    the start and end of the series, but the numerical algorithm does.
    """
    thresh = convert_units_to(thresh, pr, "hydro")

    cond = pr > thresh
    out = rl.resample_and_rl(
        cond,
        resample_before_rl,
        rl.longest_run,
        freq=freq,
    )
    out = to_agg_units(out, pr, "count")
    return out


@declare_units(tas="[temperature]", thresh="[temperature]")
def cooling_degree_days(
    tas: xarray.DataArray, thresh: Quantified = "18 degC", freq: str = "YS"
) -> xarray.DataArray:
    r"""Cooling degree days.

    Returns the sum of degree days above the temperature threshold at which spaces are cooled (default: 18℃).

    Parameters
    ----------
    tas : xarray.DataArray
        Mean daily temperature.
    thresh : Quantified
        Temperature threshold above which air is cooled.
    freq : str
        Resampling frequency.

    Returns
    -------
    xarray.DataArray, [time][temperature]
        Cooling degree days.

    Notes
    -----
    Let :math:`x_i` be the daily mean temperature at day :math:`i`. Then the cooling degree days above
    temperature threshold :math:`thresh` over period :math:`\phi` is given by:

    .. math::

        \sum_{i \in \phi} (x_{i}-{thresh} [x_i > thresh]

    where :math:`[P]` is 1 if :math:`P` is true, and 0 if false.
    """
    return cumulative_difference(tas, threshold=thresh, op=">", freq=freq)


@declare_units(tas="[temperature]", thresh="[temperature]")
def growing_degree_days(
    tas: xarray.DataArray, thresh: Quantified = "4.0 degC", freq: str = "YS"
) -> xarray.DataArray:
    r"""Growing degree-days over threshold temperature value.

    The sum of growing degree-days over a given mean daily temperature threshold (default: 4℃).

    Parameters
    ----------
    tas : xarray.DataArray
        Mean daily temperature.
    thresh : Quantified
        Threshold temperature on which to base evaluation.
    freq : str
        Resampling frequency.

    Returns
    -------
    xarray.DataArray, [time][temperature]
        The sum of growing degree-days above a given threshold.

    Notes
    -----
    Let :math:`TG_{ij}` be the mean daily temperature at day :math:`i` of period :math:`j`. Then the
    growing degree days are:

    .. math::

        GD4_j = \sum_{i=1}^I (TG_{ij}-{4} | TG_{ij} > {4}℃)
    """
    return cumulative_difference(tas, threshold=thresh, op=">", freq=freq)


@declare_units(tas="[temperature]", thresh="[temperature]")
def growing_season_start(
    tas: xarray.DataArray,
    thresh: Quantified = "5.0 degC",
    window: int = 5,
    freq: str = "YS",
) -> xarray.DataArray:
    r"""Start of the growing season.

    Day of the year of the start of a sequence of days with mean daily temperatures consistently above or equal to a
    given threshold (default: 5℃).

    Parameters
    ----------
    tas : xarray.DataArray
        Mean daily temperature.
    thresh : Quantified
        Threshold temperature on which to base evaluation.
    window : int
        Minimum number of days with temperature above threshold needed for evaluation.
    freq : str
        Resampling frequency.

    Returns
    -------
    xarray.DataArray, [dimensionless]
        Day of the year when temperature is superior to a threshold over a given number of days for the first time.
        If there is no such day or if a growing season is not detected, returns np.nan.

    Notes
    -----
    Let :math:`x_i` be the daily mean temperature at day of the year :math:`i` for values of :math:`i` going from 1
    to 365 or 366. The start date of the start of growing season is given by the smallest index :math:`i`:

    .. math::

       \prod_{j=i}^{i+w} [x_j >= thresh]

    where :math:`w` is the number of days the temperature threshold should be met or exceeded,
    and :math:`[P]` is 1 if :math:`P` is true, and 0 if false.
    """
    thresh = convert_units_to(thresh, tas)
    over = tas >= thresh
    out = over.resample(time=freq).map(rl.first_run, window=window, coord="dayofyear")
    out.attrs.update(units="", is_dayofyear=np.int32(1), calendar=get_calendar(tas))
    return out


@declare_units(tas="[temperature]", thresh="[temperature]")
def growing_season_end(
    tas: xarray.DataArray,
    thresh: Quantified = "5.0 degC",
    mid_date: DayOfYearStr = "07-01",
    window: int = 5,
    freq: str = "YS",
) -> xarray.DataArray:
    r"""End of the growing season.

    Day of the year of the start of a sequence of `N` (default: 5) days with mean temperatures consistently below a
    given threshold (default: 5℃), occurring after a given calendar date (default: July 1).

    Warnings
    --------
    The default `freq` and `mid_date` parameters are valid for the northern hemisphere.

    Parameters
    ----------
    tas : xarray.DataArray
        Mean daily temperature.
    thresh : Quantified
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

    Notes
    -----
    Let :math:`x_i` be the daily mean temperature at day of the year :math:`i` for values of :math:`i` going from 1
    to 365 or 366. The start date of the end of growing season is given by the smallest index :math:`i`:

    .. math::

       \prod_{j=i}^{i+w} [x_j < thresh]

    where :math:`w` is the number of days where temperature should be inferior to a given threshold after a given date,
    and :math:`[P]` is 1 if :math:`P` is true, and 0 if false.
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
    out.attrs.update(units="", is_dayofyear=np.int32(1), calendar=get_calendar(tas))
    return out


@declare_units(tas="[temperature]", thresh="[temperature]")
def growing_season_length(
    tas: xarray.DataArray,
    thresh: Quantified = "5.0 degC",
    window: int = 6,
    mid_date: DayOfYearStr = "07-01",
    freq: str = "YS",
    op: str = ">=",
) -> xarray.DataArray:
    r"""Growing season length.

    The number of days between the first occurrence of at least `N` (default: 6) consecutive days with mean daily
    temperature over a threshold (default: 5℃) and the first occurrence of at least `N` consecutive days with mean
    daily temperature below the same threshold after a certain date, usually July 1st (06-01) in the northern emispher
    and January 1st (01-01) in the southern hemisphere.

    Warnings
    --------
    The default `freq` and `mid_date` parameters are valid for the northern hemisphere.

    Parameters
    ----------
    tas : xarray.DataArray
        Mean daily temperature.
    thresh : Quantified
        Threshold temperature on which to base evaluation.
    window : int
        Minimum number of days with temperature above threshold to mark the beginning and end of growing season.
    mid_date : str
        Date of the year after which to look for the end of the season. Should have the format '%m-%d'.
    freq : str
        Resampling frequency.
    op : {">", ">=", "gt", "ge"}
        Comparison operation. Default: ">=".

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

    For the Northern Hemisphere:

    >>> gsl_nh = growing_season_length(tas, mid_date="07-01", freq="AS")

    If working in the Southern Hemisphere, one can use:

    >>> gsl_sh = growing_season_length(tas, mid_date="01-01", freq="AS-JUL")

    References
    ----------
    :cite:cts:`project_team_eca&d_algorithm_2013`

    """
    thresh = convert_units_to(thresh, tas)
    cond = compare(tas, op, thresh, constrain=(">=", ">"))

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
    mid_date: DayOfYearStr | None = "01-01",
    thresh: Quantified = "0.0 degC",
    freq: str = "AS-JUL",
    op: str = "<",
) -> xarray.DataArray:
    r"""Frost season length.

    The number of days between the first occurrence of at least `N` (default: 5) consecutive days with minimum daily
    temperature under a threshold (default: 0℃) and the first occurrence of at least `N` consecutive days with
    minimum daily temperature above the same threshold.
    A mid-date can be given to limit the earliest day the end of season can take.

    Warnings
    --------
    The default `freq` and `mid_date` parameters are valid for the northern hemisphere.

    Parameters
    ----------
    tasmin : xarray.DataArray
        Minimum daily temperature.
    window : int
        Minimum number of days with temperature below threshold to mark the beginning and end of frost season.
    mid_date : str, optional
        Date the must be included in the season. It is the earliest the end of the season can be.
        If None, there is no limit.
    thresh : Quantified
        Threshold temperature on which to base evaluation.
    freq : str
        Resampling frequency.
    op : {"<", "<=", "lt", "le"}
        Comparison operation. Default: "<".

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

    For the Northern Hemisphere:

    >>> fsl_nh = frost_season_length(tasmin, freq="AS-JUL")

    If working in the Southern Hemisphere, one can use:

    >>> fsl_sh = frost_season_length(tasmin, freq="YS")
    """
    thresh = convert_units_to(thresh, tasmin)
    cond = compare(tasmin, op, thresh, constrain=("<=", "<"))

    out = cond.resample(time=freq).map(
        rl.season_length,
        window=window,
        date=mid_date,
        dim="time",
    )
    return to_agg_units(out, tasmin, "count")


@declare_units(tasmin="[temperature]", thresh="[temperature]")
def frost_free_season_start(
    tasmin: xarray.DataArray,
    thresh: Quantified = "0.0 degC",
    window: int = 5,
    freq: str = "YS",
) -> xarray.DataArray:
    r"""Start of the frost free season.

    Day of the year of the start of a sequence of days with minimum temperatures consistently above or equal to a
    threshold (default: 0℃), after a period of `N` days (default: 5) with minimum temperatures consistently
    above the same threshold.

    Parameters
    ----------
    tasmin : xarray.DataArray
        Minimum daily temperature.
    thresh : Quantified
        Threshold temperature on which to base evaluation.
    window : int
        Minimum number of days with temperature above threshold needed for evaluation.
    freq : str
        Resampling frequency.

    Returns
    -------
    xarray.DataArray, [dimensionless]
        Day of the year when minimum temperature is superior to a threshold
        over a given number of days for the first time.
        If there is no such day or if a frost free season is not detected, returns np.nan.

    Notes
    -----
    Let :math:`x_i` be the daily mean temperature at day of the year :math:`i` for values of :math:`i` going from 1
    to 365 or 366. The start date of the start of growing season is given by the smallest index :math:`i`:

    .. math::

       \prod_{j=i}^{i+w} [x_j >= thresh]

    where :math:`w` is the number of days the temperature threshold should be met or exceeded,
    and :math:`[P]` is 1 if :math:`P` is true, and 0 if false.
    """
    thresh = convert_units_to(thresh, tasmin)
    over = tasmin >= thresh
    out = over.resample(time=freq).map(rl.first_run, window=window, coord="dayofyear")
    out.attrs.update(units="", is_dayofyear=np.int32(1), calendar=get_calendar(tasmin))
    return out


@declare_units(tasmin="[temperature]", thresh="[temperature]")
def frost_free_season_end(
    tasmin: xarray.DataArray,
    thresh: Quantified = "0.0 degC",
    mid_date: DayOfYearStr = "07-01",
    window: int = 5,
    freq: str = "YS",
) -> xarray.DataArray:
    r"""End of the frost free season.

    Day of the year of the start of a sequence of days with minimum temperatures consistently below a threshold
    (default: 0℃), after a period of `N` days (default: 5) with minimum temperatures consistently above the same
    threshold.

    Warnings
    --------
    The default `freq` and `mid_date` parameters are valid for the northern hemisphere.

    Parameters
    ----------
    tasmin : xarray.DataArray
        Minimum daily temperature.
    thresh : Quantified
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
        Day of the year when minimum temperature is inferior to a threshold over a given number of days for the first time.
        If there is no such day or if a frost free season is not detected, returns np.nan.
        If the frost free season does not end within the time period, returns the last day of the period.
    """
    thresh = convert_units_to(thresh, tasmin)
    cond = tasmin >= thresh

    out = cond.resample(time=freq).map(
        rl.run_end_after_date,
        window=window,
        date=mid_date,
        dim="time",
        coord="dayofyear",
    )
    out.attrs.update(units="", is_dayofyear=np.int32(1), calendar=get_calendar(tasmin))
    return out


@declare_units(tasmin="[temperature]", thresh="[temperature]")
def frost_free_season_length(
    tasmin: xarray.DataArray,
    window: int = 5,
    mid_date: DayOfYearStr | None = "07-01",
    thresh: Quantified = "0.0 degC",
    freq: str = "YS",
    op: str = ">=",
) -> xarray.DataArray:
    r"""Frost free season length.

    The number of days between the first occurrence of at least `N` (default: 5) consecutive days with minimum daily
    temperature above a threshold (default: 0℃) and the first occurrence of at least `N` consecutive days with
    minimum daily temperature below the same threshold.
    A mid-date can be given to limit the earliest day the end of season can take.

    Warnings
    --------
    The default `freq` and `mid_date` parameters are valid for the northern hemisphere.

    Parameters
    ----------
    tasmin : xarray.DataArray
        Minimum daily temperature.
    window : int
        Minimum number of days with temperature above threshold to mark the beginning and end of frost free season.
    mid_date : str, optional
        Date the must be included in the season. It is the earliest the end of the season can be.
        If None, there is no limit.
    thresh : Quantified
        Threshold temperature on which to base evaluation.
    freq : str
        Resampling frequency.
    op : {">", ">=", "gt", "ge"}
        Comparison operation. Default: ">=".

    Returns
    -------
    xarray.DataArray, [time]
        Frost free season length.

    Notes
    -----
    Let :math:`TN_{ij}` be the minimum temperature at day :math:`i` of period :math:`j`. Then counted is
    the number of days between the first occurrence of at least N consecutive days with:

    .. math::

        TN_{ij} >= 0 ℃

    and the first subsequent occurrence of at least N consecutive days with:

    .. math::

        TN_{ij} < 0 ℃

    Examples
    --------
    >>> from xclim.indices import frost_season_length
    >>> tasmin = xr.open_dataset(path_to_tasmin_file).tasmin

    For the Northern Hemisphere:

    >>> ffsl_nh = frost_free_season_length(tasmin, freq="YS")

    If working in the Southern Hemisphere, one can use:

    >>> ffsl_sh = frost_free_season_length(tasmin, freq="AS-JUL")
    """
    thresh = convert_units_to(thresh, tasmin)
    cond = compare(tasmin, op, thresh, constrain=(">=", ">"))

    out = cond.resample(time=freq).map(
        rl.season_length,
        window=window,
        date=mid_date,
        dim="time",
    )
    return to_agg_units(out, tasmin, "count")


# FIXME: `tas` should instead be `tasmin` if we want to follow expected definitions.
@declare_units(tasmin="[temperature]", thresh="[temperature]")
def last_spring_frost(
    tasmin: xarray.DataArray,
    thresh: Quantified = "0 degC",
    op: str = "<",
    before_date: DayOfYearStr = "07-01",
    window: int = 1,
    freq: str = "YS",
) -> xarray.DataArray:
    r"""Last day of temperatures inferior to a threshold temperature.

    Returns last day of period where minimum temperature is inferior to a threshold over a given number of days
    (default: 1) and limited to a final calendar date (default: July 1).

    Warnings
    --------
    The default `freq` and `before_date` parameters are valid for the northern hemisphere.

    Parameters
    ----------
    tasmin : xarray.DataArray
        Mean daily temperature.
    thresh : Quantified
        Threshold temperature on which to base evaluation.
    op : {"<", "<=", "lt", "le"}
        Comparison operation. Default: "<".
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
    thresh = convert_units_to(thresh, tasmin)
    cond = compare(tasmin, op, thresh, constrain=("<", "<="))

    out = cond.resample(time=freq).map(
        rl.last_run_before_date,
        window=window,
        date=before_date,
        dim="time",
        coord="dayofyear",
    )
    out.attrs.update(units="", is_dayofyear=np.int32(1), calendar=get_calendar(tasmin))
    return out


@declare_units(tas="[temperature]", thresh="[temperature]")
def first_day_temperature_below(
    tas: xarray.DataArray,
    thresh: Quantified = "0 degC",
    op: str = "<",
    after_date: DayOfYearStr = "07-01",
    window: int = 1,
    freq: str = "YS",
) -> xarray.DataArray:
    r"""First day of temperatures inferior to a given temperature threshold.

    Returns first day of period where temperature is inferior to a threshold over a given number of days (default: 1),
    limited to a starting calendar date (default: July 1).

    Warnings
    --------
    The default `freq` and `after_date` parameters are valid for the northern hemisphere.

    Parameters
    ----------
    tas : xarray.DataArray
        Daily temperature.
    thresh : Quantified
        Threshold temperature on which to base evaluation.
    op : {"<", "<=", "lt", "le"}
        Comparison operation. Default: ">".
    after_date : str
        Date of the year after which to look for the first event. Should have the format '%m-%d'.
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
    # noqa

    return first_day_threshold_reached(
        tas,
        threshold=thresh,
        op=op,
        after_date=after_date,
        window=window,
        freq=freq,
        constrain=("<", "<="),
    )


@declare_units(tas="[temperature]", thresh="[temperature]")
def first_day_temperature_above(
    tas: xarray.DataArray,
    thresh: Quantified = "0 degC",
    op: str = ">",
    after_date: DayOfYearStr = "01-01",
    window: int = 1,
    freq: str = "YS",
) -> xarray.DataArray:
    r"""First day of temperatures superior to a given temperature threshold.

    Returns first day of period where temperature is superior to a threshold over a given number of days (default: 1),
    limited to a starting calendar date (default: January 1).

    Warnings
    --------
    The default `freq` and `after_date` parameters are valid for the northern hemisphere.

    Parameters
    ----------
    tas : xarray.DataArray
        Daily temperature.
    thresh : Quantified
        Threshold temperature on which to base evaluation.
    op : {">", ">=", "gt", "ge"}
        Comparison operation. Default: ">".
    after_date : str
        Date of the year after which to look for the first event. Should have the format '%m-%d'.
    window : int
        Minimum number of days with temperature above threshold needed for evaluation.
    freq : str
        Resampling frequency.

    Returns
    -------
    xarray.DataArray, [dimensionless]
        Day of the year when temperature is superior to a threshold over a given number of days for the first time.
        If there is no such day, returns np.nan.

    Notes
    -----
    Let :math:`x_i` be the daily mean|max|min temperature at day of the year :math:`i` for values of :math:`i` going
    from 1 to 365 or 366. The first day above temperature threshold is given by the smallest index :math:`i`:

    .. math::

       \prod_{j=i}^{i+w} [x_j > thresh]

    where :math:`w` is the number of days the temperature threshold should be exceeded, and :math:`[P]` is
    1 if :math:`P` is true, and 0 if false.
    """
    return first_day_threshold_reached(
        tas,
        threshold=thresh,
        op=op,
        after_date=after_date,
        window=window,
        freq=freq,
        constrain=(">", ">="),
    )


@declare_units(prsn="[precipitation]", thresh="[precipitation]")
def first_snowfall(
    prsn: xarray.DataArray,
    thresh: Quantified = "UNSET",
    freq: str = "AS-JUL",
) -> xarray.DataArray:
    r"""First day with snowfall rate above a threshold.

    Returns the first day of a period where snowfall exceeds a threshold (current default: 0.5 mm/day
    liquid water equivalent snowfall rate. xclim >=0.45.0 default: 1 mm/day).

    Warnings
    --------
    The default `freq` is valid for the northern hemisphere.

    Parameters
    ----------
    prsn : xarray.DataArray
        Snowfall flux.
    thresh : Quantified
        Threshold snowfall flux or liquid water equivalent snowfall rate. (Current default: 0.5 mm/day liquid water equivalent snowfall rate.
        xclim >=0.45.0 default: 1 mm/day)
    freq : str
        Resampling frequency.

    Returns
    -------
    xarray.DataArray
        Last day of the year where snowfall is superior to a threshold.
        If there is no such day, returns np.nan.

    References
    ----------
    :cite:cts:`cbcl_climate_2020`.

    Notes
    -----
    The 1 mm/day liquid water equivalent snowfall rate threshold in :cite:cts:`frei_snowfall_2018` corresponds
    to the 1 cm/day snowfall rate threshold  in :cite:cts:`cbcl_climate_2020` using a snow denstiy of 100 kg/m**3.

    If threshold and prsn differ by a density (i.e. [length/time] vs. [mass/area/time]), a liquid water equivalent
    snowfall rate is assumed and the threshold is converted using a 1000 kg m-3 density.

    The current default threshold "UNSET" is a placeholder and will be changed to the default 1 mm/day  in xclim>=0.45.0.
    """
    if thresh == "UNSET":
        warnings.warn(
            "The default value for this indicator will change in xclim>=0.45.0,  from `0.5 mm/day` to `1 mm/day`. Using `0.5 mm/day` for now."
        )
        thresh = "0.5 mm/day"
    thresh = convert_units_to(thresh, prsn, context="hydro")
    cond = prsn >= thresh

    out = cond.resample(time=freq).map(
        rl.first_run,
        window=1,
        dim="time",
        coord="dayofyear",
    )
    out.attrs.update(units="", is_dayofyear=np.int32(1), calendar=get_calendar(prsn))
    return out


@declare_units(prsn="[precipitation]", thresh="[precipitation]")
def last_snowfall(
    prsn: xarray.DataArray,
    thresh: Quantified = "UNSET",
    freq: str = "AS-JUL",
) -> xarray.DataArray:
    r"""Last day with snowfall above a threshold.

    Returns the last day of a period where snowfall exceeds a threshold (current default: 0.5 mm/day liquid water equivalent snowfall rate.
    xclim >=0.45.0 default: 1 mm/day).

    Warnings
    --------
    The default `freq` is valid for the northern hemisphere.

    Parameters
    ----------
    prsn : xarray.DataArray
        Snowfall flux.
    thresh : Quantified
        Threshold snowfall flux or liquid water equivalent snowfall rate. (Current default: 0.5 mm/day liquid water equivalent snowfall rate.
        xclim >=0.45.0 default: 1 mm/day)
    freq : str
        Resampling frequency.

    Returns
    -------
    xarray.DataArray
        Last day of the year where snowfall is superior to a threshold.
        If there is no such day, returns np.nan.


    References
    ----------
    :cite:cts:`cbcl_climate_2020`.

    Notes
    -----
    The 1 mm/day liquid water equivalent snowfall rate threshold in :cite:cts:`frei_snowfall_2018` corresponds
    to the 1 cm/day snowfall rate threshold  in :cite:cts:`cbcl_climate_2020` using a snow denstiy of 100 kg/m**3.

    If threshold and prsn differ by a density (i.e. [length/time] vs. [mass/area/time]), a liquid water equivalent
    snowfall rate is assumed and the threshold is converted using a 1000 kg m-3 density.

    The current default threshold "UNSET" is a placeholder and will be changed to the default 1 mm/day  in xclim>=0.45.0.
    """
    if thresh == "UNSET":
        warnings.warn(
            "The default value for this indicator will change in xclim>=0.45.0,  from `0.5 mm/day` to `1 mm/day`. Using `0.5 mm/day` for now."
        )
        thresh = "0.5 mm/day"
    thresh = convert_units_to(thresh, prsn, context="hydro")
    cond = prsn >= thresh

    out = cond.resample(time=freq).map(
        rl.last_run,
        window=1,
        dim="time",
        coord="dayofyear",
    )
    out.attrs.update(units="", is_dayofyear=np.int32(1), calendar=get_calendar(prsn))
    return out


@declare_units(
    prsn="[precipitation]",
    low="[precipitation]",
    high="[precipitation]",
)
def days_with_snow(
    prsn: xarray.DataArray,
    low: Quantified = "0 kg m-2 s-1",
    high: Quantified = "1E6 kg m-2 s-1",
    freq: str = "AS-JUL",
) -> xarray.DataArray:
    r"""Days with snow.

    Return the number of days where snowfall is within low and high thresholds.

    Warnings
    --------
    The default `freq` is valid for the northern hemisphere.

    Parameters
    ----------
    prsn : xarray.DataArray
        Snowfall flux
    low : Quantified
        Minimum threshold snowfall flux or liquid water equivalent snowfall rate.
    high : Quantified
        Maximum threshold snowfall flux or liquid water equivalent snowfall rate.
    freq : str
        Resampling frequency.

    Returns
    -------
    xarray.DataArray, [days]
        Number of days where snowfall is between low and high thresholds.

    References
    ----------
    :cite:cts:`matthews_planning_2017`

    Notes
    -----
    If threshold and prsn differ by a density (i.e. [length/time] vs. [mass/area/time]), a liquid water equivalent
    snowfall rate is assumed and the threshold is converted using a 1000 kg m-3 density.
    """
    low = convert_units_to(low, prsn, context="hydro")
    high = convert_units_to(high, prsn, context="hydro")
    out = domain_count(prsn, low, high, freq)
    return to_agg_units(out, prsn, "count")


@declare_units(prsn="[precipitation]", thresh="[precipitation]")
def snowfall_frequency(
    prsn: xarray.DataArray,
    thresh: Quantified = "1 mm/day",
    freq: str = "AS-JUL",
) -> xarray.DataArray:
    r"""Percentage of snow days.

    Return the percentage of days where snowfall exceeds a threshold (default: 1 mm/day).

    Warnings
    --------
    The default `freq` is valid for the northern hemisphere.

    Parameters
    ----------
    prsn : xarray.DataArray
        Snowfall flux.
    thresh : Quantified
        Threshold snowfall flux or liquid water equivalent snowfall rate (default: 1 mm/day).
    freq : str
        Resampling frequency.

    Returns
    -------
    xarray.DataArray, [%]
        Percentage of days where snowfall exceeds a threshold.

    References
    ----------
    :cite:cts:`frei_snowfall_2018`

    Notes
    -----
    The 1 mm/day liquid water equivalent snowfall rate threshold in :cite:cts:`frei_snowfall_2018` corresponds
    to the 1 cm/day snowfall rate threshold  in :cite:cts:`cbcl_climate_2020` using a snow denstiy of 100 kg/m**3.

    If threshold and prsn differ by a density (i.e. [length/time] vs. [mass/area/time]), a liquid water equivalent
    snowfall rate is assumed and the threshold is converted using a 1000 kg m-3 density.
    """
    # High threshold here just needs to be a big value. It is converted to same units as
    # so that a warning message won't be triggered just because of this value
    thresh_units = pint2cfunits(str2pint(thresh))
    high = f"{convert_units_to('1E6 kg m-2 s-1', thresh_units, context='hydro')} {thresh_units}"
    snow_days = days_with_snow(prsn, low=thresh, high=high, freq=freq)
    total_days = prsn.resample(time=freq).count(dim="time")
    snow_freq = snow_days / total_days * 100
    snow_freq = snow_freq.assign_attrs(**snow_days.attrs)
    snow_freq.attrs["units"] = "%"
    return snow_freq


@declare_units(prsn="[precipitation]", thresh="[precipitation]")
def snowfall_intensity(
    prsn: xarray.DataArray,
    thresh: Quantified = "1 mm/day",
    freq: str = "AS-JUL",
) -> xarray.DataArray:
    r"""Mean daily snowfall rate during snow days.

    Return mean daily snowfall rate during days where snowfall exceeds a threshold (default: 1 mm/day).

    Warnings
    --------
    The default `freq` is valid for the northern hemisphere.

    Parameters
    ----------
    prsn : xarray.DataArray
        Snowfall flux.
    thresh : Quantified
        Threshold snowfall flux or liquid water equivalent snowfall rate (default: 1 mm/day).
    freq : str
        Resampling frequency.

    Returns
    -------
    xarray.DataArray,
        Mean daily liquid water equivalent snowfall rate during days where snowfall exceeds a threshold.

    References
    ----------
    :cite:cts:`frei_snowfall_2018`

    Notes
    -----
    The 1 mm/day liquid water equivalent snowfall rate threshold in :cite:cts:`frei_snowfall_2018` corresponds
    to the 1 cm/day snowfall rate threshold  in :cite:cts:`cbcl_climate_2020` using a snow denstiy of 100 kg/m**3.

    If threshold and prsn differ by a density (i.e. [length/time] vs. [mass/area/time]), a liquid water equivalent
    snowfall rate is assumed and the threshold is converted using a 1000 kg m-3 density.
    """
    thresh = convert_units_to(thresh, "mm/day", context="hydro")
    lwe_prsn = convert_units_to(prsn, "mm/day", context="hydro")

    cond = lwe_prsn >= thresh
    mean = lwe_prsn.where(cond).resample(time=freq).mean(dim="time")
    out = mean.fillna(0)

    return out.assign_attrs(units=lwe_prsn.units)


@declare_units(tasmax="[temperature]", thresh="[temperature]")
def heat_wave_index(
    tasmax: xarray.DataArray,
    thresh: Quantified = "25.0 degC",
    window: int = 5,
    freq: str = "YS",
    op: str = ">",
    resample_before_rl: bool = True,
) -> xarray.DataArray:
    """Heat wave index.

    Number of days that are part of a heatwave, defined as five or more consecutive days over a threshold of 25℃.

    Parameters
    ----------
    tasmax : xarray.DataArray
        Maximum daily temperature.
    thresh : Quantified
        Threshold temperature on which to designate a heatwave.
    window : int
        Minimum number of days with temperature above threshold to qualify as a heatwave.
    freq : str
        Resampling frequency.
    op : {">", ">=", "gt", "ge"}
        Comparison operation. Default: ">".
    resample_before_rl : bool
        Determines if the resampling should take place before or after the run
        length encoding (or a similar algorithm) is applied to runs.

    Returns
    -------
    DataArray, [time]
        Heat wave index.
    """
    thresh = convert_units_to(thresh, tasmax)
    over = compare(tasmax, op, thresh, constrain=(">", ">="))
    out = rl.resample_and_rl(
        over,
        resample_before_rl,
        rl.windowed_run_count,
        window=window,
        freq=freq,
    )
    return to_agg_units(out, tasmax, "count")


@declare_units(tas="[temperature]", thresh="[temperature]")
def heating_degree_days(
    tas: xarray.DataArray,
    thresh: Quantified = "17.0 degC",
    freq: str = "YS",
) -> xarray.DataArray:
    r"""Heating degree days.

    Sum of degree days below the temperature threshold (default: 17℃) at which spaces are heated.

    Parameters
    ----------
    tas : xarray.DataArray
        Mean daily temperature.
    thresh : Quantified
        Threshold temperature on which to base evaluation.
    freq : str
        Resampling frequency.

    Returns
    -------
    xarray.DataArray, [time][temperature]
        Heating degree days index.

    Notes
    -----
    This index intentionally differs from its ECA&D :cite:p:`project_team_eca&d_algorithm_2013` equivalent: HD17.
    In HD17, values below zero are not clipped before the sum. The present definition should provide a better
    representation of the energy demand for heating buildings to the given threshold.

    Let :math:`TG_{ij}` be the daily mean temperature at day :math:`i` of period :math:`j`. Then the
    heating degree days are:

    .. math::

        HD17_j = \sum_{i=1}^{I} (17℃ - TG_{ij}) | TG_{ij} < 17℃)
    """
    return cumulative_difference(tas, threshold=thresh, op="<", freq=freq)


@declare_units(tasmax="[temperature]", thresh="[temperature]")
def hot_spell_max_length(
    tasmax: xarray.DataArray,
    thresh: Quantified = "30 degC",
    thresh_tasmax: str = "UNSET",
    window: int = 1,
    freq: str = "YS",
    op: str = ">",
    resample_before_rl: bool = True,
) -> xarray.DataArray:
    r"""Longest hot spell.

    Longest spell of high temperatures over a given period.
    Longest series of at least {window} consecutive days with temperature at or above {thresh}.

    Parameters
    ----------
    tasmax : xarray.DataArray
        Maximum daily temperature.
    thresh : Quantified
        The temperature threshold needed to trigger a hot spell.
    window : int
        Minimum number of days with temperatures below thresholds to qualify as a hot spell.
    freq : str
        Resampling frequency.
    op : {">", ">=", "gt", "ge"}
        Comparison operation. Default: ">".
    resample_before_rl : bool
        Determines if the resampling should take place before or after the run
        length encoding (or a similar algorithm) is applied to runs.

    Returns
    -------
    xarray.DataArray, [days]
        The {freq} longest spell in hot periods of minimum {window} days.

    Notes
    -----
    The threshold on `tasmax` follows the one used in heat waves. A day temperature threshold between 30° and 35°C
    was selected by Health Canada professionals, following a temperature–mortality analysis. This absolute temperature
    threshold characterize the occurrence of hot weather events that can result in adverse health outcomes for Canadian
    communities :cite:p:`casati_regional_2013`.

    In :cite:t:`robinson_definition_2001` where heat waves are also considered, the corresponding parameters would
    be `thresh_tasmax=39.44, window=2` (103F).

    References
    ----------
    :cite:cts:`casati_regional_2013,robinson_definition_2001`
    """
    if thresh_tasmax != "UNSET":
        warnings.warn(
            "The call signature for this indicator will change from `thresh_tasmax` to `thresh` in xclim>=0.45.0.  Passing `thresh_tasmax` value to `thresh`."
        )
        thresh = thresh_tasmax
    thresh = convert_units_to(thresh, tasmax)

    cond = compare(tasmax, op, thresh, constrain=(">", ">="))
    max_l = rl.resample_and_rl(
        cond,
        resample_before_rl,
        rl.longest_run,
        freq=freq,
    )
    out = max_l.where(max_l >= window, 0)
    return to_agg_units(out, tasmax, "count")


@declare_units(tasmax="[temperature]", thresh="[temperature]")
def hot_spell_total_length(
    tasmax: xarray.DataArray,
    thresh: Quantified = "30 degC",
    thresh_tasmax: str = "UNSET",
    window: int = 3,
    freq: str = "YS",
    op: str = ">",
    resample_before_rl: bool = True,
) -> xarray.DataArray:
    r"""Total length of hot spells.

    Total length of spells of high temperatures over a given period.
    Total length of series of at least {window} consecutive days with temperature at or above {thresh}.

    Parameters
    ----------
    tasmax : xarray.DataArray
        Maximum daily temperature.
    thresh : Quantified
        The temperature threshold needed to trigger a hot spell.
    window : int
        Minimum number of days with temperatures below thresholds to qualify as a hot spell.
    freq : str
        Resampling frequency.
    op : {">", ">=", "gt", "ge"}
        Comparison operation. Default: ">".
    resample_before_rl : bool
        Determines if the resampling should take place before or after the run
        length encoding (or a similar algorithm) is applied to runs.

    Returns
    -------
    xarray.DataArray, [days]
        The {freq} total number of days in hot periods of minimum {window} days.

    Notes
    -----
    The threshold on `tasmax` follows the one used in heat waves. A day temperature threshold between 30° and 35°C
    was selected by Health Canada professionals, following a temperature–mortality analysis. This absolute temperature
    threshold characterize the occurrence of hot weather events that can result in adverse health outcomes for Canadian
    communities :cite:p:`casati_regional_2013`.

    In :cite:t:`robinson_definition_2001` where heat waves are also considered, the corresponding parameters would
    be `thresh_tasmax=39.44, window=2` (103F).
    """
    if thresh_tasmax != "UNSET":
        warnings.warn(
            "The call signature for this indicator will change from `thresh_tasmax` to `thresh` in xclim>=0.45.0.  Passing `thresh_tasmax` value to `thresh`."
        )
        thresh = thresh_tasmax

    thresh = convert_units_to(thresh, tasmax)

    cond = compare(tasmax, op, thresh, constrain=(">", ">="))
    max_l = rl.resample_and_rl(
        cond,
        resample_before_rl,
        rl.windowed_run_count,
        window=1,
        freq=freq,
    )
    out = max_l.where(max_l >= window, 0)
    return to_agg_units(out, tasmax, "count")


@declare_units(tasmax="[temperature]", thresh="[temperature]")
def hot_spell_frequency(
    tasmax: xarray.DataArray,
    thresh: Quantified = "30 degC",
    thresh_tasmax: str = "UNSET",
    window: int = 3,
    freq: str = "YS",
    op: str = ">",
    resample_before_rl: bool = True,
) -> xarray.DataArray:
    """Hot spell frequency.

    The number of hot spell events, defined as a sequence of consecutive {window} days
    with mean daily temperature above a {thresh}.

    Parameters
    ----------
    tasmax : xarray.DataArray
        Maximum daily temperature.
    thresh : Quantified
        Threshold temperature below which a hot spell begins.
    window : int
        Minimum number of days with temperature above threshold to qualify as a hot spell.
    freq : str
        Resampling frequency.
    op : {">", ">=", "gt", "ge"}
        Comparison operation. Default: ">".
    resample_before_rl : bool
        Determines if the resampling should take place before or after the run

    Returns
    -------
    xarray.DataArray, [unitless]
        The {freq} number of hot periods of minimum {window} days.

    Notes
    -----
    The threshold on `tasmax` follows the one used in heat waves. A day temperature threshold between 30° and 35°C
    was selected by Health Canada professionals, following a temperature–mortality analysis. This absolute temperature
    threshold characterize the occurrence of hot weather events that can result in adverse health outcomes for Canadian
    communities :cite:p:`casati_regional_2013`.

    In :cite:t:`robinson_definition_2001` where heat waves are also considered, the corresponding parameters would
    be `thresh_tasmax=39.44, window=2` (103F).

    References
    ----------
    :cite:cts:`casati_regional_2013,robinson_definition_2001`
    """
    if thresh_tasmax != "UNSET":
        warnings.warn(
            "The call signature for this indicator will change from `thresh_tasmax` to `thresh` in xclim>=0.45.0.  Passing `thresh_tasmax` value to `thresh`."
        )
        thresh = thresh_tasmax
    thresh = convert_units_to(thresh, tasmax)

    cond = compare(tasmax, op, thresh, constrain=(">", ">="))
    out = rl.resample_and_rl(
        cond,
        resample_before_rl,
        rl.windowed_run_events,
        window=window,
        freq=freq,
    )
    out.attrs["units"] = ""
    return out


@declare_units(snd="[length]", thresh="[length]")
def snd_season_length(
    snd: xarray.DataArray,
    thresh: Quantified = "2 cm",
    freq: str = "AS-JUL",
    op: str = ">=",
) -> xarray.DataArray:
    """The number of days with snow depth above a threshold.

    Number of days where surface snow depth is greater or equal to given threshold (default: 2 cm).

    Warnings
    --------
    The default `freq` is valid for the northern hemisphere.

    Parameters
    ----------
    snd : xarray.DataArray
        Surface snow thickness.
    thresh : Quantified
        Threshold snow thickness.
    freq : str
        Resampling frequency.
    op : {">", ">=", "gt", "ge"}
        Comparison operation. Default: ">=".

    Returns
    -------
    xarray.DataArray, [time]
        Number of days where snow depth is greater than or equal to threshold.
    """
    valid = at_least_n_valid(snd.where(snd > 0), n=1, freq=freq)
    thresh = convert_units_to(thresh, snd)
    out = threshold_count(snd, op, thresh, freq)
    return to_agg_units(out, snd, "count").where(~valid)


@declare_units(snw="[mass]/[area]", thresh="[mass]/[area]")
def snw_season_length(
    snw: xarray.DataArray,
    thresh: Quantified = "20 kg m-2",
    freq: str = "AS-JUL",
    op: str = ">=",
) -> xarray.DataArray:
    """The number of days with snow water above a threshold.

    Number of days where surface snow water is greater or equal to given threshold (default: 2 cm).

    Warnings
    --------
    The default `freq` is valid for the northern hemisphere.

    Parameters
    ----------
    snw : xarray.DataArray
        Surface snow amount.
    thresh : str
        Threshold snow amount.
    freq : str
        Resampling frequency.
    op : {">", ">=", "gt", "ge"}
        Comparison operation. Default: ">=".

    Returns
    -------
    xarray.DataArray, [time]
        Number of days where snow water is greater than or equal to threshold.
    """
    valid = at_least_n_valid(snw.where(snw > 0), n=1, freq=freq)
    thresh = convert_units_to(thresh, snw)
    out = threshold_count(snw, op, thresh, freq)
    return to_agg_units(out, snw, "count").where(~valid)


@declare_units(tasmin="[temperature]", thresh="[temperature]")
def tn_days_above(
    tasmin: xarray.DataArray,
    thresh: Quantified = "20.0 degC",
    freq: str = "YS",
    op: str = ">",
):
    """The number of days with tasmin above a threshold (number of tropical nights).

    Number of days where minimum daily temperature exceeds a threshold (default: 20℃).

    Parameters
    ----------
    tasmin : xarray.DataArray
        Minimum daily temperature.
    thresh : Quantified
        Threshold temperature on which to base evaluation.
    freq : str
        Resampling frequency.
    op : {">", ">=", "gt", "ge"}
        Comparison operation. Default: ">".

    Returns
    -------
    xarray.DataArray, [time]
        Number of days where tasmin {op} threshold.

    Notes
    -----
    Let :math:`TN_{ij}` be the minimum daily temperature at day :math:`i` of period :math:`j`. Then
    counted is the number of days where:

    .. math::

        TN_{ij} > Threshold [℃]
    """
    thresh = convert_units_to(thresh, tasmin)
    f = threshold_count(tasmin, op, thresh, freq, constrain=(">", ">="))
    return to_agg_units(f, tasmin, "count")


@declare_units(tasmin="[temperature]", thresh="[temperature]")
def tn_days_below(
    tasmin: xarray.DataArray,
    thresh: Quantified = "-10.0 degC",
    freq: str = "YS",
    op: str = "<",
) -> xarray.DataArray:
    """Number of days with tasmin below a threshold.

    Number of days where minimum daily temperature is below a threshold (default: -10℃).

    Parameters
    ----------
    tasmin : xarray.DataArray
        Minimum daily temperature.
    thresh : Quantified
        Threshold temperature on which to base evaluation.
    freq : str
        Resampling frequency.
    op : {"<", "<=", "lt", "le"}
        Comparison operation. Default: "<".

    Returns
    -------
    xarray.DataArray, [time]
        Number of days where tasmin {op} threshold.

    Notes
    -----
    Let :math:`TN_{ij}` be the minimum daily temperature at day :math:`i` of period :math:`j`. Then
    counted is the number of days where:

    .. math::

        TN_{ij} < Threshold [℃]
    """
    thresh = convert_units_to(thresh, tasmin)
    f1 = threshold_count(tasmin, op, thresh, freq, constrain=("<", "<="))
    return to_agg_units(f1, tasmin, "count")


@declare_units(tas="[temperature]", thresh="[temperature]")
def tg_days_above(
    tas: xarray.DataArray,
    thresh: Quantified = "10.0 degC",
    freq: str = "YS",
    op: str = ">",
):
    """The number of days with tas above a threshold.

    Number of days where mean daily temperature exceeds a threshold (default: 10℃).

    Parameters
    ----------
    tas : xarray.DataArray
        Mean daily temperature.
    thresh : Quantified
        Threshold temperature on which to base evaluation.
    freq : str
        Resampling frequency.
    op : {">", ">=", "gt", "ge"}
        Comparison operation. Default: ">".

    Returns
    -------
    xarray.DataArray, [time]
        Number of days where tas {op} threshold.

    Notes
    -----
    Let :math:`TG_{ij}` be the mean daily temperature at day :math:`i` of period :math:`j`. Then
    counted is the number of days where:

    .. math::

        TG_{ij} > Threshold [℃]
    """
    thresh = convert_units_to(thresh, tas)
    f = threshold_count(tas, op, thresh, freq, constrain=(">", ">="))
    return to_agg_units(f, tas, "count")


@declare_units(tas="[temperature]", thresh="[temperature]")
def tg_days_below(
    tas: xarray.DataArray,
    thresh: Quantified = "10.0 degC",
    freq: str = "YS",
    op: str = "<",
):
    """The number of days with tas below a threshold.

    Number of days where mean daily temperature is below a threshold (default: 10℃).

    Parameters
    ----------
    tas : xarray.DataArray
        Mean daily temperature.
    thresh : Quantified
        Threshold temperature on which to base evaluation.
    freq : str
        Resampling frequency.
    op : {"<", "<=", "lt", "le"}
        Comparison operation. Default: "<".

    Returns
    -------
    xarray.DataArray, [time]
        Number of days where tas {op} threshold.

    Notes
    -----
    Let :math:`TG_{ij}` be the mean daily temperature at day :math:`i` of period :math:`j`. Then counted is the number
    of days where:

    .. math::

        TG_{ij} < Threshold [℃]
    """
    thresh = convert_units_to(thresh, tas)
    f1 = threshold_count(tas, op, thresh, freq, constrain=("<", "<="))
    return to_agg_units(f1, tas, "count")


@declare_units(tasmax="[temperature]", thresh="[temperature]")
def tx_days_above(
    tasmax: xarray.DataArray,
    thresh: Quantified = "25.0 degC",
    freq: str = "YS",
    op: str = ">",
) -> xarray.DataArray:
    """The number of days with tasmax above a threshold (number of summer days).

    Number of days where maximum daily temperature exceeds a threshold (default: 25℃).

    Parameters
    ----------
    tasmax : xarray.DataArray
        Maximum daily temperature.
    thresh : Quantified
        Threshold temperature on which to base evaluation.
    freq : str
        Resampling frequency.
    op : {">", ">=", "gt", "ge"}
        Comparison operation. Default: ">".

    Returns
    -------
    xarray.DataArray, [time]
        Number of days where tasmax {op} threshold (number of summer days).

    Notes
    -----
    Let :math:`TX_{ij}` be the maximum daily temperature at day :math:`i` of period :math:`j`. Then counted is the
    number of days where:

    .. math::

        TX_{ij} > Threshold [℃]
    """
    thresh = convert_units_to(thresh, tasmax)
    f = threshold_count(tasmax, op, thresh, freq, constrain=(">", ">="))
    return to_agg_units(f, tasmax, "count")


@declare_units(tasmax="[temperature]", thresh="[temperature]")
def tx_days_below(
    tasmax: xarray.DataArray,
    thresh: Quantified = "25.0 degC",
    freq: str = "YS",
    op: str = "<",
):
    """The number of days with tmax below a threshold.

    Number of days where maximum daily temperature is below a threshold (default: 25℃).

    Parameters
    ----------
    tasmax : xarray.DataArray
        Maximum daily temperature.
    thresh : Quantified
        Threshold temperature on which to base evaluation.
    freq : str
        Resampling frequency.
    op : {"<", "<=", "lt", "le"}
        Comparison operation. Default: "<".

    Returns
    -------
    xarray.DataArray, [time]
        Number of days where tasmin {op} threshold.

    Notes
    -----
    Let :math:`TX_{ij}` be the maximum daily temperature at day :math:`i` of period :math:`j`. Then
    counted is the number of days where:

    .. math::

        TX_{ij} < Threshold [℃]
    """
    thresh = convert_units_to(thresh, tasmax)
    f1 = threshold_count(tasmax, op, thresh, freq, constrain=("<", "<="))
    return to_agg_units(f1, tasmax, "count")


@declare_units(tasmax="[temperature]", thresh="[temperature]")
def warm_day_frequency(
    tasmax: xarray.DataArray,
    thresh: Quantified = "30 degC",
    freq: str = "YS",
    op: str = ">",
) -> xarray.DataArray:
    """Frequency of extreme warm days.

    Return the number of days with maximum daily temperature exceeding threshold (default: 30℃) per period.

    Parameters
    ----------
    tasmax : xarray.DataArray
        Maximum daily temperature.
    thresh : Quantified
        Threshold temperature on which to base evaluation.
    freq : str
        Resampling frequency.
    op : {">", ">=", "gt", "ge"}
        Comparison operation. Default: ">".

    Returns
    -------
    xarray.DataArray, [time]
        Number of days with tasmax {op} threshold per period.

    Notes
    -----
    Let :math:`TX_{ij}` be the maximum daily temperature at day :math:`i` of period :math:`j`. Then counted is the
    number of days where:

    .. math::

        TN_{ij} > Threshold [℃]
    """
    thresh = convert_units_to(thresh, tasmax)
    events = threshold_count(tasmax, op, thresh, freq, constrain=(">", ">="))
    return to_agg_units(events, tasmax, "count")


@declare_units(tasmin="[temperature]", thresh="[temperature]")
def warm_night_frequency(
    tasmin: xarray.DataArray,
    thresh: Quantified = "22 degC",
    freq: str = "YS",
    op: str = ">",
) -> xarray.DataArray:
    """Frequency of extreme warm nights.

    Return the number of days with minimum daily temperature exceeding threshold (default: 22℃) per period.

    Parameters
    ----------
    tasmin : xarray.DataArray
        Minimum daily temperature.
    thresh : Quantified
        Threshold temperature on which to base evaluation.
    freq : str
        Resampling frequency.
    op : {">", ">=", "gt", "ge"}
        Comparison operation. Default: ">".

    Returns
    -------
    xarray.DataArray, [time]
        Number of days with tasmin {op} threshold per period.
    """
    thresh = convert_units_to(thresh, tasmin)
    events = threshold_count(tasmin, op, thresh, freq, constrain=(">", ">="))
    return to_agg_units(events, tasmin, "count")


@declare_units(pr="[precipitation]", thresh="[precipitation]")
def wetdays(
    pr: xarray.DataArray,
    thresh: Quantified = "1.0 mm/day",
    freq: str = "YS",
    op: str = ">=",
) -> xarray.DataArray:
    """Wet days.

    Return the total number of days during period with precipitation over threshold (default: 1.0 mm/day).

    Parameters
    ----------
    pr : xarray.DataArray
        Daily precipitation.
    thresh : Quantified
        Precipitation value over which a day is considered wet.
    freq : str
        Resampling frequency.
    op : {">", ">=", "gt", "ge"}
        Comparison operation. Default: ">=".

    Returns
    -------
    xarray.DataArray, [time]
        The number of wet days for each period [day].

    Examples
    --------
    The following would compute for each grid cell of file `pr.day.nc` the number days with precipitation over 5 mm
    at the seasonal frequency, i.e. DJF, MAM, JJA, SON, DJF, etc.:

    >>> from xclim.indices import wetdays
    >>> pr = xr.open_dataset(path_to_pr_file).pr
    >>> wd = wetdays(pr, thresh="5 mm/day", freq="QS-DEC")
    """
    thresh = convert_units_to(thresh, pr, "hydro")

    wd = threshold_count(pr, op, thresh, freq, constrain=(">", ">="))
    return to_agg_units(wd, pr, "count")


@declare_units(pr="[precipitation]", thresh="[precipitation]")
def wetdays_prop(
    pr: xarray.DataArray,
    thresh: Quantified = "1.0 mm/day",
    freq: str = "YS",
    op: str = ">=",
) -> xarray.DataArray:
    """Proportion of wet days.

    Return the proportion of days during period with precipitation over threshold (default: 1.0 mm/day).

    Parameters
    ----------
    pr : xarray.DataArray
        Daily precipitation.
    thresh : Quantified
        Precipitation value over which a day is considered wet.
    freq : str
        Resampling frequency.
    op : {">", ">=", "gt", "ge"}
        Comparison operation. Default: ">=".

    Returns
    -------
    xarray.DataArray, [time]
        The proportion of wet days for each period [1].

    Examples
    --------
    The following would compute for each grid cell of file `pr.day.nc` the proportion of days with precipitation over
    5 mm at the seasonal frequency, i.e. DJF, MAM, JJA, SON, DJF, etc.:

    >>> from xclim.indices import wetdays_prop
    >>> pr = xr.open_dataset(path_to_pr_file).pr
    >>> wd = wetdays_prop(pr, thresh="5 mm/day", freq="QS-DEC")
    """
    thresh = convert_units_to(thresh, pr, "hydro")

    wd = compare(pr, op, thresh, constrain=(">", ">="))
    fwd = wd.resample(time=freq).mean(dim="time").assign_attrs(units="1")
    return fwd


# NOTE : A spell index could be used below
@declare_units(tasmin="[temperature]", thresh="[temperature]")
def maximum_consecutive_frost_days(
    tasmin: xarray.DataArray,
    thresh: Quantified = "0.0 degC",
    freq: str = "AS-JUL",
    resample_before_rl: bool = True,
) -> xarray.DataArray:
    r"""Maximum number of consecutive frost days (Tn < 0℃).

    The maximum number of consecutive days within the period where the minimum daily temperature
    is under a given threshold (default: 0°C).

    Warnings
    --------
    The default `freq` is valid for the northern hemisphere.

    Parameters
    ----------
    tasmin : xarray.DataArray
        Minimum daily temperature.
    thresh : Quantified
        Threshold temperature.
    freq : str
        Resampling frequency.
    resample_before_rl : bool
      Determines if the resampling should take place before or after the run
      length encoding (or a similar algorithm) is applied to runs.

    Returns
    -------
    xarray.DataArray, [time]
        The maximum number of consecutive frost days (tasmin < threshold per period).

    Notes
    -----
    Let :math:`\mathbf{t}=t_0, t_1, \ldots, t_n` be a minimum daily temperature series and :math:`thresh` the threshold
    below which a day is considered a frost day. Let :math:`\mathbf{s}` be the sorted vector of indices :math:`i`
    where :math:`[t_i < thresh] \neq [t_{i+1} < thresh]`, that is, the days where the temperature crosses the threshold.
    Then the maximum number of consecutive frost days is given by

    .. math::

       \max(\mathbf{d}) \quad \mathrm{where} \quad d_j = (s_j - s_{j-1}) [t_{s_j} < thresh]

    where :math:`[P]` is 1 if :math:`P` is true, and 0 if false. Note that this formula does not handle sequences at
    the start and end of the series, but the numerical algorithm does.
    """
    t = convert_units_to(thresh, tasmin)
    group = tasmin < t
    out = rl.resample_and_rl(
        group,
        resample_before_rl,
        rl.longest_run,
        freq=freq,
    )
    return to_agg_units(out, tasmin, "count")


# NOTE : A spell index could be used below
@declare_units(pr="[precipitation]", thresh="[precipitation]")
def maximum_consecutive_dry_days(
    pr: xarray.DataArray,
    thresh: Quantified = "1 mm/day",
    freq: str = "YS",
    resample_before_rl: bool = True,
) -> xarray.DataArray:
    r"""Maximum number of consecutive dry days.

    Return the maximum number of consecutive days within the period where precipitation
    is below a certain threshold (default: 1 mm/day).

    Parameters
    ----------
    pr : xarray.DataArray
        Mean daily precipitation flux.
    thresh : Quantified
        Threshold precipitation on which to base evaluation.
    freq : str
        Resampling frequency.
    resample_before_rl : bool
      Determines if the resampling should take place before or after the run
      length encoding (or a similar algorithm) is applied to runs.

    Returns
    -------
    xarray.DataArray, [time]
        The maximum number of consecutive dry days (precipitation < threshold per period).

    Notes
    -----
    Let :math:`\mathbf{p}=p_0, p_1, \ldots, p_n` be a daily precipitation series and :math:`thresh` the threshold
    under which a day is considered dry. Then let :math:`\mathbf{s}` be the sorted vector of indices :math:`i` where
    :math:`[p_i < thresh] \neq [p_{i+1} < thresh]`, that is, the days where the precipitation crosses the threshold.
    Then the maximum number of consecutive dry days is given by

    .. math::

       \max(\mathbf{d}) \quad \mathrm{where} \quad d_j = (s_j - s_{j-1}) [p_{s_j} < thresh]

    where :math:`[P]` is 1 if :math:`P` is true, and 0 if false. Note that this formula does not handle sequences at
    the start and end of the series, but the numerical algorithm does.
    """
    t = convert_units_to(thresh, pr, context="hydro")
    group = pr < t
    out = rl.resample_and_rl(
        group,
        resample_before_rl,
        rl.longest_run,
        freq=freq,
    )
    return to_agg_units(out, pr, "count")


@declare_units(tasmin="[temperature]", thresh="[temperature]")
def maximum_consecutive_frost_free_days(
    tasmin: xarray.DataArray,
    thresh: Quantified = "0 degC",
    freq: str = "YS",
    resample_before_rl: bool = True,
) -> xarray.DataArray:
    r"""Maximum number of consecutive frost free days (Tn >= 0℃).

    Return the maximum number of consecutive days within the period where the minimum daily temperature is
    above or equal to a certain threshold (default: 0℃).

    Warnings
    --------
    The default `freq` is valid for the northern hemisphere.

    Parameters
    ----------
    tasmin : xarray.DataArray
        Minimum daily temperature.
    thresh : Quantified
        Threshold temperature.
    freq : str
        Resampling frequency.
    resample_before_rl : bool
      Determines if the resampling should take place before or after the run
      length encoding (or a similar algorithm) is applied to runs.

    Returns
    -------
    xarray.DataArray, [time]
        The maximum number of consecutive frost free days (tasmin >= threshold per period).

    Notes
    -----
    Let :math:`\mathbf{t}=t_0, t_1, \ldots, t_n` be a daily minimum temperature series and :math:`thresh` the threshold
    above or equal to which a day is considered a frost free day. Let :math:`\mathbf{s}` be the sorted vector of
    indices :math:`i` where :math:`[t_i <= thresh] \neq [t_{i+1} <= thresh]`, that is, the days where the temperature
    crosses the threshold. Then the maximum number of consecutive frost free days is given by:

    .. math::

       \max(\mathbf{d}) \quad \mathrm{where} \quad d_j = (s_j - s_{j-1}) [t_{s_j} >= thresh]

    where :math:`[P]` is 1 if :math:`P` is true, and 0 if false. Note that this formula does not handle sequences at
    the start and end of the series, but the numerical algorithm does.
    """
    t = convert_units_to(thresh, tasmin)
    group = tasmin >= t
    out = rl.resample_and_rl(
        group,
        resample_before_rl,
        rl.longest_run,
        freq=freq,
    )
    return to_agg_units(out, tasmin, "count")


# NOTE : A spell index could be used below
@declare_units(tasmax="[temperature]", thresh="[temperature]")
def maximum_consecutive_tx_days(
    tasmax: xarray.DataArray,
    thresh: Quantified = "25 degC",
    freq: str = "YS",
    resample_before_rl: bool = True,
) -> xarray.DataArray:
    r"""Maximum number of consecutive days with tasmax above a threshold (summer days).

    Return the maximum number of consecutive days within the period where the maximum daily temperature is
    above a certain threshold (default: 25℃).

    Parameters
    ----------
    tasmax : xarray.DataArray
        Max daily temperature.
    thresh : Quantified
        Threshold temperature.
    freq : str
        Resampling frequency.
    resample_before_rl : bool
      Determines if the resampling should take place before or after the run
      length encoding (or a similar algorithm) is applied to runs.

    Returns
    -------
    xarray.DataArray, [time]
        The maximum number of days with tasmax > thresh per periods (summer days).

    Notes
    -----
    Let :math:`\mathbf{t}=t_0, t_1, \ldots, t_n` be a daily maximum temperature series and :math:`thresh` the threshold
    above which a day is considered a summer day. Let :math:`\mathbf{s}` be the sorted vector of indices :math:`i`
    where :math:`[t_i < thresh] \neq [t_{i+1} < thresh]`, that is, the days where the temperature crosses the threshold.
    Then the maximum number of consecutive tx_days (summer days) is given by:

    .. math::

       \max(\mathbf{d}) \quad \mathrm{where} \quad d_j = (s_j - s_{j-1}) [t_{s_j} > thresh]

    where :math:`[P]` is 1 if :math:`P` is true, and 0 if false. Note that this formula does not handle sequences at
    the start and end of the series, but the numerical algorithm does.
    """
    t = convert_units_to(thresh, tasmax)
    group = tasmax > t
    out = rl.resample_and_rl(
        group,
        resample_before_rl,
        rl.longest_run,
        freq=freq,
    )
    return to_agg_units(out, tasmax, "count")


@declare_units(siconc="[]", areacello="[area]", thresh="[]")
def sea_ice_area(
    siconc: xarray.DataArray, areacello: xarray.DataArray, thresh: Quantified = "15 pct"
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
    thresh : Quantified
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
    "What is the difference between sea ice area and extent?" - :cite:cts:`nsidc_frequently_2008`
    """
    t = convert_units_to(thresh, siconc)
    factor = convert_units_to("100 pct", siconc)
    out = xarray.dot(siconc.where(siconc >= t, 0), areacello) / factor
    out.attrs["units"] = areacello.units
    return out


@declare_units(siconc="[]", areacello="[area]", thresh="[]")
def sea_ice_extent(
    siconc: xarray.DataArray, areacello: xarray.DataArray, thresh: Quantified = "15 pct"
) -> xarray.DataArray:
    """Total sea ice extent.

    Sea ice extent measures the *ice-covered* area, where a region is considered ice-covered if its sea ice
    concentration is above a threshold, usually set to 15%.

    Parameters
    ----------
    siconc : xarray.DataArray
        Sea ice concentration (area fraction).
    areacello : xarray.DataArray
        Grid cell area.
    thresh : Quantified
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
    "What is the difference between sea ice area and extent?" - :cite:cts:`nsidc_frequently_2008`
    """
    t = convert_units_to(thresh, siconc)
    out = xarray.dot(siconc >= t, areacello)
    out.attrs["units"] = areacello.units
    return out


@declare_units(sfcWind="[speed]", thresh="[speed]")
def windy_days(
    sfcWind: xarray.DataArray, thresh: Quantified = "10.8 m s-1", freq: str = "MS"
) -> xarray.DataArray:
    r"""Windy days.

    The number of days with average near-surface wind speed above threshold (default: 10.8 m/s).

    Parameters
    ----------
    sfcWind : xarray.DataArray
        Daily average near-surface wind speed.
    thresh : Quantified
        Threshold average near-surface wind speed on which to base evaluation.
    freq : str
        Resampling frequency.

    Returns
    -------
    xarray.DataArray, [time]
        Number of days with average near-surface wind speed above threshold.

    Notes
    -----
    Let :math:`WS_{ij}` be the windspeed at day :math:`i` of period :math:`j`. Then counted is the number of days where:

    .. math::

        WS_{ij} >= Threshold [m s-1]
    """
    thresh = convert_units_to(thresh, sfcWind)
    out = threshold_count(sfcWind, ">=", thresh, freq)
    out = to_agg_units(out, sfcWind, "count")
    return out


@declare_units(pr="[precipitation]", prc="[precipitation]", thresh="[precipitation]")
def rprctot(
    pr: xarray.DataArray,
    prc: xarray.DataArray,
    thresh: Quantified = "1.0 mm/day",
    freq: str = "YS",
    op: str = ">=",
) -> xarray.DataArray:
    """Proportion of accumulated precipitation arising from convective processes.

    Return the proportion of total accumulated precipitation due to convection on days with total precipitation
    greater or equal to a given threshold (default: 1.0 mm/day) during the given period.

    Parameters
    ----------
    pr : xarray.DataArray
        Daily precipitation.
    prc : xarray.DataArray
        Daily convective precipitation.
    thresh : Quantified
        Precipitation value over which a day is considered wet.
    freq : str
        Resampling frequency.
    op : {">", ">=", "gt", "ge"}
        Comparison operation. Default: ">=".

    Returns
    -------
    xarray.DataArray, [dimensionless]
        The proportion of the total precipitation accounted for by convective precipitation for each period.
    """
    thresh = convert_units_to(thresh, pr, "hydro")
    prc = convert_units_to(prc, pr)

    wd = compare(pr, op, thresh)
    pr_tot = rate2amount(pr).where(wd).resample(time=freq).sum(dim="time")
    prc_tot = rate2amount(prc).where(wd).resample(time=freq).sum(dim="time")

    ratio = prc_tot / pr_tot
    ratio = ratio.assign_attrs(units="")

    return ratio


@declare_units(tas="[temperature]", thresh="[temperature]", sum_thresh="K days")
def degree_days_exceedance_date(
    tas: xarray.DataArray,
    thresh: Quantified = "0 degC",
    sum_thresh: Quantified = "25 K days",
    op: str = ">",
    after_date: DayOfYearStr = None,
    freq: str = "YS",
) -> xarray.DataArray:
    r"""Degree-days exceedance date.

    Day of year when the sum of degree days exceeds a threshold (default: 25 K days).
    Degree days are computed above or below a given temperature threshold (default: 0℃).

    Parameters
    ----------
    tas : xarray.DataArray
        Mean daily temperature.
    thresh : Quantified
        Threshold temperature on which to base degree-days evaluation.
    sum_thresh : Quantified
        Threshold of the degree days sum.
    op : {">", "gt", "<", "lt", ">=", "ge", "<=", "le"}
        If equivalent to '>', degree days are computed as `tas - thresh` and if
        equivalent to '<', they are computed as `thresh - tas`.
    after_date: str, optional
        Date at which to start the cumulative sum.
        In "mm-dd" format, defaults to the start of the sampling period.
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
    See https://en.wikipedia.org/wiki/Growing_degree-day for examples (:cite:t:`wikipedia_contributors_growing_2021`).
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
    out.attrs.update(units="", is_dayofyear=np.int32(1), calendar=get_calendar(tas))
    return out


@declare_units(snd="[length]", thresh="[length]")
def winter_storm(
    snd: xarray.DataArray, thresh: Quantified = "25 cm", freq: str = "AS-JUL"
) -> xarray.DataArray:
    """Days with snowfall over threshold.

    Number of days with snowfall accumulation greater or equal to threshold (default: 25 cm).

    Warnings
    --------
    The default `freq` is valid for the northern hemisphere.

    Parameters
    ----------
    snd : xarray.DataArray
        Surface snow depth.
    thresh : Quantified
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


@declare_units(pr="[precipitation]", thresh="[length]")
def dry_spell_frequency(
    pr: xarray.DataArray,
    thresh: Quantified = "1.0 mm",
    window: int = 3,
    freq: str = "YS",
    resample_before_rl: bool = True,
    op: str = "sum",
) -> xarray.DataArray:
    """Return the number of dry periods of n days and more.

    Periods during which the accumulated or maximal daily precipitation amount on a window of n days is under threshold.

    Parameters
    ----------
    pr : xarray.DataArray
        Daily precipitation.
    thresh : Quantified
        Precipitation amount under which a period is considered dry.
        The value against which the threshold is compared depends on  `op` .
    window : int
        Minimum length of the spells.
    freq : str
      Resampling frequency.
    resample_before_rl : bool
      Determines if the resampling should take place before or after the run
      length encoding (or a similar algorithm) is applied to runs.
    op: {"sum","max"}
      Operation to perform on the window.
      Default is "sum", which checks that the sum of accumulated precipitation over the whole window is less than the
      threshold.
      "max" checks that the maximal daily precipitation amount within the window is less than the threshold.
      This is the same as verifying that each individual day is below the threshold.

    Returns
    -------
    xarray.DataArray, [unitless]
        The {freq} number of dry periods of minimum {window} days.

    Examples
    --------
    >>> from xclim.indices import dry_spell_frequency
    >>> pr = xr.open_dataset(path_to_pr_file).pr
    >>> dsf = dry_spell_frequency(pr=pr, op="sum")
    >>> dsf = dry_spell_frequency(pr=pr, op="max")
    """
    pram = rate2amount(convert_units_to(pr, "mm/d", context="hydro"), out_units="mm")
    thresh = convert_units_to(thresh, pram, context="hydro")

    agg_pr = getattr(pram.rolling(time=window, center=True), op)()
    cond = agg_pr < thresh
    out = rl.resample_and_rl(
        cond,
        resample_before_rl,
        rl.windowed_run_events,
        window=1,
        freq=freq,
    )
    out.attrs["units"] = ""
    return out


@declare_units(pr="[precipitation]", thresh="[length]")
def dry_spell_total_length(
    pr: xarray.DataArray,
    thresh: Quantified = "1.0 mm",
    window: int = 3,
    op: str = "sum",
    freq: str = "YS",
    resample_before_rl: bool = True,
    **indexer,
) -> xarray.DataArray:
    """Total length of dry spells.

    Total number of days in dry periods of a minimum length, during which the maximum or
    accumulated precipitation within a window of the same length is under a threshold.

    Parameters
    ----------
    pr : xarray.DataArray
        Daily precipitation.
    thresh : Quantified
        Accumulated precipitation value under which a period is considered dry.
    window : int
        Number of days when the maximum or accumulated precipitation is under threshold.
    op : {"max", "sum"}
        Reduce operation.
    freq : str
        Resampling frequency.
    indexer
        Indexing parameters to compute the indicator on a temporal subset of the data.
        It accepts the same arguments as :py:func:`xclim.indices.generic.select_time`.
        Indexing is done after finding the dry days, but before finding the spells.

    Returns
    -------
    xarray.DataArray, [days]
        The {freq} total number of days in dry periods of minimum {window} days.

    Notes
    -----
    The algorithm assumes days before and after the timeseries are "wet", meaning that the condition for being
    considered part of a dry spell is stricter on the edges. For example, with `window=3` and `op='sum'`, the first day
    of the series is considered part of a dry spell only if the accumulated precipitation within the first three days is
    under the threshold. In comparison, a day in the middle of the series is considered part of a dry spell if any of
    the three 3-day periods of which it is part are considered dry (so a total of five days are included in the
    computation, compared to only three).
    """
    pram = rate2amount(convert_units_to(pr, "mm/d", context="hydro"), out_units="mm")
    thresh = convert_units_to(thresh, pram, context="hydro")

    pram_pad = pram.pad(time=(0, window))
    mask = getattr(pram_pad.rolling(time=window), op)() < thresh
    dry = (mask.rolling(time=window).sum() >= 1).shift(time=-(window - 1))
    dry = dry.isel(time=slice(0, pram.time.size)).astype(float)

    dry = select_time(dry, **indexer)

    out = rl.resample_and_rl(
        dry,
        resample_before_rl,
        rl.windowed_run_count,
        window=1,
        freq=freq,
    )
    return to_agg_units(out, pram, "count")


@declare_units(pr="[precipitation]", thresh="[length]")
def dry_spell_max_length(
    pr: xarray.DataArray,
    thresh: Quantified = "1.0 mm",
    window: int = 1,
    op: str = "sum",
    freq: str = "YS",
    resample_before_rl: bool = True,
    **indexer,
) -> xarray.DataArray:
    """Longest dry spell.

    Maximum number of consecutive days in a dry period of minimum length, during which the maximum or
    accumulated precipitation within a window of the same length is under a threshold.

    Parameters
    ----------
    pr : xarray.DataArray
        Daily precipitation.
    thresh : Quantified
        Accumulated precipitation value under which a period is considered dry.
    window : int
        Number of days when the maximum or accumulated precipitation is under threshold.
    op : {"max", "sum"}
        Reduce operation.
    freq : str
        Resampling frequency.
    indexer
        Indexing parameters to compute the indicator on a temporal subset of the data.
        It accepts the same arguments as :py:func:`xclim.indices.generic.select_time`.
        Indexing is done after finding the dry days, but before finding the spells.

    Returns
    -------
    xarray.DataArray, [days]
        The {freq} longest spell in dry periods of minimum {window} days.

    Notes
    -----
    The algorithm assumes days before and after the timeseries are "wet", meaning that the condition for being
    considered part of a dry spell is stricter on the edges. For example, with `window=3` and `op='sum'`, the first day
    of the series is considered part of a dry spell only if the accumulated precipitation within the first three days is
    under the threshold. In comparison, a day in the middle of the series is considered part of a dry spell if any of
    the three 3-day periods of which it is part are considered dry (so a total of five days are included in the
    computation, compared to only three).
    """
    pram = rate2amount(convert_units_to(pr, "mm/d", context="hydro"), out_units="mm")
    thresh = convert_units_to(thresh, pram, context="hydro")

    pram_pad = pram.pad(time=(0, window))
    mask = getattr(pram_pad.rolling(time=window), op)() < thresh
    dry = (mask.rolling(time=window).sum() >= 1).shift(time=-(window - 1))
    dry = dry.isel(time=slice(0, pram.time.size)).astype(float)

    dry = select_time(dry, **indexer)

    out = rl.resample_and_rl(
        dry,
        resample_before_rl,
        rl.longest_run,
        freq=freq,
    )
    return to_agg_units(out, pram, "count")


@declare_units(pr="[precipitation]", thresh="[length]")
def wet_spell_frequency(
    pr: xarray.DataArray,
    thresh: Quantified = "1.0 mm",
    window: int = 3,
    freq: str = "YS",
    resample_before_rl: bool = True,
    op: str = "sum",
) -> xarray.DataArray:
    """Return the number of wet periods of n days and more.

    Periods during which the accumulated or maximal daily precipitation amount on a window of n days is over threshold.

    Parameters
    ----------
    pr : xarray.DataArray
        Daily precipitation.
    thresh : Quantified
        Precipitation amount over which a period is considered dry.
        The value against which the threshold is compared depends on  `op` .
    window : int
        Minimum length of the spells.
    freq : str
      Resampling frequency.
    resample_before_rl : bool
      Determines if the resampling should take place before or after the run
      length encoding (or a similar algorithm) is applied to runs.
    op: {"sum","max"}
      Operation to perform on the window.
      Default is "sum", which checks that the sum of accumulated precipitation over the whole window is more than the
      threshold.
      "max" checks that the maximal daily precipitation amount within the window is more than the threshold.
      This is the same as verifying that each individual day is above the threshold.

    Returns
    -------
    xarray.DataArray, [unitless]
        The {freq} number of wet periods of minimum {window} days.

    Examples
    --------
    >>> from xclim.indices import wet_spell_frequency
    >>> pr = xr.open_dataset(path_to_pr_file).pr
    >>> dsf = wet_spell_frequency(pr=pr, op="sum")
    >>> dsf = wet_spell_frequency(pr=pr, op="max")
    """
    pram = rate2amount(convert_units_to(pr, "mm/d", context="hydro"), out_units="mm")
    thresh = convert_units_to(thresh, pram, context="hydro")

    agg_pr = getattr(pram.rolling(time=window, center=True), op)()
    cond = agg_pr >= thresh
    out = rl.resample_and_rl(
        cond,
        resample_before_rl,
        rl.windowed_run_events,
        window=1,
        freq=freq,
    )
    out.attrs["units"] = ""
    return out


@declare_units(pr="[precipitation]", thresh="[length]")
def wet_spell_total_length(
    pr: xarray.DataArray,
    thresh: Quantified = "1.0 mm",
    window: int = 3,
    op: str = "sum",
    freq: str = "YS",
    resample_before_rl: bool = True,
    **indexer,
) -> xarray.DataArray:
    """Total length of dry spells.

    Total number of days in wet periods of a minimum length, during which the maximum or
    accumulated precipitation within a window of the same length is over a threshold.

    Parameters
    ----------
    pr : xarray.DataArray
        Daily precipitation.
    thresh : Quantified
        Accumulated precipitation value over which a period is considered dry.
    window : int
        Number of days when the maximum or accumulated precipitation is over threshold.
    op : {"max", "sum"}
        Reduce operation.
    freq : str
        Resampling frequency.
    indexer
        Indexing parameters to compute the indicator on a temporal subset of the data.
        It accepts the same arguments as :py:func:`xclim.indices.generic.select_time`.
        Indexing is done after finding the dry days, but before finding the spells.

    Returns
    -------
    xarray.DataArray, [days]
        The {freq} total number of days in wet periods of minimum {window} days.

    Notes
    -----
    The algorithm assumes days before and after the timeseries are "dry", meaning that the condition for being
    considered part of a wet spell is stricter on the edges. For example, with `window=3` and `op='sum'`, the first day
    of the series is considered part of a wet spell only if the accumulated precipitation within the first three days is
    over the threshold. In comparison, a day in the middle of the series is considered part of a wet spell if any of
    the three 3-day periods of which it is part are considered wet (so a total of five days are included in the
    computation, compared to only three).
    """
    pram = rate2amount(convert_units_to(pr, "mm/d", context="hydro"), out_units="mm")
    thresh = convert_units_to(thresh, pram, context="hydro")

    pram_pad = pram.pad(time=(0, window))
    mask = getattr(pram_pad.rolling(time=window), op)() >= thresh
    wet = (mask.rolling(time=window).sum() < 1).shift(time=-(window - 1))
    wet = wet.isel(time=slice(0, pram.time.size)).astype(float)

    wet = select_time(wet, **indexer)

    out = rl.resample_and_rl(
        wet,
        resample_before_rl,
        rl.windowed_run_count,
        window=1,
        freq=freq,
    )
    return to_agg_units(out, pram, "count")


@declare_units(pr="[precipitation]", thresh="[length]")
def wet_spell_max_length(
    pr: xarray.DataArray,
    thresh: Quantified = "1.0 mm",
    window: int = 1,
    op: str = "sum",
    freq: str = "YS",
    resample_before_rl: bool = True,
    **indexer,
) -> xarray.DataArray:
    """Longest wet spell.

    Maximum number of consecutive days in a wet period of minimum length, during which the maximum or
    accumulated precipitation within a window of the same length is over a threshold.

    Parameters
    ----------
    pr : xarray.DataArray
        Daily precipitation.
    thresh : Quantified
        Accumulated precipitation value over which a period is considered dry.
    window : int
        Number of days when the maximum or accumulated precipitation is over threshold.
    op : {"max", "sum"}
        Reduce operation.
    freq : str
        Resampling frequency.
    indexer
        Indexing parameters to compute the indicator on a temporal subset of the data.
        It accepts the same arguments as :py:func:`xclim.indices.generic.select_time`.
        Indexing is done after finding the dry days, but before finding the spells.

    Returns
    -------
    xarray.DataArray, [days]
        The {freq} longest spell in wet periods of minimum {window} days.

    Notes
    -----
    The algorithm assumes days before and after the timeseries are "dry", meaning that the condition for being
    considered part of a wet spell is stricter on the edges. For example, with `window=3` and `op='sum'`, the first day
    of the series is considered part of a wet spell only if the accumulated precipitation within the first three days is
    over the threshold. In comparison, a day in the middle of the series is considered part of a wet spell if any of
    the three 3-day periods of which it is part are considered wet (so a total of five days are included in the
    computation, compared to only three).
    """
    pram = rate2amount(convert_units_to(pr, "mm/d", context="hydro"), out_units="mm")
    thresh = convert_units_to(thresh, pram, context="hydro")

    pram_pad = pram.pad(time=(0, window))
    mask = getattr(pram_pad.rolling(time=window), op)() >= thresh
    wet = (mask.rolling(time=window).sum() < 1).shift(time=-(window - 1))
    wet = wet.isel(time=slice(0, pram.time.size)).astype(float)

    wet = select_time(wet, **indexer)

    out = rl.resample_and_rl(
        wet,
        resample_before_rl,
        rl.longest_run,
        freq=freq,
    )
    return to_agg_units(out, pram, "count")

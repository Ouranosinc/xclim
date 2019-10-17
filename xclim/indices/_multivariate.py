import logging
import warnings
from typing import Optional

import numpy as np
import xarray as xr

from xclim import run_length as rl
from xclim import utils
from xclim.utils import declare_units
from xclim.utils import units

# logging.basicConfig(level=logging.DEBUG)
# logging.captureWarnings(True)

xr.set_options(enable_cftimeindex=True)  # Set xarray to use cftimeindex


# Frequencies : YS: year start, QS-DEC: seasons starting in december, MS: month start
# See http://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases

# -------------------------------------------------- #
# ATTENTION: ASSUME ALL INDICES WRONG UNTIL TESTED ! #
# -------------------------------------------------- #

__all__ = [
    "cold_spell_duration_index",
    "cold_and_dry_days",
    "daily_freezethaw_cycles",
    "daily_temperature_range",
    "daily_temperature_range_variability",
    "days_over_precip_thresh",
    "extreme_temperature_range",
    "fraction_over_precip_thresh",
    "heat_wave_frequency",
    "heat_wave_max_length",
    "liquid_precip_ratio",
    "rain_on_frozen_ground_days",
    "tg90p",
    "tg10p",
    "tn90p",
    "tn10p",
    "tx90p",
    "tx10p",
    "tx_tn_days_above",
    "warm_spell_duration_index",
    "winter_rain_ratio",
]


@declare_units("days", tasmin="[temperature]", tn10="[temperature]")
def cold_spell_duration_index(
    tasmin: xr.DataArray, tn10: float, window: int = 6, freq: Optional[str] = None
) -> xr.DataArray:
    r"""Cold spell duration index

    Number of days with at least six consecutive days where the daily minimum temperature is below the 10th
    percentile.

    Parameters
    ----------
    tasmin : xr.DataArray
      Minimum daily temperature.
    tn10 : float
      10th percentile of daily minimum temperature.
    window : int
      Minimum number of days with temperature below threshold to qualify as a cold spell. Default: 6.
    freq : Optional[str]
      Resampling frequency; Defaults to "YS".

    Returns
    -------
    xr.DataArray
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
    >>> import xclim.utils as xcu
    >>> tn10 = xcu.percentile_doy(historical_tasmin, per=.1)
    >>> cold_spell_duration_index(reference_tasmin, tn10)
    """
    freq = freq or "YS"

    tn10 = utils.convert_units_to(tn10, tasmin)

    # Create time series out of doy values.
    thresh = utils.resample_doy(tn10, tasmin)

    below = tasmin < thresh

    return below.resample(time=freq).apply(
        rl.windowed_run_count, window=window, dim="time"
    )


def cold_and_dry_days(
    tas: xr.DataArray, tgin25, pr, wet25, freq: Optional[str] = None
) -> xr.DataArray:
    r"""Cold and dry days.

    Returns the total number of days where "Cold" and "Dry" conditions coincide.

    Parameters
    ----------
    tas : xr.DataArray
      Mean daily temperature values [℃] or [K]
    tgin25 : xr.DataArray
      First quartile of daily mean temperature computed by month.
    pr : xr.DataArray
      Daily precipitation.
    wet25 : xr.DataArray
      First quartile of daily total precipitation computed by month.
    freq : Optional[str]
      Resampling frequency; Defaults to "YS".

    Returns
    -------
    xr.DataArray
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
    # freq = freq or "YS"
    # c1 = tas < utils.convert_units_to(tgin25, tas)
    # c2 = (pr > utils.convert_units_to('1 mm', pr)) * (pr < utils.convert_units_to(wet25, pr))

    # c = (c1 * c2) * 1
    # return c.resample(time=freq).sum(dim='time')


@declare_units(
    "days",
    tasmax="[temperature]",
    tasmin="[temperature]",
    thresh_tasmax="[temperature]",
    thresh_tasmin="[temperature]",
)
def daily_freezethaw_cycles(
    tasmax: xr.DataArray,
    tasmin: xr.DataArray,
    thresh_tasmax: str = "UNSET 0 degC",
    thresh_tasmin: str = "UNSET 0 degC",
    freq: Optional[str] = None,
) -> xr.DataArray:
    r"""Number of days with a diurnal freeze-thaw cycle

    The number of days where Tmax > thresh_tasmax and Tmin <= thresh_tasmin.

    Parameters
    ----------
    tasmax : xr.DataArray
      Maximum daily temperature [℃] or [K]
    tasmin : xr.DataArray
      Minimum daily temperature values [℃] or [K]
    thresh_tasmax : str
      The temperature threshold needed to trigger a thaw event [℃] or [K]. Default : '0 degC'
    thresh_tasmin : str
      The temperature threshold needed to trigger a freeze event [℃] or [K]. Default : '0 degC'
    freq : Optional[str]
      Resampling frequency; Defaults to "YS".

    Returns
    -------
    xr.DataArray
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
    freq = freq or "YS"

    if thresh_tasmax.startswith("UNSET ") or thresh_tasmin.startswith("UNSET"):
        thresh_tasmax, thresh_tasmin = (
            thresh_tasmax.replace("UNSET ", ""),
            thresh_tasmin.replace("UNSET ", ""),
        )

    thaw_threshold = utils.convert_units_to(thresh_tasmax, tasmax)
    freeze_threshold = utils.convert_units_to(thresh_tasmin, tasmin)

    ft = (tasmin <= freeze_threshold) * (tasmax > thaw_threshold) * 1
    out = ft.resample(time=freq).sum(dim="time")

    return out


@declare_units("K", tasmax="[temperature]", tasmin="[temperature]")
def daily_temperature_range(tasmax, tasmin, freq: Optional[str] = None) -> xr.DataArray:
    r"""Mean of daily temperature range.

    The mean difference between the daily maximum temperature and the daily minimum temperature.

    Parameters
    ----------
    tasmax : xarray.DataArray
      Maximum daily temperature values [℃] or [K]
    tasmin : xarray.DataArray
      Minimum daily temperature values [℃] or [K]
    freq : Optional[str]
      Resampling frequency; Defaults to "YS".

    Returns
    -------
    xr.DataArray
      The average variation in daily temperature range for the given time period.

    Notes
    -----
    Let :math:`TX_{ij}` and :math:`TN_{ij}` be the daily maximum and minimum temperature at day :math:`i`
    of period :math:`j`. Then the mean diurnal temperature range in period :math:`j` is:

    .. math::

        DTR_j = \frac{ \sum_{i=1}^I (TX_{ij} - TN_{ij}) }{I}
    """
    freq = freq or "YS"
    dtr = tasmax - tasmin
    out = dtr.resample(time=freq).mean(dim="time", keep_attrs=True)
    out.attrs["units"] = tasmax.units
    return out


@declare_units("K", tasmax="[temperature]", tasmin="[temperature]")
def daily_temperature_range_variability(
    tasmax: xr.DataArray, tasmin: xr.DataArray, freq: Optional[str] = None
) -> xr.DataArray:
    r"""Mean absolute day-to-day variation in daily temperature range.

    Mean absolute day-to-day variation in daily temperature range.

    Parameters
    ----------
    tasmax : xr.DataArray
      Maximum daily temperature values [℃] or [K]
    tasmin : xr.DataArray
      Minimum daily temperature values [℃] or [K]
    freq : Optional[str]
      Resampling frequency; Defaults to "YS".

    Returns
    -------
    xr.DataArray
      The average day-to-day variation in daily temperature range for the given time period.

    Notes
    -----
    Let :math:`TX_{ij}` and :math:`TN_{ij}` be the daily maximum and minimum temperature at
    day :math:`i` of period :math:`j`. Then calculated is the absolute day-to-day differences in
    period :math:`j` is:

    .. math::

       vDTR_j = \frac{ \sum_{i=2}^{I} |(TX_{ij}-TN_{ij})-(TX_{i-1,j}-TN_{i-1,j})| }{I}
    """

    vdtr = abs((tasmax - tasmin).diff(dim="time"))
    out = vdtr.resample(time=freq).mean(dim="time")
    out.attrs["units"] = tasmax.units
    return out


@declare_units("K", tasmax="[temperature]", tasmin="[temperature]")
def extreme_temperature_range(
    tasmax: xr.DataArray, tasmin: xr.DataArray, freq: Optional[str] = None
) -> xr.DataArray:
    r"""Extreme intra-period temperature range.

    The maximum of max temperature (TXx) minus the minimum of min temperature (TNn) for the given time period.

    Parameters
    ----------
    tasmax : xr.DataArray
      Maximum daily temperature values [℃] or [K]
    tasmin : xr.DataArray
      Minimum daily temperature values [℃] or [K]
    freq : Optional[str[
      Resampling frequency; Defaults to "YS".

    Returns
    -------
    xr.DataArray
      Extreme intra-period temperature range for the given time period.

    Notes
    -----
    Let :math:`TX_{ij}` and :math:`TN_{ij}` be the daily maximum and minimum temperature at day :math:`i`
    of period :math:`j`. Then the extreme temperature range in period :math:`j` is:

    .. math::

        ETR_j = max(TX_{ij}) - min(TN_{ij})
    """
    freq = freq or "YS"
    tx_max = tasmax.resample(time=freq).max(dim="time")
    tn_min = tasmin.resample(time=freq).min(dim="time")

    out = tx_max - tn_min
    out.attrs["units"] = tasmax.units
    return out


@declare_units(
    "",
    tasmin="[temperature]",
    tasmax="[temperature]",
    thresh_tasmin="[temperature]",
    thresh_tasmax="[temperature]",
)
def heat_wave_frequency(
    tasmin: xr.DataArray,
    tasmax: xr.DataArray,
    thresh_tasmin: str = "22.0 degC",
    thresh_tasmax: str = "30 degC",
    window: int = 3,
    freq: Optional[str] = None,
) -> xr.DataArray:
    # Dev note : we should decide if it is deg K or C
    r"""Heat wave frequency

    Number of heat waves over a given period. A heat wave is defined as an event
    where the minimum and maximum daily temperature both exceeds specific thresholds
    over a minimum number of days.

    Parameters
    ----------

    tasmin : xr.DataArray
      Minimum daily temperature [℃] or [K]
    tasmax : xr.DataArray
      Maximum daily temperature [℃] or [K]
    thresh_tasmin : str
      The minimum temperature threshold needed to trigger a heatwave event [℃] or [K]. Default : '22 degC'
    thresh_tasmax : str
      The maximum temperature threshold needed to trigger a heatwave event [℃] or [K]. Default : '30 degC'
    window : int
      Minimum number of days with temperatures above thresholds to qualify as a heatwave.
    freq : Optional[str]
      Resampling frequency; Defaults to "YS".

    Returns
    -------
    xr.DataArray
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
    freq = freq or "YS"
    thresh_tasmax = utils.convert_units_to(thresh_tasmax, tasmax)
    thresh_tasmin = utils.convert_units_to(thresh_tasmin, tasmin)

    cond = (tasmin > thresh_tasmin) & (tasmax > thresh_tasmax)
    group = cond.resample(time=freq)
    return group.apply(rl.windowed_run_events, window=window, dim="time")


@declare_units(
    "days",
    tasmin="[temperature]",
    tasmax="[temperature]",
    thresh_tasmin="[temperature]",
    thresh_tasmax="[temperature]",
)
def heat_wave_max_length(
    tasmin: xr.DataArray,
    tasmax: xr.DataArray,
    thresh_tasmin: str = "22.0 degC",
    thresh_tasmax: str = "30 degC",
    window: int = 3,
    freq: Optional[str] = None,
) -> xr.DataArray:
    # Dev note : we should decide if it is deg K or C
    r"""Heat wave max length

    Maximum length of heat waves over a given period. A heat wave is defined as an event
    where the minimum and maximum daily temperature both exceeds specific thresholds
    over a minimum number of days.

    By definition heat_wave_max_length must be >= window.

    Parameters
    ----------

    tasmin : xr.DataArray
      Minimum daily temperature [℃] or [K]
    tasmax : xr.DataArray
      Maximum daily temperature [℃] or [K]
    thresh_tasmin : str
      The minimum temperature threshold needed to trigger a heatwave event [℃] or [K]. Default : '22 degC'
    thresh_tasmax : str
      The maximum temperature threshold needed to trigger a heatwave event [℃] or [K]. Default : '30 degC'
    window : int
      Minimum number of days with temperatures above thresholds to qualify as a heatwave.
    freq : Optional[str]
      Resampling frequency; Defaults to "YS".

    Returns
    -------
    xr.DataArray
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
    freq = freq or "YS"
    thresh_tasmax = utils.convert_units_to(thresh_tasmax, tasmax)
    thresh_tasmin = utils.convert_units_to(thresh_tasmin, tasmin)

    cond = (tasmin > thresh_tasmin) & (tasmax > thresh_tasmax)
    group = cond.resample(time=freq)
    max_l = group.apply(rl.longest_run, dim="time")
    return max_l.where(max_l >= window, 0)


@declare_units("", pr="[precipitation]", prsn="[precipitation]", tas="[temperature]")
def liquid_precip_ratio(
    pr: xr.DataArray,
    prsn: xr.DataArray = None,
    tas: xr.DataArray = None,
    freq: Optional[str] = None,
) -> xr.DataArray:
    r"""Ratio of rainfall to total precipitation

    The ratio of total liquid precipitation over the total precipitation. If solid precipitation is not provided,
    then precipitation is assumed solid if the temperature is below 0°C.

    Parameters
    ----------
    pr : xr.DataArray
      Mean daily precipitation flux [Kg m-2 s-1] or [mm].
    prsn : xr.DataArray
      Mean daily solid precipitation flux [Kg m-2 s-1] or [mm].
    tas : xr.DataArray
      Mean daily temperature [℃] or [K]
    freq : Optional[str]
      Resampling frequency; Defaults to "QS-DEC".

    Returns
    -------
    xr.DataArray
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
    freq = freq or "QS-DEC"

    if prsn is None:
        tu = units.parse_units(tas.attrs["units"].replace("-", "**-"))
        fu = "degC"
        frz = 0
        if fu != tu:
            frz = units.convert(frz, fu, tu)
        prsn = pr.where(tas < frz, 0)

    tot = pr.resample(time=freq).sum(dim="time")
    rain = tot - prsn.resample(time=freq).sum(dim="time")
    ratio = rain / tot
    return ratio


@declare_units(
    "days", pr="[precipitation]", tas="[temperature]", thresh="[precipitation]"
)
def rain_on_frozen_ground_days(
    pr: xr.DataArray,
    tas: xr.DataArray,
    thresh: str = "1 mm/d",
    freq: Optional[str] = None,
) -> xr.DataArray:
    """Number of rain on frozen ground events

    Number of days with rain above a threshold after a series of seven days below freezing temperature.
    Precipitation is assumed to be rain when the temperature is above 0℃.

    Parameters
    ----------
    pr : xr.DataArray
      Mean daily precipitation flux [Kg m-2 s-1] or [mm]
    tas : xr.DataArray
      Mean daily temperature [℃] or [K]
    thresh : str
      Precipitation threshold to consider a day as a rain event. Default : '1 mm/d'
    freq : Optional[str]
      Resampling frequency; Defaults to "YS".

    Returns
    -------
    xr.DataArray
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
    freq = freq or "YS"
    t = utils.convert_units_to(thresh, pr)
    frz = utils.convert_units_to("0 C", tas)

    def func(x, axis):
        """Check that temperature conditions are below 0 for seven days and above after."""
        frozen = x == np.array([0, 0, 0, 0, 0, 0, 0, 1], bool)
        return frozen.all(axis=axis)

    tcond = (tas > frz).rolling(time=8).reduce(func)
    pcond = pr > t

    return (tcond * pcond * 1).resample(time=freq).sum(dim="time")


@declare_units(
    "days", pr="[precipitation]", per="[precipitation]", thresh="[precipitation]"
)
def days_over_precip_thresh(
    pr: xr.DataArray,
    per: xr.DataArray,
    thresh: str = "1 mm/day",
    freq: Optional[str] = None,
) -> xr.DataArray:
    r"""Number of wet days with daily precipitation over a given percentile.

    Number of days over period where the precipitation is above a threshold defining wet days and above a given
    percentile for that day.

    Parameters
    ----------
    pr : xr.DataArray
      Mean daily precipitation flux [Kg m-2 s-1] or [mm/day]
    per : xr.DataArray
      Daily percentile of wet day precipitation flux [Kg m-2 s-1] or [mm/day].
    thresh : str
       Precipitation value over which a day is considered wet [Kg m-2 s-1] or [mm/day].
    freq : Optional[str]
      Resampling frequency; Defaults to "YS".

    Returns
    -------
    xr.DataArray
      Count of days with daily precipitation above the given percentile [days]

    Example
    -------
    >>> p75 = historical_pr.quantile(.75, dim="time", keep_attrs=True)
    >>> r75p = days_over_precip_thresh(pr, p75)
    """
    freq = freq or "YS"
    per = utils.convert_units_to(per, pr)
    thresh = utils.convert_units_to(thresh, pr)

    tp = np.maximum(per, thresh)
    if "dayofyear" in per.coords:
        # Create time series out of doy values.
        tp = utils.resample_doy(tp, pr)

    # Compute the days where precip is both over the wet day threshold and the percentile threshold.
    over = pr > tp

    return over.resample(time=freq).sum(dim="time")


@declare_units(
    "", pr="[precipitation]", per="[precipitation]", thresh="[precipitation]"
)
def fraction_over_precip_thresh(
    pr: xr.DataArray,
    per: xr.DataArray,
    thresh: str = "1 mm/day",
    freq: Optional[str] = None,
) -> xr.DataArray:
    r"""Fraction of precipitation due to wet days with daily precipitation over a given percentile.

    Percentage of the total precipitation over period occurring in days where the precipitation is above a threshold
    defining wet days and above a given percentile for that day.

    Parameters
    ----------
    pr : xr.DataArray
      Mean daily precipitation flux [Kg m-2 s-1] or [mm/day].
    per : xr.DataArray
      Daily percentile of wet day precipitation flux [Kg m-2 s-1] or [mm/day].
    thresh : str
       Precipitation value over which a day is considered wet [Kg m-2 s-1] or [mm/day].
    freq : Optional[str]
      Resampling frequency; Defaults to "YS".

    Returns
    -------
    xr.DataArray
      Fraction of precipitation over threshold during wet days days.

    """
    freq = freq or "YS"
    per = utils.convert_units_to(per, pr)
    thresh = utils.convert_units_to(thresh, pr)

    tp = np.maximum(per, thresh)
    if "dayofyear" in per.coords:
        # Create time series out of doy values.
        tp = utils.resample_doy(tp, pr)

    # Total precip during wet days over period
    total = pr.where(pr > thresh).resample(time=freq).sum(dim="time")

    # Compute the days where precip is both over the wet day threshold and the percentile threshold.
    over = pr.where(pr > tp).resample(time=freq).sum(dim="time")

    return over / total


@declare_units("days", tas="[temperature]", t90="[temperature]")
def tg90p(
    tas: xr.DataArray, t90: xr.DataArray, freq: Optional[str] = None
) -> xr.DataArray:
    r"""Number of days with daily mean temperature over the 90th percentile.

    Number of days with daily mean temperature over the 90th percentile.

    Parameters
    ----------
    tas : xr.DataArray
      Mean daily temperature [℃] or [K]
    t90 : xr.DataArray
      90th percentile of daily mean temperature [℃] or [K]
    freq : Optional[str]
      Resampling frequency; Defaults to "YS".

    Returns
    -------
    xr.DataArray
      Count of days with daily mean temperature below the 10th percentile [days]

    Notes
    -----
    The 90th percentile should be computed for a 5 day window centered on each calendar day for a reference period.

    Example
    -------
    >>> t90 = percentile_doy(historical_tas, per=0.9)
    >>> hot_days = tg90p(tas, t90)
    """
    t90 = utils.convert_units_to(t90, tas)

    # Create time series out of doy values.
    thresh = utils.resample_doy(t90, tas)

    # Identify the days over the 90th percentile
    over = tas > thresh

    return over.resample(time=freq).sum(dim="time")


@declare_units("days", tas="[temperature]", t10="[temperature]")
def tg10p(
    tas: xr.DataArray, t10: xr.DataArray, freq: Optional[str] = None
) -> xr.DataArray:
    r"""Number of days with daily mean temperature below the 10th percentile.

    Number of days with daily mean temperature below the 10th percentile.

    Parameters
    ----------
    tas : xr.DataArray
      Mean daily temperature [℃] or [K]
    t10 : xr.DataArray
      10th percentile of daily mean temperature [℃] or [K]
    freq : Optional[str]
      Resampling frequency; Defaults to "YS".

    Returns
    -------
    xr.DataArray
      Count of days with daily mean temperature below the 10th percentile [days]

    Notes
    -----
    The 10th percentile should be computed for a 5 day window centered on each calendar day for a reference period.

    Example
    -------
    >>> t10 = percentile_doy(historical_tas, per=0.1)
    >>> cold_days = tg10p(tas, t10)
    """
    freq = freq or "YS"
    t10 = utils.convert_units_to(t10, tas)

    # Create time series out of doy values.
    thresh = utils.resample_doy(t10, tas)

    # Identify the days below the 10th percentile
    below = tas < thresh

    return below.resample(time=freq).sum(dim="time")


@declare_units("days", tasmin="[temperature]", t90="[temperature]")
def tn90p(
    tasmin: xr.DataArray, t90: xr.DataArray, freq: Optional[str] = None
) -> xr.DataArray:
    r"""Number of days with daily minimum temperature over the 90th percentile.

    Number of days with daily minimum temperature over the 90th percentile.

    Parameters
    ----------
    tasmin : xr.DataArray
      Minimum daily temperature [℃] or [K]
    t90 : xr.DataArray
      90th percentile of daily minimum temperature [℃] or [K]
    freq : Optional[str]
      Resampling frequency; Defaults to "YS".

    Returns
    -------
    xr.DataArray
      Count of days with daily minimum temperature below the 10th percentile [days]

    Notes
    -----
    The 90th percentile should be computed for a 5 day window centered on each calendar day for a reference period.

    Example
    -------
    >>> t90 = percentile_doy(historical_tas, per=0.9)
    >>> hot_days = tg90p(tas, t90)
    """
    freq = freq or "YS"
    t90 = utils.convert_units_to(t90, tasmin)

    # Create time series out of doy values.
    thresh = utils.resample_doy(t90, tasmin)

    # Identify the days with min temp above 90th percentile.
    over = tasmin > thresh

    return over.resample(time=freq).sum(dim="time")


@declare_units("days", tasmin="[temperature]", t10="[temperature]")
def tn10p(
    tasmin: xr.DataArray, t10: xr.DataArray, freq: Optional[str] = None
) -> xr.DataArray:
    r"""Number of days with daily minimum temperature below the 10th percentile.

    Number of days with daily minimum temperature below the 10th percentile.

    Parameters
    ----------

    tasmin : xr.DataArray
      Mean daily temperature [℃] or [K]
    t10 : xr.DataArray
      10th percentile of daily minimum temperature [℃] or [K]
    freq : Optional[str]
      Resampling frequency; Defaults to "YS".

    Returns
    -------
    xr.DataArray
      Count of days with daily minimum temperature below the 10th percentile [days]

    Notes
    -----
    The 10th percentile should be computed for a 5 day window centered on each calendar day for a reference period.

    Example
    -------
    >>> t10 = percentile_doy(historical_tas, per=0.1)
    >>> cold_days = tg10p(tas, t10)
    """
    freq = freq or "YS"
    t10 = utils.convert_units_to(t10, tasmin)

    # Create time series out of doy values.
    thresh = utils.resample_doy(t10, tasmin)

    # Identify the days below the 10th percentile
    below = tasmin < thresh

    return below.resample(time=freq).sum(dim="time")


@declare_units("days", tasmax="[temperature]", t90="[temperature]")
def tx90p(
    tasmax: xr.DataArray, t90: xr.DataArray, freq: Optional[str] = None
) -> xr.DataArray:
    r"""Number of days with daily maximum temperature over the 90th percentile.

    Number of days with daily maximum temperature over the 90th percentile.

    Parameters
    ----------
    tasmax : xr.DataArray
      Maximum daily temperature [℃] or [K]
    t90 : xr.DataArray
      90th percentile of daily maximum temperature [℃] or [K]
    freq : Optional[str]
      Resampling frequency; Defaults to "YS".

    Returns
    -------
    xr.DataArray
      Count of days with daily maximum temperature below the 10th percentile [days]

    Notes
    -----
    The 90th percentile should be computed for a 5 day window centered on each calendar day for a reference period.

    Example
    -------
    >>> t90 = percentile_doy(historical_tas, per=0.9)
    >>> hot_days = tg90p(tas, t90)
    """
    freq = freq or "YS"
    t90 = utils.convert_units_to(t90, tasmax)

    # Create time series out of doy values.
    thresh = utils.resample_doy(t90, tasmax)

    # Identify the days with max temp above 90th percentile.
    over = tasmax > thresh

    return over.resample(time=freq).sum(dim="time")


@declare_units("days", tasmax="[temperature]", t10="[temperature]")
def tx10p(
    tasmax: xr.DataArray, t10: xr.DataArray, freq: Optional[str] = None
) -> xr.DataArray:
    r"""Number of days with daily maximum temperature below the 10th percentile.

    Number of days with daily maximum temperature below the 10th percentile.

    Parameters
    ----------
    tasmax : xr.DataArray
      Maximum daily temperature [℃] or [K]
    t10 : xr.DataArray
      10th percentile of daily maximum temperature [℃] or [K]
    freq : Optional[str]
      Resampling frequency; Defaults to "YS".

    Returns
    -------
    xr.DataArray
      Count of days with daily maximum temperature below the 10th percentile [days]

    Notes
    -----
    The 10th percentile should be computed for a 5 day window centered on each calendar day for a reference period.

    Example
    -------
    >>> t10 = percentile_doy(historical_tas, per=0.1)
    >>> cold_days = tg10p(tas, t10)
    """
    freq = freq or "YS"
    t10 = utils.convert_units_to(t10, tasmax)

    # Create time series out of doy values.
    thresh = utils.resample_doy(t10, tasmax)

    # Identify the days below the 10th percentile
    below = tasmax < thresh

    return below.resample(time=freq).sum(dim="time")


@declare_units(
    "days",
    tasmin="[temperature]",
    tasmax="[temperature]",
    thresh_tasmin="[temperature]",
    thresh_tasmax="[temperature]",
)
def tx_tn_days_above(
    tasmin: xr.DataArray,
    tasmax: xr.DataArray,
    thresh_tasmin: str = "22 degC",
    thresh_tasmax: str = "30 degC",
    freq: Optional[str] = "YS",
) -> xr.DataArray:
    r"""Number of days with both hot maximum and minimum daily temperatures.

    The number of days per period with tasmin above a threshold and tasmax above another threshold.

    Parameters
    ----------
    tasmin : xr.DataArray
      Minimum daily temperature [℃] or [K]
    tasmax : xr.DataArray
      Maximum daily temperature [℃] or [K]
    thresh_tasmin : str
      Threshold temperature for tasmin on which to base evaluation [℃] or [K]. Default : '22 degC'
    thresh_tasmax : str
      Threshold temperature for tasmax on which to base evaluation [℃] or [K]. Default : '30 degC'
    freq : Optional[str]
      Resampling frequency; Defaults to "YS".

    Returns
    -------
    xr.DataArray
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
    freq = freq or "YS"
    thresh_tasmax = utils.convert_units_to(thresh_tasmax, tasmax)
    thresh_tasmin = utils.convert_units_to(thresh_tasmin, tasmin)
    events = ((tasmin > thresh_tasmin) & (tasmax > thresh_tasmax)) * 1
    return events.resample(time=freq).sum(dim="time")


@declare_units("days", tasmax="[temperature]", tx90="[temperature]")
def warm_spell_duration_index(
    tasmax: xr.DataArray, tx90: float, window: int = 6, freq: Optional[str] = None
) -> xr.DataArray:
    r"""Warm spell duration index

    Number of days with at least six consecutive days where the daily maximum temperature is above the 90th
    percentile. The 90th percentile should be computed for a 5-day window centred on each calendar day in the
    1961-1990 period.

    Parameters
    ----------
    tasmax : xr.DataArray
      Maximum daily temperature [℃] or [K]
    tx90 : float
      90th percentile of daily maximum temperature [℃] or [K]
    window : int
      Minimum number of days with temperature below threshold to qualify as a warm spell.
    freq : Optional[str]
      Resampling frequency; Defaults to "YS".

    Returns
    -------
    xr.DataArray
      Count of days with at least six consecutive days where the daily maximum temperature is above the 90th
      percentile [days].

    References
    ----------
    From the Expert Team on Climate Change Detection, Monitoring and Indices (ETCCDMI).
    Used in Alexander, L. V., et al. (2006), Global observed changes in daily climate extremes of temperature and
    precipitation, J. Geophys. Res., 111, D05109, doi: 10.1029/2005JD006290.

    """
    freq = freq or "YS"
    # Create time series out of doy values.
    thresh = utils.resample_doy(tx90, tasmax)

    above = tasmax > thresh

    return above.resample(time=freq).apply(
        rl.windowed_run_count, window=window, dim="time"
    )


@declare_units("", pr="[precipitation]", prsn="[precipitation]", tas="[temperature]")
def winter_rain_ratio(
    *,
    pr: xr.DataArray = None,
    prsn: xr.DataArray = None,
    tas: xr.DataArray = None,
    freq: Optional[str] = None
) -> xr.DataArray:
    """Ratio of rainfall to total precipitation during winter

    The ratio of total liquid precipitation over the total precipitation over the winter months (DJF. If solid
    precipitation is not provided, then precipitation is assumed solid if the temperature is below 0°C.

    Parameters
    ----------
    pr : xr.DataArray
      Mean daily precipitation flux [Kg m-2 s-1] or [mm].
    prsn : xr.DataArray
      Mean daily solid precipitation flux [Kg m-2 s-1] or [mm].
    tas : xr.DataArray
      Mean daily temperature [℃] or [K]
    freq : str
      Resampling frequency; Defaults to "QS-DEC".

    Returns
    -------
    xr.DataArray
      Ratio of rainfall to total precipitation during winter months (DJF)
    """
    freq = freq or "QS-DEC"
    ratio = liquid_precip_ratio(pr, prsn, tas, freq=freq)
    winter = ratio.indexes["time"].month == 12
    return ratio[winter]

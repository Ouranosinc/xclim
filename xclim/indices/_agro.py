# noqa: D100
import datetime
import math
from typing import Optional, Tuple


import numpy as np
import xarray

import xclim.indices as xci
import xclim.indices.run_length as rl
from xclim.core.calendar import select_time
from xclim.core.units import convert_units_to, declare_units, rate2amount, to_agg_units
from xclim.core.utils import DayOfYearStr
from xclim.indices._threshold import first_day_above, first_day_below, freshet_start
from xclim.indices.generic import aggregate_between_dates, day_lengths

# Frequencies : YS: year start, QS-DEC: seasons starting in december, MS: month start
# See https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html

# -------------------------------------------------- #
# ATTENTION: ASSUME ALL INDICES WRONG UNTIL TESTED ! #
# -------------------------------------------------- #

__all__ = [
    "biologically_effective_degree_days",
    "huglin_index",
    "cool_night_index",
    "corn_heat_units",
    "dry_spell_frequency",
    "dry_spell_total_length",
    "effective_growing_degree_days",
    "latitude_temperature_index",
    "qian_weighted_mean_average",
    "water_budget",
    "rain_season",
    "rain_season_start",
    "rain_season_end",
    "rain_season_length",
    "rain_season_prcptot",
]


@declare_units(
    tasmin="[temperature]",
    tasmax="[temperature]",
    thresh_tasmin="[temperature]",
    thresh_tasmax="[temperature]",
)
def corn_heat_units(
    tasmin: xarray.DataArray,
    tasmax: xarray.DataArray,
    thresh_tasmin: str = "4.44 degC",
    thresh_tasmax: str = "10 degC",
) -> xarray.DataArray:
    r"""Corn heat units.

    Temperature-based index used to estimate the development of corn crops.

    Parameters
    ----------
    tasmin : xarray.DataArray
      Minimum daily temperature.
    tasmax : xarray.DataArray
      Maximum daily temperature.
    thresh_tasmin : str
      The minimum temperature threshold needed for corn growth.
    thresh_tasmax : str
      The maximum temperature threshold needed for corn growth.

    Returns
    -------
    xarray.DataArray, [dimensionless]
      Daily corn heat units.

    Notes
    -----
    The thresholds of 4.44°C for minimum temperatures and 10°C for maximum temperatures were selected following
    the assumption that no growth occurs below these values.

    Let :math:`TX_{i}` and :math:`TN_{i}` be the daily maximum and minimum temperature at day :math:`i`. Then the daily
    corn heat unit is:

    .. math::
        CHU_i = \frac{YX_{i} + YN_{i}}{2}

    with

    .. math::

        YX_i & = 3.33(TX_i -10) - 0.084(TX_i -10)^2, &\text{if } TX_i > 10°C

        YN_i & = 1.8(TN_i -4.44), &\text{if } TN_i > 4.44°C

    where :math:`YX_{i}` and :math:`YN_{i}` is 0 when :math:`TX_i \leq 10°C` and :math:`TN_i \leq 4.44°C`, respectively.

    References
    ----------
    Equations from Bootsma, A., G. Tremblay et P. Filion. 1999: Analyse sur les risques associés aux unités thermiques
    disponibles pour la production de maïs et de soya au Québec. Centre de recherches de l’Est sur les céréales et
    oléagineux, Ottawa, 28 p.

    Can be found in Audet, R., Côté, H., Bachand, D. and Mailhot, A., 2012: Atlas agroclimatique du Québec. Évaluation
    des opportunités et des risques agroclimatiques dans un climat en évolution.
    """

    tasmin = convert_units_to(tasmin, "degC")
    tasmax = convert_units_to(tasmax, "degC")
    thresh_tasmin = convert_units_to(thresh_tasmin, "degC")
    thresh_tasmax = convert_units_to(thresh_tasmax, "degC")

    mask_tasmin = tasmin > thresh_tasmin
    mask_tasmax = tasmax > thresh_tasmax

    chu = (
        xarray.where(mask_tasmin, 1.8 * (tasmin - thresh_tasmin), 0)
        + xarray.where(
            mask_tasmax,
            (3.33 * (tasmax - thresh_tasmax) - 0.084 * (tasmax - thresh_tasmax) ** 2),
            0,
        )
    ) / 2

    chu.attrs["units"] = ""
    return chu


@declare_units(
    tas="[temperature]",
    tasmax="[temperature]",
    lat="[]",
    thresh="[temperature]",
)
def huglin_index(
    tas: xarray.DataArray,
    tasmax: xarray.DataArray,
    lat: xarray.DataArray,
    thresh: str = "10 degC",
    method: str = "smoothed",
    start_date: DayOfYearStr = "04-01",
    end_date: DayOfYearStr = "10-01",
    freq: str = "YS",
) -> xarray.DataArray:
    r"""Huglin Heliothermal Index.

    Growing-degree days with a base of 10°C and adjusted for latitudes between 40°N and 50°N for April to September
    (Northern Hemisphere; October to March in Southern Hemisphere).
    Used as a heat-summation metric in viticulture agroclimatology.

    Parameters
    ----------
    tas: xarray.DataArray
      Mean daily temperature.
    tasmax: xarray.DataArray
      Maximum daily temperature.
    lat: xarray.DataArray
      Latitude coordinate.
    thresh: str
      The temperature threshold.
    method: {"smoothed", "icclim", "jones"}
      The formula to use for the latitude coefficient calculation.
    start_date: DayOfYearStr
      The hemisphere-based start date to consider (north = April, south = October).
    end_date: DayOfYearStr
      The hemisphere-based start date to consider (north = October, south = April). This date is non-inclusive.
    freq : str
      Resampling frequency (default: "YS"; For Southern Hemisphere, should be "AS-JUL").

    Returns
    -------
    xarray.DataArray, [unitless]
      Huglin heliothermal index (HI).

    Notes
    -----
    Let :math:`TX_{i}` and :math:`TG_{i}` be the daily maximum and mean temperature at day :math:`i` and
    :math:`T_{thresh}` the base threshold needed for heat summation (typically, 10 degC). A day-length multiplication,
    :math:`k`, based on latitude, :math:`lat`, is also considered. Then the Huglin heliothermal index for dates between
    1 April and 30 September is:

    .. math::
        HI = \sum_{i=\text{April 1}}^{\text{September 30}} \left( \frac{TX_i  + TG_i)}{2} - T_{thresh} \right) * k

    For the `smoothed` method, the day-length multiplication factor, :math:`k`, is calculated as follows:

    .. math::
        k = f(lat) = \begin{cases}
                        1, & \text{if } |lat| <= 40 \\
                        1 + ((abs(lat) - 40) / 10) * 0.06, & \text{if } 40 < |lat| <= 50 \\
                        NaN, & \text{if } |lat| > 50 \\
                     \end{cases}

    For compatibility with ICCLIM, `end_date` should be set to `11-01`, `method` should be set to `icclim`. The
    day-length multiplication factor, :math:`k`, is calculated as follows:

    .. math::
        k = f(lat) = \begin{cases}
                        1.0, & \text{if } |lat| <= 40 \\
                        1.02, & \text{if } 40 < |lat| <= 42 \\
                        1.03, & \text{if } 42 < |lat| <= 44 \\
                        1.04, & \text{if } 44 < |lat| <= 46 \\
                        1.05, & \text{if } 46 < |lat| <= 48 \\
                        1.06, & \text{if } 48 < |lat| <= 50 \\
                        NaN, & \text{if } |lat| > 50 \\
                    \end{cases}

    A more robust day-length calculation based on latitude, calendar, day-of-year, and obliquity is available with
    `method="jones"`. See: :py:func:`xclim.indices.generic.day_lengths` or Hall and Jones (2010) for more information.

    References
    ----------
    Huglin heliothermal index originally published in Huglin, P. (1978). Nouveau mode d’évaluation des possibilités
    héliothermiques d’un milieu viticole. Dans Symposium International sur l’Écologie de la Vigne (p. 89‑98). Ministère
    de l’Agriculture et de l’Industrie Alimentaire.

    Modified day-length for Huglin heliothermal index published in Hall, A., & Jones, G. V. (2010). Spatial analysis of
    climate in winegrape-growing regions in Australia. Australian Journal of Grape and Wine Research, 16(3), 389‑404.
    https://doi.org/10.1111/j.1755-0238.2010.00100.x
    """
    tas = convert_units_to(tas, "degC")
    tasmax = convert_units_to(tasmax, "degC")
    thresh = convert_units_to(thresh, "degC")

    if method.lower() == "smoothed":
        lat_mask = abs(lat) <= 50
        lat_coefficient = ((abs(lat) - 40) / 10).clip(min=0) * 0.06
        k = 1 + xarray.where(lat_mask, lat_coefficient, np.NaN)
        k_aggregated = 1
    elif method.lower() == "icclim":
        k_f = [0, 0.02, 0.03, 0.04, 0.05, 0.06]

        k = 1 + xarray.where(
            abs(lat) <= 40,
            k_f[0],
            xarray.where(
                (40 < abs(lat)) & (abs(lat) <= 42),
                k_f[1],
                xarray.where(
                    (42 < abs(lat)) & (abs(lat) <= 44),
                    k_f[2],
                    xarray.where(
                        (44 < abs(lat)) & (abs(lat) <= 46),
                        k_f[3],
                        xarray.where(
                            (46 < abs(lat)) & (abs(lat) <= 48),
                            k_f[4],
                            xarray.where(
                                (48 < abs(lat)) & (abs(lat) <= 50), k_f[5], np.NaN
                            ),
                        ),
                    ),
                ),
            ),
        )
        k_aggregated = 1
    elif method.lower() == "jones":
        day_length = day_lengths(
            dates=tas.time,
            lat=lat,
            start_date=start_date,
            end_date=end_date,
            freq=freq,
        )
        k = 1
        k_aggregated = 2.8311e-4 * day_length + 0.30834
    else:
        raise NotImplementedError(f"'{method}' method is not implemented.")

    hi = (((tas + tasmax) / 2) - thresh).clip(min=0) * k
    hi = (
        aggregate_between_dates(hi, start=start_date, end=end_date, freq=freq)
        * k_aggregated
    )

    hi.attrs["units"] = ""
    return hi


@declare_units(
    tasmin="[temperature]",
    tasmax="[temperature]",
    lat="[]",
    thresh_tasmin="[temperature]",
    low_dtr="[temperature]",
    high_dtr="[temperature]",
    max_daily_degree_days="[temperature]",
)
def biologically_effective_degree_days(
    tasmin: xarray.DataArray,
    tasmax: xarray.DataArray,
    lat: Optional[xarray.DataArray] = None,
    thresh_tasmin: str = "10 degC",
    method: str = "gladstones",
    low_dtr: str = "10 degC",
    high_dtr: str = "13 degC",
    max_daily_degree_days: str = "9 degC",
    start_date: DayOfYearStr = "04-01",
    end_date: DayOfYearStr = "11-01",
    freq: str = "YS",
) -> xarray.DataArray:
    r"""Biologically effective growing degree days.

    Growing-degree days with a base of 10°C and an upper limit of 19°C and adjusted for latitudes between 40°N and 50°N
    for April to October (Northern Hemisphere; October to April in Southern Hemisphere). A temperature range adjustment
    also promotes small and large swings in daily temperature range. Used as a heat-summation metric in viticulture
    agroclimatology.

    Parameters
    ----------
    tasmin: xarray.DataArray
      Minimum daily temperature.
    tasmax: xarray.DataArray
      Maximum daily temperature.
    lat: xarray.DataArray, optional
      Latitude coordinate.
    thresh_tasmin: str
      The minimum temperature threshold.
    method: {"gladstones", "icclim", "jones"}
      The formula to use for the calculation.
      The "gladstones" integrates a daily temperature range and latitude coefficient. End_date should be "11-01".
      The "icclim" method ignores daily temperature range and latitude coefficient. End date should be "10-01".
      The "jones" method integrates axial tilt, latitude, and day-of-year on coefficient. End_date should be "11-01".
    low_dtr: str
      The lower bound for daily temperature range adjustment (default: 10°C).
    high_dtr: str
      The higher bound for daily temperature range adjustment (default: 13°C).
    max_daily_degree_days: str
      The maximum amount of biologically effective degrees days that can be summed daily.
    start_date: DayOfYearStr
      The hemisphere-based start date to consider (north = April, south = October).
    end_date: DayOfYearStr
      The hemisphere-based start date to consider (north = October, south = April). This date is non-inclusive.
    freq : str
      Resampling frequency (default: "YS"; For Southern Hemisphere, should be "AS-JUL").

    Returns
    -------
    xarray.DataArray
      Biologically effective growing degree days (BEDD).

    Warnings
    --------
    Lat coordinate must be provided if method is "gladstones" or "jones".

    Notes
    -----
    The tasmax ceiling of 19°C is assumed to be the max temperature beyond which no further gains from daily temperature
    occur.

    Let :math:`TX_{i}` and :math:`TN_{i}` be the daily maximum and minimum temperature at day :math:`i`, :math:`lat`
    the latitude of the point of interest, :math:`degdays_{max}` the maximum amount of degrees that can be summed per
    day (typically, 9). Then the sum of daily biologically effective growing degree day (BEDD) units between 1 April and
    31 October is:

    .. math::
        BEDD_i = \sum_{i=\text{April 1}}^{\text{October 31}} min\left( \left( max\left( \frac{TX_i  + TN_i)}{2} - 10, 0 \right) * k \right) + TR_{adj}, degdays_{max}\right)

    .. math::
        TR_{adj} = f(TX_{i}, TN_{i}) = \begin{cases}
                                0.25(TX_{i} - TN_{i} - 13), & \text{if } (TX_{i} - TN_{i}) > 13 \\
                                0, & \text{if } 10 < (TX_{i} - TN_{i}) < 13\\
                                0.25(TX_{i} - TN_{i} - 10), & \text{if } (TX_{i} - TN_{i}) < 10 \\
                                       \end{cases}

    .. math::
        k = f(lat) = 1 + \left(\frac{\left| lat  \right|}{50} * 0.06,  \text{if }40 < |lat| <50, \text{else } 0\right)

    A second version of the BEDD (`method="icclim"`) does not consider :math:`TR_{adj}` and :math:`k` and employs a
    different end date (30 September). The simplified formula is as follows:

    .. math::
        BEDD_i = \sum_{i=\text{April 1}}^{\text{September 30}} min\left( max\left(\frac{TX_i  + TN_i)}{2} - 10, 0\right), degdays_{max}\right)

    References
    ----------
    Indice originally from Gladstones, J.S. (1992). Viticulture and environment: a study of the effects of
    environment on grapegrowing and wine qualities, with emphasis on present and future areas for growing winegrapes
    in Australia. Adelaide:  Winetitles.

    ICCLIM modified formula originally from Project team ECA&D, KNMI (2013). EUMETNET/ECSN optional programme: European Climate
    Assessment & Dataset (ECA&D) - Algorithm Theoretical Basis Document (ATBD). (KNMI Project number: EPJ029135, v10.7).
    https://www.ecad.eu/documents/atbd.pdf
    """
    tasmin = convert_units_to(tasmin, "degC")
    tasmax = convert_units_to(tasmax, "degC")
    thresh_tasmin = convert_units_to(thresh_tasmin, "degC")
    max_daily_degree_days = convert_units_to(max_daily_degree_days, "degC")

    if method.lower() in ["gladstones", "jones"] and lat is not None:
        low_dtr = convert_units_to(low_dtr, "degC")
        high_dtr = convert_units_to(high_dtr, "degC")
        dtr = tasmax - tasmin
        tr_adj = 0.25 * xarray.where(
            dtr > high_dtr,
            dtr - high_dtr,
            xarray.where(dtr < low_dtr, dtr - low_dtr, 0),
        )
        if method.lower() == "gladstones":
            lat_mask = abs(lat) <= 50
            k = 1 + xarray.where(lat_mask, max(((abs(lat) - 40) / 10) * 0.06, 0), 0)
            k_aggregated = 1
        else:
            day_length = day_lengths(
                dates=tasmin.time,
                lat=lat,
                start_date=start_date,
                end_date=end_date,
                freq=freq,
            )
            k = 1
            k_huglin = 2.8311e-4 * day_length + 0.30834
            k_aggregated = 1.1135 * k_huglin - 0.1352
    elif method.lower() == "icclim":
        k = 1
        tr_adj = 0
        k_aggregated = 1
    else:
        raise NotImplementedError()

    bedd = ((((tasmin + tasmax) / 2) - thresh_tasmin).clip(min=0) * k + tr_adj).clip(
        max=max_daily_degree_days
    )

    bedd = (
        aggregate_between_dates(bedd, start=start_date, end=end_date, freq=freq)
        * k_aggregated
    )

    bedd.attrs["units"] = "K days"
    return bedd


@declare_units(tasmin="[temperature]", lat="[]")
def cool_night_index(
    tasmin: xarray.DataArray, lat: xarray.DataArray, freq: str = "YS"
) -> xarray.DataArray:
    """Cool Night Index.

    Mean minimum temperature for September (northern hemisphere) or March (Southern hemishere).
    Used in calculating the Géoviticulture Multicriteria Classification System.

    Parameters
    ----------
    tasmin : xarray.DataArray
      Minimum daily temperature.
    lat: xarray.DataArray, optional
      Latitude coordinate.
    freq : str
      Resampling frequency.

    Returns
    -------
    xarray.DataArray, [degC]
      Mean of daily minimum temperature for month of interest.

    Notes
    -----
    Given that this indice only examines September and March months, it possible to send in DataArrays containing only
    these timesteps. Users should be aware that due to the missing values checks in wrapped Indicators, datasets that
    are missing several months will be flagged as invalid. This check can be ignored by setting the following context:

    >>> with xclim.set_options(check_missing='skip', data_validation='log'):
    >>>     cni = xclim.atmos.cool_night_index(...)  # xdoctest: +SKIP

    References
    ----------
    Indice originally published in Tonietto, J., & Carbonneau, A. (2004). A multicriteria climatic classification system
    or grape-growing regions worldwide. Agricultural and Forest Meteorology, 124(1–2), 81‑97.
    https://doi.org/10.1016/j.agrformet.2003.06.001
    """
    tasmin = convert_units_to(tasmin, "degC")

    # Use September in northern hemisphere, March in southern hemisphere.
    months = tasmin.time.dt.month
    month = xarray.where(lat > 0, 9, 3)
    tasmin = tasmin.where(months == month, drop=True)

    cni = tasmin.resample(time=freq).mean()
    cni.attrs["units"] = "degC"
    return cni


@declare_units(tas="[temperature]", lat="[]")
def latitude_temperature_index(
    tas: xarray.DataArray,
    lat: xarray.DataArray,
    lat_factor: float = 75,
    freq: str = "YS",
) -> xarray.DataArray:
    """Latitude-Temperature Index.

    Mean temperature of the warmest month with a latitude-based scaling factor.
    Used for categorizing winegrowing regions.

    Parameters
    ----------
    tas: xarray.DataArray
      Mean daily temperature.
    lat: xarray.DataArray
      Latitude coordinate.
    lat_factor: float
      Latitude factor. Maximum poleward latitude. Default: 75.
    freq : str
      Resampling frequency.

    Returns
    -------
    xarray.DataArray, [unitless]
      Latitude Temperature Index.

    Notes
    -----
    The latitude factor of `75` is provided for examining the poleward expansion of winegrowing climates under scenarios
    of climate change. For comparing 20th century/observed historical records, the original scale factor of `60` is more
    appropriate.

    Let :math:`Tn_{j}` be the average temperature for a given month :math:`j`, :math:`lat_{f}` be the latitude factor,
    and :math:`lat` be the latitude of the area of interest. Then the Latitude-Temperature Index (:math:`LTI`) is:

    .. math::
        LTI = max(TN_{j}: j = 1..12)(lat_f - |lat|)

    References
    ----------
    Indice originally published in Jackson, D. I., & Cherry, N. J. (1988). Prediction of a District’s Grape-Ripening
    Capacity Using a Latitude-Temperature Index (LTI). American Journal of Enology and Viticulture, 39(1), 19‑28.

    Modified latitude factor from Kenny, G. J., & Shao, J. (1992). An assessment of a latitude-temperature index for
    predicting climate suitability for grapes in Europe. Journal of Horticultural Science, 67(2), 239‑246.
    https://doi.org/10.1080/00221589.1992.11516243
    """
    tas = convert_units_to(tas, "degC")

    tas = tas.resample(time="MS").mean(dim="time")
    mtwm = tas.resample(time=freq).max(dim="time")

    lat_mask = (abs(lat) >= 0) & (abs(lat) <= lat_factor)
    lat_coeff = xarray.where(lat_mask, lat_factor - abs(lat), 0)

    lti = mtwm * lat_coeff
    lti.attrs["units"] = ""
    return lti


@declare_units(
    pr="[precipitation]",
    tasmin="[temperature]",
    tasmax="[temperature]",
    tas="[temperature]",
)
def water_budget(
    pr: xarray.DataArray,
    tasmin: Optional[xarray.DataArray] = None,
    tasmax: Optional[xarray.DataArray] = None,
    tas: Optional[xarray.DataArray] = None,
    method: str = "BR65",
) -> xarray.DataArray:
    r"""Precipitation minus potential evapotranspiration.

    Precipitation minus potential evapotranspiration as a measure of an approximated surface water budget,
    where the potential evapotranspiration is calculated with a given method.

    Parameters
    ----------
    pr : xarray.DataArray
      Daily precipitation.
    tasmin : xarray.DataArray
      Minimum daily temperature.
    tasmax : xarray.DataArray
      Maximum daily temperature.
    tas : xarray.DataArray
      Mean daily temperature.
    method : str
      Method to use to calculate the potential evapotranspiration.

    Notes
    -----
    Available methods are listed in the description of xclim.indicators.atmos.potential_evapotranspiration.

    Returns
    -------
    xarray.DataArray,
      Precipitation minus potential evapotranspiration.
    """
    pr = convert_units_to(pr, "kg m-2 s-1")

    pet = xci.potential_evapotranspiration(
        tasmin=tasmin, tasmax=tasmax, tas=tas, method=method
    )

    if xarray.infer_freq(pet.time) == "MS":
        with xarray.set_options(keep_attrs=True):
            pr = pr.resample(time="MS").mean(dim="time")

    out = pr - pet

    out.attrs["units"] = pr.attrs["units"]
    return out


@declare_units(pr="[precipitation]", thresh="[length]")
def dry_spell_frequency(
    pr: xarray.DataArray,
    thresh: str = "1.0 mm",
    window: int = 3,
    freq: str = "YS",
    op: str = "sum",
) -> xarray.DataArray:
    """
    Return the number of dry periods of n days and more, during which the accumulated or maximal daily precipitation
    amount on a window of n days is under the threshold.

    Parameters
    ----------
    pr : xarray.DataArray
      Daily precipitation.
    thresh : str
      Precipitation amount under which a period is considered dry.
      The value against which the threshold is compared depends on  `op` .
    window : int
      Minimum length of the spells.
    freq : str
      Resampling frequency.
    op: {"sum","max"}
      Operation to perform on the window.
      Default is "sum", which checks that the sum of accumulated precipitation over the whole window is less than the
      threshold.
      "max" checks that the maximal daily precipitation amount within the window is less than the threshold.
      This is the same as verifying that each individual day is below the threshold.

    Returns
    -------
    xarray.DataArray
      The {freq} number of dry periods of minimum {window} days.

    Examples
    --------
    >>> pr = xr.open_dataset(path_to_pr_file).pr
    >>> dry_spell_frequency(pr=pr, op="sum")
    >>> dry_spell_frequency(pr=pr, op="max")
    """
    pram = rate2amount(pr, out_units="mm")
    thresh = convert_units_to(thresh, pram)

    agg_pr = getattr(pram.rolling(time=window, center=True), op)()
    out = (
        (agg_pr < thresh)
        .resample(time=freq)
        .map(rl.windowed_run_events, window=1, dim="time")
    )

    out.attrs["units"] = ""
    return out


@declare_units(pr="[precipitation]", thresh="[length]")
def dry_spell_total_length(
    pr: xarray.DataArray,
    thresh: str = "1.0 mm",
    window: int = 3,
    op: str = "sum",
    freq: str = "YS",
    **indexer,
) -> xarray.DataArray:
    """
    Total length of dry spells

    Total number of days in dry periods of a minimum length, during which the maximum or
    accumulated precipitation within a window of the same length is under a threshold.

    Parameters
    ----------
    pr : xarray.DataArray
      Daily precipitation.
    thresh : str
      Accumulated precipitation value under which a period is considered dry.
    window : int
      Number of days where the maximum or accumulated precipitation is under threshold.
    op : {"max", "sum"}
      Reduce operation.
    freq : str
      Resampling frequency.
    indexer :
      Indexing parameters to compute the indicator on a temporal subset of the data.
      It accepts the same arguments as :py:func:`xclim.indices.generic.select_time`.
      Indexing is done after finding the dry days, but before finding the spells.

    Returns
    -------
    xarray.DataArray
      The {freq} total number of days in dry periods of minimum {window} days.

    Notes
    -----
    The algorithm assumes days before and after the timeseries are "wet", meaning that
    the condition for being considered part of a dry spell is stricter on the edges. For
    example, with `window=3` and `op='sum'`, the first day of the series is considered
    part of a dry spell only if the accumulated precipitation within the first 3 days is
    under the threshold. In comparison, a day in the middle of the series is considered
    part of a dry spell if any of the three 3-day periods of which it is part are
    considered dry (so a total of five days are included in the computation, compared to only 3.)
    """
    pram = rate2amount(pr, out_units="mm")
    thresh = convert_units_to(thresh, pram)

    pram_pad = pram.pad(time=(0, window))
    mask = getattr(pram_pad.rolling(time=window), op)() < thresh
    dry = (mask.rolling(time=window).sum() >= 1).shift(time=-(window - 1))
    dry = dry.isel(time=slice(0, pram.time.size)).astype(float)

    out = select_time(dry, **indexer).resample(time=freq).sum("time")
    return to_agg_units(out, pram, "count")


@declare_units(tas="[temperature]")
def qian_weighted_mean_average(
    tas: xarray.DataArray, dim: str = "time"
) -> xarray.DataArray:
    r"""Binomial smoothed, five-day weighted mean average temperature.

    Calculates a five-day weighted moving average with emphasis on temperatures closer to day of interest.

    Parameters
    ----------
    tas: xarray.DataArray
      Daily mean temperature.
    dim: str
      Time dimension.

    Returns
    -------
    xarray.DataArray
      Binomial smoothed, five-day weighted mean average temperature.

    Notes
    -----
    Let :math:`X_{n}` be the average temperature for day :math:`n` and :math:`X_{t}` be the daily mean temperature
    on day :math:`t`. Then the weighted mean average can be calculated as follows:

    .. math::
        \overline{X}_{n} = \frac{X_{n-2} + 4X_{n-1} + 6X_{n} + 4X_{n+1} + X_{n+2}}{16}

    References
    ----------
    Indice oririginally published in Qian, B., Zhang, X., Chen, K., Feng, Y., & O’Brien, T. (2009). Observed Long-Term
    Trends for Agroclimatic Conditions in Canada. Journal of Applied Meteorology and Climatology, 49(4), 604‑618.
    https://doi.org/10.1175/2009JAMC2275.1

    Inspired by Bootsma, A., & Gameda and D.W. McKenney, S. (2005). Impacts of potential climate change on selected
    agroclimatic indices in Atlantic Canada. Canadian Journal of Soil Science, 85(2), 329‑343.
    https://doi.org/10.4141/S04-019
    """
    units = tas.attrs["units"]

    weights = xarray.DataArray([0.0625, 0.25, 0.375, 0.25, 0.0625], dims=["window"])
    weighted_mean = tas.rolling({dim: 5}, center=True).construct("window").dot(weights)

    weighted_mean.attrs["units"] = units
    return weighted_mean


@declare_units(
    tasmax="[temperature]",
    tasmin="[temperature]",
    thresh="[temperature]",
)
def effective_growing_degree_days(
    tasmax: xarray.DataArray,
    tasmin: xarray.DataArray,
    *,
    thresh: str = "5 degC",
    method: str = "bootsma",
    after_date: DayOfYearStr = "07-01",
    dim: str = "time",
    freq: str = "YS",
) -> xarray.DataArray:
    r"""Effective growing degree days.

    Growing degree days based on a dynamic start and end of the growing season.

    Parameters
    ----------
    tasmax: xarray.DataArray
      Daily mean temperature.
    tasmin: xarray.DataArray
      Daily minimum temperature.
    thresh: str
      The minimum temperature threshold.
    method: {"bootsma", "qian"}
      The window method used to determine the temperature-based start date.
      For "bootsma", the start date is defined as 10 days after the average temperature exceeds a threshold (5 degC).
      For "qian", the start date is based on a weighted 5-day rolling average, based on `qian_weighted_mean_average()`.
    after_date : str
      Date of the year after which to look for the first frost event. Should have the format '%m-%d'.
    dim: str
      Time dimension.
    freq : str
      Resampling frequency.

    Returns
    -------
    xarray.DataArray

    Notes
    -----
    The effective growing degree days for a given year :math:`EGDD_i` can be calculated as follows:

    .. math::
        EGDD_i = \sum_{i=\text{j_{start}}^{\text{j_{end}}} max\left(TG - Thresh, 0 \right)

    Where :math:`TG` is the mean daly temperature, and :math:`j_{start}` and :math:`j_{end}` are the start and end dates
    of the growing season. The growing season start date methodology is determined via the `method` flag.
    For "bootsma", the start date is defined as 10 days after the average temperature exceeds a threshold (5 degC).
    For "qian", the start date is based on a weighted 5-day rolling average, based on `qian_weighted_mean_average()`.

    The end date is determined as the day preceding the first day with minimum temperature below 0 degC.

    References
    ----------
    Indice originally published in Bootsma, A., & Gameda and D.W. McKenney, S. (2005). Impacts of potential climate
    change on selected agroclimatic indices in Atlantic Canada. Canadian Journal of Soil Science, 85(2), 329‑343.
    https://doi.org/10.4141/S04-019
    """
    tasmax = convert_units_to(tasmax, "degC")
    tasmin = convert_units_to(tasmin, "degC")
    thresh = convert_units_to(thresh, "degC")

    tas = (tasmin + tasmax) / 2
    tas.attrs["units"] = "degC"

    if method.lower() == "bootsma":
        fda = first_day_above(tasmin=tas, thresh="5.0 degC", window=1, freq=freq)
        start = fda + 10
    elif method.lower() == "qian":
        tas_weighted = qian_weighted_mean_average(tas=tas, dim=dim)
        start = freshet_start(tas_weighted, thresh=thresh, window=5, freq=freq)
    else:
        raise NotImplementedError(f"Method: {method}.")

    # The day before the first day below 0 degC
    end = (
        first_day_below(
            tasmin=tasmin,
            thresh="0 degC",
            after_date=after_date,
            window=1,
            freq=freq,
        )
        - 1
    )

    deg_days = (tas - thresh).clip(min=0)
    egdd = aggregate_between_dates(deg_days, start=start, end=end, freq=freq)

    return to_agg_units(egdd, tas, op="delta_prod")


@declare_units(
    pr="[precipitation]",
    etp="[evapotranspiration]",
    s_thresh_wet="[length]",
    s_thresh_dry="[length]",
    e_thresh="[length]",
)
def rain_season(
    pr: xarray.DataArray,
    etp: xarray.DataArray = None,
    start_next: xarray.DataArray = None,
    s_thresh_wet: str = "25.0 mm",
    s_window_wet: int = 3,
    s_thresh_dry: str = "1.0 mm",
    s_dry_days: int = 7,
    s_window_dry: int = 30,
    s_start_date: str = "",
    s_end_date: str = "",
    e_op: str = "max",
    e_thresh: str = "5.0 mm",
    e_window: int = 20,
    e_etp_rate: str = "",
    e_start_date: str = "",
    e_end_date: str = "",
) -> Tuple[xarray.DataArray, xarray.DataArray, xarray.DataArray, xarray.DataArray]:

    """
    Calculate rain season start, end, length and accumulated precipitation.

    Parameters
    ----------
    pr : xarray.DataArray
        Daily precipitation.
    etp : xarray.DataArray
        Daily evapotranspiration.
    start_next : xarray.DataArray
        First day of the next rain season.
    s_thresh_wet : str
        Accumulated precipitation threshold associated with {s_window_wet}.
    s_window_wet: int
        Number of days where accumulated precipitation is above {s_thresh_wet}.
    s_thresh_dry: str
        Daily precipitation threshold associated with {s_window_dry].
    s_dry_days: int
        Maximum number of dry days in {s_window_tot}.
    s_window_dry: int
        Number of days, after {s_window_wet}, during which daily precipitation is not greater than or equal to
        {s_thresh_dry} for {s_dry_days} consecutive days.
    s_start_date: str
        First day of year where season can start ("mm-dd").
    s_end_date: str
        Last day of year where season can start ("mm-dd").
    e_op : str
        Resampling operator = {"max", "sum", "etp}
        If "max": based on the occurrence (or not) of an event during the last days of a rain season.
            The rain season stops when no daily precipitation greater than {e_thresh} have occurred over a period of
            {e_window} days.
        If "sum": based on a total amount of precipitation received during the last days of the rain season.
            The rain season stops when the total amount of precipitation is less than {e_thresh} over a period of
            {e_window} days.
        If "etp": calculation is based on the period required for a water column of height {e_thresh] to evaporate,
            considering that any amount of precipitation received during that period must evaporate as well. If {etp} is
            not available, the evapotranspiration rate is assumed to be {e_etp_rate}.
    e_thresh : str
        Maximum or accumulated precipitation threshold associated with {e_window}.
        If {e_op} == "max": maximum daily precipitation  during a period of {e_window} days.
        If {e_op} == "sum": accumulated precipitation over {e_window} days.
        If {e_op} == "etp": height of water column that must evaporate.
    e_window: int
        If {e_op} in ["max", "sum"]: number of days used to verify if the rain season is ending.
    e_etp_rate: str
        If {e_op} == "etp": evapotranspiration rate.
        Otherwise: not used.
    e_start_date: str
        First day of year at or after which the season can end ("mm-dd").
    e_end_date: str
        Last day of year at or before which the season can end ("mm-dd").
    """

    def rename_dimensions(
        da: xarray.DataArray, lat_name: str = "latitude", lon_name: str = "longitude"
    ) -> xarray.DataArray:

        if ("location" not in da.dims) and (
            (lat_name not in da.dims) or (lon_name not in da.dims)
        ):
            if "dim_0" in list(da.dims):
                da = da.rename({"dim_0": "time"})
                da = da.rename({"dim_1": lat_name, "dim_2": lon_name})
            elif ("lat" in list(da.dims)) or ("lon" in list(da.dims)):
                da = da.rename({"lat": lat_name, "lon": lon_name})
            elif ("rlat" in list(da.dims)) or ("rlon" in list(da.dims)):
                da = da.rename({"rlat": lat_name, "rlon": lon_name})
            elif (lat_name not in list(da.dims)) and (lon_name not in list(da.dims)):
                if lat_name == "latitude":
                    da = da.expand_dims(latitude=1)
                if lon_name == "longitude":
                    da = da.expand_dims(longitude=1)
        return da

    # Rename dimensions.
    pr = rename_dimensions(pr)
    if etp is not None:
        etp = rename_dimensions(etp)

    # Calculate rain season start.
    start = xarray.DataArray(
        rain_season_start(
            pr,
            s_thresh_wet,
            s_window_wet,
            s_thresh_dry,
            s_dry_days,
            s_window_dry,
            s_start_date,
            s_end_date,
        )
    )

    # Calculate rain season end.
    end = xarray.DataArray(
        rain_season_end(
            pr,
            etp,
            start,
            start_next,
            e_op,
            e_thresh,
            e_window,
            e_etp_rate,
            e_start_date,
            e_end_date,
        )
    )

    # Calculate rain season length.
    length = xarray.DataArray(rain_season_length(start, end))

    # Calculate rain quantity.
    prcptot = xarray.DataArray(rain_season_prcptot(pr, start, end))

    return start, end, length, prcptot


@declare_units(pr="[precipitation]", thresh_wet="[length]", thresh_dry="[length]")
def rain_season_start(
    pr: xarray.DataArray,
    thresh_wet: str = "25.0 mm",
    window_wet: int = 3,
    thresh_dry: str = "1.0 mm",
    dry_days: int = 7,
    window_dry: int = 30,
    start_date: str = "",
    end_date: str = "",
) -> xarray.DataArray:

    """
    Detect the first day of the rain season.

    Rain season starts on the first day of a sequence of {window_wet} days with accumulated precipitation greater than
    or equal to {thresh_wet} that is followed by a period of {window_dry} days with fewer than {dry_days} consecutive
    days with less than {thresh_dry} daily precipitation. The search is constrained by {start_date} and {end_date}."

    Parameters
    ----------
    pr : xarray.DataArray
        Precipitation data.
    thresh_wet : str
        Accumulated precipitation threshold associated with {window_wet}.
    window_wet: int
        Number of days where accumulated precipitation is above {thresh_wet}.
    thresh_dry: str
        Daily precipitation threshold associated with {window_dry}.
    dry_days: int
        Maximum number of dry days in {window_dry}.
    window_dry: int
        Number of days, after {window_wet}, during which daily precipitation is not greater than or equal to
        {thresh_dry} for {dry_days} consecutive days.
    start_date: str
        First day of year where season can start ("mm-dd").
    end_date: str
        Last day of year where season can start ("mm-dd").

    Returns
    -------
    xarray.DataArray, [dimensionless]
        Rain season start (day of year).

    Examples
    --------
    Successful season start:
        . . . . 10 10 10 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 . . .
                 ^
    False start:
        . . . . 10 10 10 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 . . .
    Not even a start:
        . . . .  8  8  8 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 . . .
    given the numbers correspond to daily precipitation, based on default parameter values.

    References
    ----------
    This index was suggested by:
    Sivakumar, M.V.K. (1988). Predicting rainy season potential from the onset of rains in Southern Sahelian and
    Sudanian climatic zones of West Africa. Agricultural and Forest Meteorology, 42(4): 295-305.
    https://doi.org/10.1016/0168-1923(88)90039-1
    and by:
    Dodd, D.E.S. & Jolliffe, I.T. (2001) Early detection of the start of the wet season in semiarid tropical climates of
    Western Africa. Int. J. Climatol., 21, 1251‑1262. https://doi.org/10.1002/joc.640
    This correspond to definition no. 2, which is a simplification of an index mentioned in:
    Jolliffe, I.T. & Sarria-Dodd, D.E. (1994) Early detection of the start of the wet season in tropical climates. Int.
    J. Climatol., 14: 71-76. https://doi.org/10.1002/joc.3370140106
    which is based on:
    Stern, R.D., Dennett, M.D., & Garbutt, D.J. (1981) The start of the rains in West Africa. J. Climatol., 1: 59-68.
    https://doi.org/10.1002/joc.3370010107
    """

    # Unit conversion.
    pram = rate2amount(pr, out_units="mm")
    thresh_wet = convert_units_to(thresh_wet, pram)
    thresh_dry = convert_units_to(thresh_dry, pram)

    # Eliminate negative values.
    pram = xarray.where(pram < 0, 0, pram)
    pram.attrs["units"] = "mm"

    # Assign search boundaries.
    start_doy = 1
    if start_date != "":
        start_doy = datetime.datetime.strptime(start_date, "%m-%d").timetuple().tm_yday
    end_doy = 365
    if end_date != "":
        end_doy = datetime.datetime.strptime(end_date, "%m-%d").timetuple().tm_yday
    if (start_date == "") and (end_date != ""):
        start_doy = 1 if end_doy == 365 else end_doy + 1
    elif (start_date != "") and (end_date == ""):
        end_doy = 365 if start_doy == 1 else start_doy - 1

    # Flag the first day of each sequence of {window_wet} days with a total of {thresh_wet} in precipitation
    # (assign True).
    wet = xarray.DataArray(pram.rolling(time=window_wet).sum() >= thresh_wet).shift(
        time=-(window_wet - 1), fill_value=False
    )

    # Identify dry days (assign 1).
    dry_day = xarray.where(pram < thresh_dry, 1, 0)

    # Identify each day that is not followed by a sequence of {window_dry} days within a period of {window_tot} days,
    # starting after {window_wet} days (assign True).
    dry_seq = None
    for i in range(window_dry - dry_days - 1):
        dry_day_i = dry_day.shift(time=-(i + window_wet), fill_value=False)
        dry_seq_i = xarray.DataArray(
            dry_day_i.rolling(time=dry_days).sum() >= dry_days
        ).shift(time=-(dry_days - 1), fill_value=False)
        if i == 0:
            dry_seq = dry_seq_i.copy()
        else:
            dry_seq = dry_seq | dry_seq_i
    no_dry_seq = dry_seq.astype(bool) == 0

    # Flag days between {start_date} and {end_date} (or the opposite).
    if end_doy >= start_doy:
        doy = (pram.time.dt.dayofyear >= start_doy) & (
            pram.time.dt.dayofyear <= end_doy
        )
    else:
        doy = (pram.time.dt.dayofyear <= end_doy) | (
            pram.time.dt.dayofyear >= start_doy
        )

    # Obtain the first day of each year where conditions apply.
    start = (
        (wet & no_dry_seq & doy)
        .resample(time="YS")
        .map(rl.first_run, window=1, dim="time", coord="dayofyear")
    )
    start = xarray.where((start < 1) | (start > 365), np.nan, start)
    start.attrs["units"] = "1"

    return start


@declare_units(
    pr="[precipitation]",
    etp="[evapotranspiration]",
    thresh="[length]",
    etp_rate="[length]",
)
def rain_season_end(
    pr: xarray.DataArray,
    etp: xarray.DataArray = None,
    start: xarray.DataArray = None,
    start_next: xarray.DataArray = None,
    op: str = "max",
    thresh: str = "5.0 mm",
    window: int = 20,
    etp_rate: str = "0.0 mm",
    start_date: str = "",
    end_date: str = "",
) -> xarray.DataArray:

    """
    Detect the last day of the rain season.

    Three methods are available:
    - If {op}=="max", season ends when no daily precipitation is greater than {thresh} over a period of {window} days.
    - If {op}=="sum", season ends when cumulated precipitation over a period of {window} days is smaller than {thresh}.
    - If {op}=="etp", season ends after a water column of height {thresh} has evaporated at daily rate specified in
      {etp} or {etp_rate}, considering that the cumulated precipitation during this period must also evaporate.
    Search is constrained by {start_date} and {end_date}.

    Parameters
    ----------
    pr : xarray.DataArray
        Daily precipitation.
    etp : xarray.DataArray
        Daily evapotranspiration.
    start : xarray.DataArray
        First day of the current rain season.
    start_next : xarray.DataArray
        First day of the next rain season.
    op : str
        Resampling operator = {"max", "sum", "etp}
        If "max": based on the occurrence (or not) of an event during the last days of a rain season.
            The rain season stops when no daily precipitation greater than {thresh} have occurred over a period of
            {window} days.
        If "sum": based on a total amount of precipitation received during the last days of the rain season.
            The rain season stops when the total amount of precipitation is less than {thresh} over a period of
            {window} days.
        If "etp": calculation is based on the period required for a water column of height {thresh] to evaporate,
            considering that any amount of precipitation received during that period must evaporate as well. If {etp} is
            not available, the evapotranspiration rate is assumed to be {etp_rate}.
    thresh : str
        Maximum or accumulated precipitation threshold associated with {window}.
        If {op} == "max": maximum daily precipitation  during a period of {window} days.
        If {op} == "sum": accumulated precipitation over {window} days.
        If {op} == "etp": height of water column that must evaporate.
    window: int
        If {op} in ["max", "sum"]: number of days used to verify if the rain season is ending.
    etp_rate: str
        If {op} == "etp": evapotranspiration rate.
        Otherwise: not used.
    start_date: str
        First day of year at or after which the season can end ("mm-dd").
    end_date: str
        Last day of year at or before which the season can end ("mm-dd").

    Returns
    -------
    xarray.DataArray, [dimensionless]
        Rain season end (day of year).

    Examples
    --------
    Successful season end with {op} == "max":
        . . . . 5 0 1 0 1 0 1 0 1 0 1 0 1 0 4 0 1 0 1 0 1 . . . (pr)
                ^
    Successful season end with {op} == "sum":
        . 5 5 5 5 0 1 0 2 0 3 0 4 0 3 0 2 0 1 0 1 0 1 0 1 . . . (pr)
                ^
    Successful season end with {op} == "etp":
        . 5 5 5 5 3 3 3 3 5 0 0 1 1 0 5 . . (pr)
        . 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 . . (etp_rate)
                                  ^
    given the numbers correspond to daily precipitation or evapotranspiration, based on default parameter values.

    References
    ----------
    The algorithm corresponding to {op} = "max", referred to as the agronomic criterion, is suggested by:
    Somé, L. & Sivakumar, M.V.k. (1994). Analyse de la longueur de la saison culturale en fonction de la date de début
    des pluies au Burkina Faso. Compte rendu des travaux no 1: Division du sol et Agroclimatologie. INERA, Burkina Faso,
    43 pp.
    It can be applied to a country such as Ivory Coast, which has a bimodal regime near its coast.
    The algorithm corresponding to {op} = "etp" is applicable to Sahelian countries with a monomodal regime, as
    mentioned by Food Security Cluster (May 23rd, 2016):
    https://fscluster.org/mali/document/les-previsions-climatiques-de-2016-du
    This includes countries such as Burkina Faso, Senegal, Mauritania, Gambia, Guinea and Bissau.
    """

    # Unit conversion.
    pram = rate2amount(pr, out_units="mm")
    etpam = None
    if etp is not None:
        etpam = rate2amount(etp, out_units="mm")
    thresh = convert_units_to(thresh, pram)
    etp_rate = convert_units_to(etp_rate, etpam if etpam is not None else pram)

    # Eliminate negative values.
    pram = xarray.where(pram < 0, 0, pram)
    pram.attrs["units"] = "mm"
    if etpam is not None:
        etpam = xarray.where(etpam < 0, 0, etpam)
        etpam.attrs["units"] = "mm"

    # Assign search boundaries.
    start_doy = 1
    if start_date != "":
        start_doy = datetime.datetime.strptime(start_date, "%m-%d").timetuple().tm_yday
    end_doy = 365
    if end_date != "":
        end_doy = datetime.datetime.strptime(end_date, "%m-%d").timetuple().tm_yday
    if (start_date == "") and (end_date != ""):
        start_doy = 1 if end_doy == 365 else end_doy + 1
    elif (start_date != "") and (end_date == ""):
        end_doy = 365 if start_doy == 1 else start_doy - 1

    # Flag days between {start_date} and {end_date} (or the opposite).
    dayofyear = pram.time.dt.dayofyear.astype(float)
    if end_doy >= start_doy:
        doy = (dayofyear >= start_doy) & (dayofyear <= end_doy)
    else:
        doy = (dayofyear <= end_doy) | (dayofyear >= start_doy)

    end = None

    if op == "etp":

        # Calculate the minimum length of the period.
        window_min = math.ceil(thresh / etp_rate) if (etp_rate > 0) else 0
        window_max = (
            end_doy - start_doy + 1
            if (end_doy >= start_doy)
            else (365 - start_doy + 1 + end_doy)
        ) - window_min
        if etp_rate == 0:
            window_min = window_max

        # Window must be varied until it's size allows for complete evaporation.
        for window_i in list(range(window_min, window_max + 1)):

            # Flag the day before each sequence of {dt} days that results in evaporating a water column, considering
            # precipitation falling during this period (assign 1).
            if etpam is None:
                dry_seq = xarray.DataArray(
                    (pram.rolling(time=window_i).sum() + thresh)
                    <= (window_i * etp_rate)
                )
            else:
                dry_seq = xarray.DataArray(
                    (pram.rolling(time=window_i).sum() + thresh)
                    <= etpam.rolling(time=window_i).sum()
                )

            # Obtain the first day of each year where conditions apply.
            end_i = (
                (dry_seq & doy)
                .resample(time="YS")
                .map(rl.first_run, window=1, dim="time", coord="dayofyear")
            )

            # Update the cells that were not assigned yet.
            if end is None:
                end = end_i.copy()
            else:
                sel = np.isnan(end) & (
                    (np.isnan(end_i).astype(int) == 0) | (end_i < end)
                )
                end = xarray.where(sel, end_i, end)

            # Exit loop if all cells were assigned a value.
            window = window_i
            if np.isnan(end).astype(int).sum() == 0:
                break

    else:

        # Shift datasets to simplify the analysis.
        dt = 0 if end_doy >= start_doy else start_doy - 1
        pram_shift = pram.copy().shift(time=-dt, fill_value=False)
        doy = doy.shift(time=-dt, fill_value=False)

        # Determine if it rains (assign 1) or not (assign 0).
        wet = (
            xarray.where(pram_shift < thresh, 0, 1)
            if op == "max"
            else xarray.where(pram_shift == 0, 0, 1)
        )

        # Flag each day (assign 1) before a sequence of:
        # {window} days with no amount reaching {thresh}:
        if op == "max":
            dry_seq = xarray.DataArray(wet.rolling(time=window).sum() == 0)
        # {window} days with a total amount reaching {thresh}:
        else:
            dry_seq = xarray.DataArray(pram_shift.rolling(time=window).sum() < thresh)
        dry_seq = dry_seq.shift(time=-window, fill_value=False)

        # Obtain the first day of each year where conditions apply.
        end = (
            (dry_seq & doy)
            .resample(time="YS")
            .map(rl.first_run, window=1, dim="time", coord="dayofyear")
        )

        # Shift result to the right.
        end += dt
        if end.max() > 365:
            transfer = xarray.ufuncs.maximum(end - 365, 0).shift(
                time=1, fill_value=np.nan
            )
            end = xarray.where(end > 365, np.nan, end)
            end = xarray.where(np.isnan(transfer).astype(bool) == 0, transfer, end)

        # Rain season can't end on (or after) the first day of the last moving {window}, because we ignore the weather
        # past the end of the dataset.
        end = xarray.where(
            (end > 365 - window) & (end == end[len(end.time) - 1]), np.nan, end
        )

    # Rain season can't end unless the last day is rainy or the window comprises rainy days.
    def rain_near_end(loc: str = "") -> xarray.DataArray:
        if loc == "":
            end_loc = end
            pram_loc = pram
        else:
            end_loc = end[end.location == loc].squeeze()
            pram_loc = pram[pram.location == loc].squeeze()
        n_days = 0
        for t in range(len(end_loc.time)):
            n_days_t = int(
                xarray.DataArray(pram_loc.time.dt.year == end_loc[t].time.dt.year)
                .astype(int)
                .sum()
            )
            if not np.isnan(end_loc[t]):
                pos_end = int(end_loc[t]) + n_days - 1
                if op in ["max", "sum"]:
                    pos_win_1 = pos_end + 1
                    pos_win_2 = min(pos_win_1 + window, n_days_t + n_days)
                else:
                    pos_win_2 = pos_end + 1
                    pos_win_1 = max(0, pos_end - window)
                pos_range = [min(pos_win_1, pos_win_2), max(pos_win_1, pos_win_2)]
                if not (
                    (
                        pram_loc.isel(time=slice(pos_range[0], pos_range[1])).sum(
                            dim="time"
                        )
                        > 0
                    )
                    or (pram_loc.isel(time=pos_end) > 0)
                ):
                    end_loc[t] = np.nan
            n_days += n_days_t
        return end_loc

    if "location" not in pram.dims:
        end = rain_near_end()
    else:
        locations = list(pram.location.values)
        for i in range(len(locations)):
            end[end.location == locations[i]] = rain_near_end(locations[i])

    # Adjust or discard rain end values that are not compatible with the current or next season start values.
    # If the season ends before or on start day, discard rain end.
    if start is not None:
        sel = (
            (np.isnan(start).astype(int) == 0)
            & (np.isnan(end).astype(int) == 0)
            & (end <= start)
        )
        end = xarray.where(sel, np.nan, end)

    # If the season ends after or on start day of the next season, the end day of the current season becomes the day
    # before the next season.
    if start_next is not None:
        sel = (
            (np.isnan(start_next).astype(int) == 0)
            & (np.isnan(end).astype(int) == 0)
            & (end >= start_next)
        )
        end = xarray.where(sel, start_next - 1, end)
        end = xarray.where(end < 1, 365, end)
    end.attrs["units"] = "1"

    return end


def rain_season_length(
    start: xarray.DataArray, end: xarray.DataArray
) -> xarray.DataArray:

    """
    Determine the length of the rain season.

    Parameters
    ----------
    start : xarray.DataArray
        Rain season start (first day of year).
    end: xarray.DataArray
        Rain season end (last day of year).

    Returns
    -------
    xarray.DataArray, [dimensionless]
        Rain season length (days/freq).
    """

    # Start and end dates in the same calendar year.
    if start.mean() <= end.mean():
        length = end - start + 1

    # Start and end dates not in the same year (left shift required).
    else:
        length = (
            xarray.DataArray(xarray.ones_like(start) * 365)
            - start
            + end.shift(time=-1, fill_value=np.nan)
            + 1
        )

    # Eliminate negative values. This is a safety measure as this should not happen.
    length = xarray.where(length < 0, 0, length)
    length.attrs["units"] = "days"

    return length


@declare_units(pr="[precipitation]")
def rain_season_prcptot(
    pr: xarray.DataArray, start: xarray.DataArray, end: xarray.DataArray
) -> xarray.DataArray:

    """
    Determine precipitation amount during rain season.

    Parameters
    ----------
    pr : xarray.DataArray
        Daily precipitation.
    start : xarray.DataArray
        Rain season start (first day of year).
    end: xarray.DataArray
        Rain season end (last day of year).

    Returns
    -------
    xarray.DataArray
        Rain season accumulated precipitation (mm/year).
    """

    # Unit conversion.
    pram = rate2amount(pr, out_units="mm")

    # Initialize the array that will contain results.
    prcptot = xarray.zeros_like(start) * np.nan

    # Calculate the sum between two dates for a given year.
    def calc_sum(year: int, start_doy: int, end_doy: int):
        sel = (
            (pram.time.dt.year == year)
            & (pram.time.dt.dayofyear >= start_doy)
            & (pram.time.dt.dayofyear <= end_doy)
        )
        return xarray.where(sel, pram, 0).sum()

    # Calculate the index.
    def calc_idx(loc: str = ""):
        if loc == "":
            prcptot_loc = prcptot
            start_loc = start
            end_loc = end
        else:
            prcptot_loc = prcptot[prcptot.location == loc].squeeze()
            start_loc = start[start.location == loc].squeeze()
            end_loc = end[end.location == loc].squeeze()

        end_shift = None
        for t in range(len(start.time.dt.year)):
            year = int(start.time.dt.year[t])

            # Start and end dates in the same calendar year.
            if start_loc.mean() <= end_loc.mean():
                if (np.isnan(start_loc[t]).astype(bool) == 0) and (
                    np.isnan(end_loc[t]).astype(bool) == 0
                ):
                    prcptot_loc[t] = calc_sum(year, int(start_loc[t]), int(end_loc[t]))

            # Start and end dates not in the same year (left shift required).
            else:
                end_shift = (
                    end_loc.shift(time=-1, fill_value=np.nan) if t == 0 else end_shift
                )
                if (np.isnan(start_loc[t]).astype(bool) == 0) and (
                    np.isnan(end_shift[t]).astype(bool) == 0
                ):
                    prcptot_loc[t] = calc_sum(year, int(start_loc[t]), 365) + calc_sum(
                        year, 1, int(end_shift[t])
                    )

        return prcptot_loc

    if "location" not in pram.dims:
        prcptot = calc_idx()
    else:
        locations = list(pram.location.values)
        for i in range(len(locations)):
            prcptot[prcptot.location == locations[i]] = calc_idx(locations[i])

    prcptot = prcptot // 1
    prcptot.attrs["units"] = "mm"

    return prcptot

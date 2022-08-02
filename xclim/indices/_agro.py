# noqa: D100
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import xarray

import xclim.indices as xci
import xclim.indices.run_length as rl
from xclim.core.calendar import parse_offset, resample_doy, select_time
from xclim.core.units import convert_units_to, declare_units, rate2amount, to_agg_units
from xclim.core.utils import DayOfYearStr, uses_dask
from xclim.indices._threshold import first_day_above, first_day_below, freshet_start
from xclim.indices.generic import aggregate_between_dates
from xclim.indices.helpers import day_lengths
from xclim.indices.stats import dist_method, fit

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
    "standardized_precipitation_index",
    "standardized_precipitation_evapotranspiration_index",
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
    Formula adapted from [BootsmaTremblay&Filion1999]_.

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
    Formula used in calculating the Corn Heat Units for the Agroclimatic Atlas of Quebec [Audet&al2012]_.

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
    .. [BootsmaTremblay&Filion1999] Bootsma, A., G. Tremblay et P. Filion. 1999: Analyse sur les risques associés aux unités thermiques disponibles pour la production de maïs et de soya au Québec. Centre de recherches de l’Est sur les céréales et oléagineux, Ottawa, 28 p.
    .. [Audet&al2012] Audet, R., Côté, H., Bachand, D. and Mailhot, A., 2012: Atlas agroclimatique du Québec. Évaluation des opportunités et des risques agroclimatiques dans un climat en évolution. https://espace.inrs.ca/id/eprint/2406
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
    (Northern Hemisphere; October to March in Southern Hemisphere). Originally proposed in [Huglin1978]_.
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
    `method="jones"`. See: :py:func:`xclim.indices.generic.day_lengths` or [Hall&Jones2010]_ for more information.

    References
    ----------
    .. [Huglin1978] Huglin, P. (1978). Nouveau mode d’évaluation des possibilités héliothermiques d’un milieu viticole. Dans Symposium International sur l’Écologie de la Vigne (p. 89‑98). Ministère de l’Agriculture et de l’Industrie Alimentaire.
    .. [Hall&Jones2010] Hall, A., & Jones, G. V. (2010). Spatial analysis of climate in winegrape-growing regions in Australia. Australian Journal of Grape and Wine Research, 16(3), 389‑404. https://doi.org/10.1111/j.1755-0238.2010.00100.x
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
        day_length = aggregate_between_dates(
            day_lengths(dates=tas.time, lat=lat, method="simple"),
            start=start_date,
            end=end_date,
            op="sum",
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
    lat: xarray.DataArray | None = None,
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
    occur. Indice originally published in [Gladstones1992]_.

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
    different end date (30 September) ([ECAD]_). The simplified formula is as follows:

    .. math::
        BEDD_i = \sum_{i=\text{April 1}}^{\text{September 30}} min\left( max\left(\frac{TX_i  + TN_i)}{2} - 10, 0\right), degdays_{max}\right)

    References
    ----------
    .. [Gladstones1992] Gladstones, J.S. (1992). Viticulture and environment: a study of the effects of environment on grapegrowing and wine qualities, with emphasis on present and future areas for growing winegrapes in Australia. Adelaide:  Winetitles.
    .. [ECAD] Project team ECA&D, KNMI (2013). EUMETNET/ECSN optional programme: European Climate Assessment & Dataset (ECA&D) - Algorithm Theoretical Basis Document (ATBD). (KNMI Project number: EPJ029135, v10.7). https://www.ecad.eu/documents/atbd.pdf
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
            if isinstance(lat, (int, float)):
                lat = xarray.DataArray(lat)
            lat_mask = abs(lat) <= 50
            k = 1 + xarray.where(
                lat_mask, ((abs(lat) - 40) * 0.06 / 10).clip(0, None), 0
            )
            k_aggregated = 1
        else:
            day_length = aggregate_between_dates(
                day_lengths(dates=tasmin.time, lat=lat, method="simple"),
                start=start_date,
                end=end_date,
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
    Used in calculating the Géoviticulture Multicriteria Classification System ([Tonietto&Carbonneau2004]_).

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
    Given that this indice only examines September and March months, it is possible to send in DataArrays containing
    only these timesteps. Users should be aware that due to the missing values checks in wrapped Indicators, datasets
    that are missing several months will be flagged as invalid. This check can be ignored by setting the following
    context:

    Examples
    --------
    >>> with xclim.set_options(
    ...     check_missing="skip", data_validation="log"
    ... ):  # doctest: +SKIP
    ...     cni = xclim.atmos.cool_night_index(...)  # doctest: +SKIP
    ... # doctest: +SKIP

    References
    ----------
    .. [Tonietto&Carbonneau2004] Tonietto, J., & Carbonneau, A. (2004). A multicriteria climatic classification system for grape-growing regions worldwide. Agricultural and Forest Meteorology, 124(1–2), 81‑97. https://doi.org/10.1016/j.agrformet.2003.06.001
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

    Mean temperature of the warmest month with a latitude-based scaling factor ([Jackson&Cherry1988]_).
    Used for categorizing wine-growing regions.

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
    The latitude factor of `75` is provided for examining the poleward expansion of wine-growing climates under scenarios
    of climate change (modified from [Kenny&Shao1992]_). For comparing 20th century/observed historical records, the original scale factor of `60` is more
    appropriate.

    Let :math:`Tn_{j}` be the average temperature for a given month :math:`j`, :math:`lat_{f}` be the latitude factor,
    and :math:`lat` be the latitude of the area of interest. Then the Latitude-Temperature Index (:math:`LTI`) is:

    .. math::
        LTI = max(TN_{j}: j = 1..12)(lat_f - |lat|)

    References
    ----------
    .. [Jackson&Cherry1988] Jackson, D. I., & Cherry, N. J. (1988). Prediction of a District’s Grape-Ripening Capacity Using a Latitude-Temperature Index (LTI). American Journal of Enology and Viticulture, 39(1), 19‑28.
    .. [Kenny&Shao1992] Kenny, G. J., & Shao, J. (1992). An assessment of a latitude-temperature index for predicting climate suitability for grapes in Europe. Journal of Horticultural Science, 67(2), 239‑246. https://doi.org/10.1080/00221589.1992.11516243
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
    evspsblpot="[precipitation]",
    tasmin="[temperature]",
    tasmax="[temperature]",
    tas="[temperature]",
    lat="[]",
)
def water_budget(
    pr: xarray.DataArray,
    evspsblpot: xarray.DataArray | None = None,
    tasmin: xarray.DataArray | None = None,
    tasmax: xarray.DataArray | None = None,
    tas: xarray.DataArray | None = None,
    lat: xarray.DataArray | None = None,
    method: str = "BR65",
) -> xarray.DataArray:
    r"""Precipitation minus potential evapotranspiration.

    Precipitation minus potential evapotranspiration as a measure of an approximated surface water budget,
    where the potential evapotranspiration can be calculated with a given method.

    Parameters
    ----------
    pr : xarray.DataArray
      Daily precipitation.
    evspsblpot: xarray.DataArray
      Potential evapotranspiration
    tasmin : xarray.DataArray
      Minimum daily temperature.
    tasmax : xarray.DataArray
      Maximum daily temperature.
    tas : xarray.DataArray
      Mean daily temperature.
    lat : xarray.DataArray
      Latitude, needed if evspsblpot is not given.
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

    if evspsblpot is None:
        pet = xci.potential_evapotranspiration(
            tasmin=tasmin, tasmax=tasmax, tas=tas, lat=lat, method=method
        )
    else:
        pet = convert_units_to(evspsblpot, "kg m-2 s-1")

    if xarray.infer_freq(pet.time) == "MS":
        with xarray.set_options(keep_attrs=True):
            pr = pr.resample(time="MS").mean(dim="time")

    out = pr - pet

    out.attrs["units"] = pr.attrs["units"]
    return out


@declare_units(
    pr="[precipitation]",
    pr_cal="[precipitation]",
)
def standardized_precipitation_index(
    pr: xarray.DataArray,
    pr_cal: xarray.DataArray,
    freq: str = "MS",
    window: int = 1,
    dist: str = "gamma",
    method: str = "APP",
) -> xarray.DataArray:
    r"""Standardized Precipitation Index (SPI).

    Parameters
    ----------
    pr : xarray.DataArray
      Daily precipitation.
    pr_cal : xarray.DataArray
      Daily precipitation used for calibration. Usually this is a temporal subset of `pr` over some reference period.
    freq : str
      Resampling frequency. A monthly or daily frequency is expected.
    window : int
      Averaging window length relative to the resampling frequency. For example, if `freq="MS"`, i.e. a monthly resampling, the window
      is an integer number of months.
    dist : {'gamma'}
      Name of the univariate distribution, only `gamma` is currently implemented
      (see :py:mod:`scipy.stats`).
    method : {'APP', 'ML'}
      Name of the fitting method, such as `ML` (maximum likelihood), `APP` (approximate). The approximate method
      uses a deterministic function that doesn't involve any optimization.

    Returns
    -------
    xarray.DataArray,
      Standardized Precipitation Index.


    Notes
    -----
    The length `N` of the N-month SPI is determined by choosing the `window = N`. Supported statistical distributions are: ["gamma"]


    Example
    -------
    Computing SPI-3 months using a gamma distribution for the fit

    .. code-block:: python

        import xclim.indices as xci
        import xarray as xr

        ds = xr.open_dataset(filename)
        pr = ds.pr
        pr_cal = pr.sel(time=slice(calibration_start_date, calibration_end_date))
        spi_3 = xci.standardized_precipitation_index(
            pr, pr_cal, freq="MS", window=3, dist="gamma", method="ML"
        )


    References
    ----------
    McKee, Thomas B., Nolan J. Doesken, and John Kleist. "The relationship of drought frequency and duration to time scales." In Proceedings of the 8th Conference on Applied Climatology, vol. 17, no. 22, pp. 179-183. 1993.
    """
    # "WPM" method doesn't seem to work for gamma or pearson3
    dist_and_methods = {"gamma": ["ML", "APP"]}
    if dist not in dist_and_methods.keys():
        raise NotImplementedError(f"The distribution `{dist}` is not supported.")
    elif method not in dist_and_methods[dist]:
        raise NotImplementedError(
            f"The method `{method}` is not supported for distribution `{dist}`."
        )

    # calibration period
    cal_period = pr_cal.time[[0, -1]].dt.strftime("%Y-%m-%dT%H:%M:%S").values.tolist()

    # Determine group type
    if freq == "D" or freq is None:
        freq = "D"
        group = "time.dayofyear"
    else:
        _, base, _, _ = parse_offset(freq)
        if base in ["M"]:
            group = "time.month"
        else:
            raise NotImplementedError(f"Resampling frequency `{freq}` not supported.")

    # Resampling precipitations
    if freq != "D":
        pr = pr.resample(time=freq).mean(keep_attrs=True)
        pr_cal = pr_cal.resample(time=freq).mean(keep_attrs=True)

        def needs_rechunking(da):
            if uses_dask(da) and len(da.chunks[da.get_axis_num("time")]) > 1:
                warnings.warn(
                    "The input data is chunked on time dimension and must be fully rechunked to"
                    " run `fit` on groups ."
                    " Beware, this operation can significantly increase the number of tasks dask"
                    " has to handle.",
                    stacklevel=2,
                )
                return True
            else:
                return False

        if needs_rechunking(pr):
            pr = pr.chunk({"time": -1})
        if needs_rechunking(pr_cal):
            pr_cal = pr_cal.chunk({"time": -1})

    # Rolling precipitations
    if window > 1:
        pr = pr.rolling(time=window).mean(skipna=False, keep_attrs=True)
        pr_cal = pr_cal.rolling(time=window).mean(skipna=False, keep_attrs=True)

    # Obtain fitting params and expand along time dimension
    def resample_to_time(da, da_ref):
        if freq == "D":
            da = resample_doy(da, da_ref)
        else:
            da = da.rename(month="time").reindex(time=da_ref.time.dt.month)
            da["time"] = da_ref.time
        return da

    params = pr_cal.groupby(group).map(lambda x: fit(x, dist, method))
    params = resample_to_time(params, pr)

    # ppf to cdf
    # zero-bounded distributions;  'pearson3' will also go in this group once it's implemented
    if dist in ["gamma", "pearson3"]:
        prob_pos = dist_method("cdf", params, pr.where(pr > 0))
        prob_zero = resample_to_time(
            pr.groupby(group).map(
                lambda x: (x == 0).sum("time") / x.notnull().sum("time")
            ),
            pr,
        )
        prob = prob_zero + (1 - prob_zero) * prob_pos

    # Invert to normal distribution with ppf and obtain SPI
    params_norm = xarray.DataArray(
        [0, 1],
        dims=["dparams"],
        coords=dict(dparams=(["loc", "scale"])),
        attrs=dict(scipy_dist="norm"),
    )
    spi = dist_method("ppf", params_norm, prob)
    spi.attrs["units"] = ""
    spi.attrs["calibration_period"] = cal_period

    return spi


@declare_units(
    wb="[precipitation]",
    wb_cal="[precipitation]",
)
def standardized_precipitation_evapotranspiration_index(
    wb: xarray.DataArray,
    wb_cal: xarray.DataArray,
    freq: str = "MS",
    window: int = 1,
    dist: str = "gamma",
    method: str = "APP",
) -> xarray.DataArray:
    r"""Standardized Precipitation Evapotranspiration Index (SPEI).

    Precipitation minus potential evapotranspiration data (PET) fitted to a statistical distribution (dist), transformed to a cdf,  and inverted back to a gaussian normal pdf. The potential evapotranspiration is calculated with a given method (method).

    Parameters
    ----------
    wb : xarray.DataArray
      Daily water budget (pr - pet).
    wb_cal : xarray.DataArray
      Daily water budget used for calibration.
    freq : str
      Resampling frequency. A monthly or daily frequency is expected.
    window : int
      Averaging window length relative to the resampling frequency. For example, if `freq="MS"`, i.e. a monthly resampling, the window
      is an integer number of months.
    dist : {'gamma'}
      Name of the univariate distribution. Only "gamma" is currently implemented. (see :py:mod:`scipy.stats`).
    method : {'APP', 'ML'}
      Name of the fitting method, such as `ML` (maximum likelihood), `APP` (approximate). The approximate method
      uses a deterministic function that doesn't involve any optimization.

    Returns
    -------
    xarray.DataArray,
      Standardized Precipitation Evapotranspiration Index.

    Notes
    -----
    See Standardized Precipitation Index (SPI) for more details on usage.
    """
    # Allowed distributions are constrained by the SPI function
    if dist in ["gamma"]:
        # Distributions bounded by zero: Water budget must be shifted, only positive values
        # are allowed. The offset choice is arbitrary and needs to be revisited.
        # In monocongo, the offset would be 1000/(60*60*24) in [kg m-2 s-1]
        # The choice can lead to differences as big as +/-0.2 in the SPEI.
        # If taken too big, there are problems with the "ML" method  (this should be an
        # issue with the fitting procedure that also needs attention)
        offset = convert_units_to("1e-4 kg m-2 s-1", wb.units)
        # Increase offset if negative values remain
        offset = offset - 2 * min(wb.min(), wb_cal.min(), 0)
        with xarray.set_options(keep_attrs=True):
            wb, wb_cal = wb + offset, wb_cal + offset

    spei = standardized_precipitation_index(wb, wb_cal, freq, window, dist, method)

    return spei


@declare_units(pr="[precipitation]", thresh="[length]")
def dry_spell_frequency(
    pr: xarray.DataArray,
    thresh: str = "1.0 mm",
    window: int = 3,
    freq: str = "YS",
    op: str = "sum",
) -> xarray.DataArray:
    """Return the number of dry periods of n days and more.

    Periods during which the accumulated or maximal daily precipitation amount on a window of n days is under threshold.

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
    """Total length of dry spells.

    Total number of days in dry periods of a minimum length, during which the maximum or
    accumulated precipitation within a window of the same length is under a threshold.

    Parameters
    ----------
    pr : xarray.DataArray
      Daily precipitation.
    thresh : str
      Accumulated precipitation value under which a period is considered dry.
    window : int
      Number of days when the maximum or accumulated precipitation is under threshold.
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
    tas: xr.DataArray
      Daily mean temperature.
    dim: str
      Time dimension.

    Returns
    -------
    xr.DataArray
      Binomial smoothed, five-day weighted mean average temperature.

    Notes
    -----
    Qian Modified Weighted Mean Indice originally proposed in [Qian&al2009]_, based on [BootsmaGameda&McKenney2005]_.

    Let :math:`X_{n}` be the average temperature for day :math:`n` and :math:`X_{t}` be the daily mean temperature
    on day :math:`t`. Then the weighted mean average can be calculated as follows:

    .. math::
        \overline{X}_{n} = \frac{X_{n-2} + 4X_{n-1} + 6X_{n} + 4X_{n+1} + X_{n+2}}{16}

    References
    ----------
    .. [Qian&al2009] Qian, B., Zhang, X., Chen, K., Feng, Y., & O’Brien, T. (2009). Observed Long-Term Trends for Agroclimatic Conditions in Canada. Journal of Applied Meteorology and Climatology, 49(4), 604‑618. https://doi.org/10.1175/2009JAMC2275.1
    .. [BootsmaGameda&McKenney2005] Bootsma, A., & Gameda and D.W. McKenney, S. (2005). Impacts of potential climate change on selected agroclimatic indices in Atlantic Canada. Canadian Journal of Soil Science, 85(2), 329‑343. https://doi.org/10.4141/S04-019
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

    Growing degree days based on a dynamic start and end of the growing season, as defined in [BootsmaGameda&McKenney2005]_.

    Parameters
    ----------
    tasmax: xr.DataArray
      Daily mean temperature.
    tasmin: xr.DataArray
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
    .. [BootsmaGameda&McKenney2005] Bootsma, A., & Gameda and D.W. McKenney, S. (2005). Impacts of potential climate change on selected agroclimatic indices in Atlantic Canada. Canadian Journal of Soil Science, 85(2), 329‑343. https://doi.org/10.4141/S04-019
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

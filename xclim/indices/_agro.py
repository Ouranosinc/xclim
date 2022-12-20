# noqa: D100
from __future__ import annotations

import warnings

import numpy as np
import xarray

import xclim.indices as xci
import xclim.indices.run_length as rl
from xclim.core.calendar import parse_offset, resample_doy, select_time
from xclim.core.units import convert_units_to, declare_units, rate2amount, to_agg_units
from xclim.core.utils import DayOfYearStr, Quantified, uses_dask
from xclim.indices._threshold import (
    first_day_temperature_above,
    first_day_temperature_below,
)
from xclim.indices.generic import aggregate_between_dates
from xclim.indices.helpers import _gather_lat, day_lengths
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
    thresh_tasmin: Quantified = "4.44 degC",
    thresh_tasmax: Quantified = "10 degC",
) -> xarray.DataArray:
    r"""Corn heat units.

    Temperature-based index used to estimate the development of corn crops.
    Formula adapted from :cite:t:`bootsma_risk_1999`.

    Parameters
    ----------
    tasmin : xarray.DataArray
        Minimum daily temperature.
    tasmax : xarray.DataArray
        Maximum daily temperature.
    thresh_tasmin : Quantified
        The minimum temperature threshold needed for corn growth.
    thresh_tasmax : Quantified
        The maximum temperature threshold needed for corn growth.

    Returns
    -------
    xarray.DataArray, [unitless]
        Daily corn heat units.

    Notes
    -----
    Formula used in calculating the Corn Heat Units for the Agroclimatic Atlas of Quebec :cite:p:`audet_atlas_2012`.

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
    :cite:cts:`audet_atlas_2012,bootsma_risk_1999`

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
    lat: xarray.DataArray | None = None,
    thresh: Quantified = "10 degC",
    method: str = "smoothed",
    start_date: DayOfYearStr = "04-01",
    end_date: DayOfYearStr = "10-01",
    freq: str = "YS",
) -> xarray.DataArray:
    r"""Huglin Heliothermal Index.

    Growing-degree days with a base of 10°C and adjusted for latitudes between 40°N and 50°N for April-September
    (Northern Hemisphere; October-March in Southern Hemisphere). Originally proposed in :cite:t:`huglin_nouveau_1978`.
    Used as a heat-summation metric in viticulture agroclimatology.

    Parameters
    ----------
    tas : xarray.DataArray
        Mean daily temperature.
    tasmax : xarray.DataArray
        Maximum daily temperature.
    lat : xarray.DataArray
        Latitude coordinate.
        If None, a CF-conformant "latitude" field must be available within the passed DataArray.
    thresh : Quantified
        The temperature threshold.
    method : {"smoothed", "icclim", "jones"}
        The formula to use for the latitude coefficient calculation.
    start_date : DayOfYearStr
        The hemisphere-based start date to consider (north = April, south = October).
    end_date : DayOfYearStr
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
    `method="jones"`.
    See: :py:func:`xclim.indices.generic.day_lengths` or :cite:t:`hall_spatial_2010` for more information.

    References
    ----------
    :cite:cts:`huglin_nouveau_1978, hall_spatial_2010`

    """
    tas = convert_units_to(tas, "degC")
    tasmax = convert_units_to(tasmax, "degC")
    thresh = convert_units_to(thresh, "degC")

    if lat is None:
        lat = _gather_lat(tas)

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
    thresh_tasmin: Quantified = "10 degC",
    method: str = "gladstones",
    low_dtr: Quantified = "10 degC",
    high_dtr: Quantified = "13 degC",
    max_daily_degree_days: Quantified = "9 degC",
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
    tasmin : xarray.DataArray
        Minimum daily temperature.
    tasmax : xarray.DataArray
        Maximum daily temperature.
    lat : xarray.DataArray, optional
        Latitude coordinate.
        If None and method in ["gladstones", "icclim"],
        a CF-conformant "latitude" field must be available within the passed DataArray.
    thresh_tasmin : Quantified
        The minimum temperature threshold.
    method : {"gladstones", "icclim", "jones"}
        The formula to use for the calculation.
        The "gladstones" integrates a daily temperature range and latitude coefficient. End_date should be "11-01".
        The "icclim" method ignores daily temperature range and latitude coefficient. End date should be "10-01".
        The "jones" method integrates axial tilt, latitude, and day-of-year on coefficient. End_date should be "11-01".
    low_dtr : Quantified
        The lower bound for daily temperature range adjustment (default: 10°C).
    high_dtr : Quantified
        The higher bound for daily temperature range adjustment (default: 13°C).
    max_daily_degree_days : Quantified
        The maximum amount of biologically effective degrees days that can be summed daily.
    start_date : DayOfYearStr
        The hemisphere-based start date to consider (north = April, south = October).
    end_date : DayOfYearStr
        The hemisphere-based start date to consider (north = October, south = April). This date is non-inclusive.
    freq : str
        Resampling frequency (default: "YS"; For Southern Hemisphere, should be "AS-JUL").

    Returns
    -------
    xarray.DataArray, [K days]
        Biologically effective growing degree days (BEDD)

    Warnings
    --------
    Lat coordinate must be provided if method is "gladstones" or "jones".

    Notes
    -----
    The tasmax ceiling of 19°C is assumed to be the max temperature beyond which no further gains from daily temperature
    occur. Indice originally published in :cite:t:`gladstones_viticulture_1992`.

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
    different end date (30 September) :cite:p:`project_team_eca&d_algorithm_2013`.
    The simplified formula is as follows:

    .. math::

        BEDD_i = \sum_{i=\text{April 1}}^{\text{September 30}} min\left( max\left(\frac{TX_i  + TN_i)}{2} - 10, 0\right), degdays_{max}\right)

    References
    ----------
    :cite:cts:`gladstones_viticulture_1992,project_team_eca&d_algorithm_2013`
    """
    tasmin = convert_units_to(tasmin, "degC")
    tasmax = convert_units_to(tasmax, "degC")
    thresh_tasmin = convert_units_to(thresh_tasmin, "degC")
    max_daily_degree_days = convert_units_to(max_daily_degree_days, "degC")

    if method.lower() in ["gladstones", "jones"]:
        low_dtr = convert_units_to(low_dtr, "degC")
        high_dtr = convert_units_to(high_dtr, "degC")
        dtr = tasmax - tasmin
        tr_adj = 0.25 * xarray.where(
            dtr > high_dtr,
            dtr - high_dtr,
            xarray.where(dtr < low_dtr, dtr - low_dtr, 0),
        )

        if lat is None:
            lat = _gather_lat(tasmin)

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


@declare_units(tasmin="[temperature]")
def cool_night_index(
    tasmin: xarray.DataArray,
    lat: xarray.DataArray | str | None = None,
    freq: str = "YS",
) -> xarray.DataArray:
    """Cool Night Index.

    Mean minimum temperature for September (northern hemisphere) or March (Southern hemisphere).
    Used in calculating the Géoviticulture Multicriteria Classification System (:cite:t:`tonietto_multicriteria_2004`).

    Parameters
    ----------
    tasmin : xarray.DataArray
        Minimum daily temperature.
    lat : xarray.DataArray or {"north", "south"}, optional
        Latitude coordinate as an array, float or string.
        If None, a CF-conformant "latitude" field must be available within the passed DataArray.
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
    >>> from xclim.indices import cool_night_index
    >>> tasmin = xr.open_dataset(path_to_tasmin_file).tasmin
    >>> cni = cool_night_index(tasmin)

    References
    ----------
    :cite:cts:`tonietto_multicriteria_2004`
    """
    tasmin = convert_units_to(tasmin, "degC")

    # Use September in northern hemisphere, March in southern hemisphere.
    months = tasmin.time.dt.month

    if lat is None:
        lat = _gather_lat(tasmin)
    if isinstance(lat, xarray.DataArray):
        month = xarray.where(lat > 0, 9, 3)
    elif isinstance(lat, str):
        if lat.lower() == "north":
            month = 9
        elif lat.lower() == "south":
            month = 3
        else:
            raise ValueError(f"Latitude value unsupported: {lat}.")
    else:
        raise ValueError(f"Latitude: {lat}.")

    tasmin = tasmin.where(months == month, drop=True)

    cni = tasmin.resample(time=freq).mean(keep_attrs=True)
    cni.attrs["units"] = "degC"
    return cni


@declare_units(tas="[temperature]", lat="[]")
def latitude_temperature_index(
    tas: xarray.DataArray,
    lat: xarray.DataArray | None = None,
    lat_factor: float = 75,
    freq: str = "YS",
) -> xarray.DataArray:
    """Latitude-Temperature Index.

    Mean temperature of the warmest month with a latitude-based scaling factor :cite:p:`jackson_prediction_1988`.
    Used for categorizing wine-growing regions.

    Parameters
    ----------
    tas : xarray.DataArray
        Mean daily temperature.
    lat : xarray.DataArray, optional
        Latitude coordinate.
        If None, a CF-conformant "latitude" field must be available within the passed DataArray.
    lat_factor : float
        Latitude factor. Maximum poleward latitude. Default: 75.
    freq : str
        Resampling frequency.

    Returns
    -------
    xarray.DataArray, [unitless]
        Latitude Temperature Index.

    Notes
    -----
    The latitude factor of `75` is provided for examining the poleward expansion of wine-growing climates under
    scenarios of climate change (modified from :cite:t:`kenny_assessment_1992`). For comparing 20th century/observed
    historical records, the original scale factor of `60` is more appropriate.

    Let :math:`Tn_{j}` be the average temperature for a given month :math:`j`, :math:`lat_{f}` be the latitude factor,
    and :math:`lat` be the latitude of the area of interest. Then the Latitude-Temperature Index (:math:`LTI`) is:

    .. math::

        LTI = max(TN_{j}: j = 1..12)(lat_f - |lat|)

    References
    ----------
    :cite:cts:`jackson_prediction_1988,kenny_assessment_1992`
    """
    tas = convert_units_to(tas, "degC")

    tas = tas.resample(time="MS").mean(dim="time", keep_attrs=True)
    mtwm = tas.resample(time=freq).max(dim="time", keep_attrs=True)

    if lat is None:
        lat = _gather_lat(tas)

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
    hurs="[]",
    rsds="[radiation]",
    rsus="[radiation]",
    rlds="[radiation]",
    rlus="[radiation]",
    sfcWind="[speed]",
)
def water_budget(
    pr: xarray.DataArray,
    evspsblpot: xarray.DataArray | None = None,
    tasmin: xarray.DataArray | None = None,
    tasmax: xarray.DataArray | None = None,
    tas: xarray.DataArray | None = None,
    lat: xarray.DataArray | None = None,
    hurs: xarray.DataArray | None = None,
    rsds: xarray.DataArray | None = None,
    rsus: xarray.DataArray | None = None,
    rlds: xarray.DataArray | None = None,
    rlus: xarray.DataArray | None = None,
    sfcWind: xarray.DataArray | None = None,
    method: str = "BR65",
) -> xarray.DataArray:
    r"""Precipitation minus potential evapotranspiration.

    Precipitation minus potential evapotranspiration as a measure of an approximated surface water budget,
    where the potential evapotranspiration can be calculated with a given method.

    Parameters
    ----------
    pr : xarray.DataArray
        Daily precipitation.
    evspsblpot: xarray.DataArray, optional
        Potential evapotranspiration
    tasmin : xarray.DataArray, optional
        Minimum daily temperature.
    tasmax : xarray.DataArray, optional
        Maximum daily temperature.
    tas : xarray.DataArray, optional
        Mean daily temperature.
    lat : xarray.DataArray, optional
        Latitude coordinate, needed if evspsblpot is not given.
        If None, a CF-conformant "latitude" field must be available within the `pr` DataArray.
    hurs : xarray.DataArray, optional
        Relative humidity.
    rsds : xarray.DataArray, optional
        Surface Downwelling Shortwave Radiation
    rsus : xarray.DataArray, optional
        Surface Upwelling Shortwave Radiation
    rlds : xarray.DataArray, optional
        Surface Downwelling Longwave Radiation
    rlus : xarray.DataArray, optional
        Surface Upwelling Longwave Radiation
    sfcWind : xarray.DataArray, optional
        Surface wind velocity (at 10 m)
    method : str
        Method to use to calculate the potential evapotranspiration.

    See Also
    --------
    xclim.indicators.atmos.potential_evapotranspiration

    Returns
    -------
    xarray.DataArray
        Precipitation minus potential evapotranspiration.
    """
    pr = convert_units_to(pr, "kg m-2 s-1", context="hydro")

    if lat is None and evspsblpot is None:
        lat = _gather_lat(pr)

    if evspsblpot is None:
        pet = xci.potential_evapotranspiration(
            tasmin=tasmin,
            tasmax=tasmax,
            tas=tas,
            lat=lat,
            hurs=hurs,
            rsds=rsds,
            rsus=rsus,
            rlds=rlds,
            rlus=rlus,
            sfcWind=sfcWind,
            method=method,
        )
    else:
        pet = convert_units_to(evspsblpot, "kg m-2 s-1", context="hydro")

    if xarray.infer_freq(pet.time) == "MS":
        pr = pr.resample(time="MS").mean(dim="time", keep_attrs=True)

    out = pr - pet

    out.attrs["units"] = pr.attrs["units"]
    return out


@declare_units(
    pr="[precipitation]",
    pr_cal="[precipitation]",
)
def standardized_precipitation_index(
    pr: xarray.DataArray,
    pr_cal: Quantified,
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
        Averaging window length relative to the resampling frequency. For example, if `freq="MS"`,
        i.e. a monthly resampling, the window is an integer number of months.
    dist : {"gamma", "fisk"}
        Name of the univariate distribution.
        (see :py:mod:`scipy.stats`).
    method : {'APP', 'ML'}
        Name of the fitting method, such as `ML` (maximum likelihood), `APP` (approximate). The approximate method
        uses a deterministic function that doesn't involve any optimization.

    Returns
    -------
    xarray.DataArray, [unitless]
        Standardized Precipitation Index.

    Notes
    -----
    The length `N` of the N-month SPI is determined by choosing the `window = N`.
    Supported statistical distributions are: ["gamma"]

    Example
    -------
    >>> from datetime import datetime
    >>> from xclim.indices import standardized_precipitation_index
    >>> ds = xr.open_dataset(path_to_pr_file)
    >>> pr = ds.pr
    >>> pr_cal = pr.sel(time=slice(datetime(1990, 5, 1), datetime(1990, 8, 31)))
    >>> spi_3 = standardized_precipitation_index(
    ...     pr, pr_cal, freq="MS", window=3, dist="gamma", method="ML"
    ... )  # Computing SPI-3 months using a gamma distribution for the fit

    References
    ----------
    :cite:cts:`mckee_relationship_1993`

    """
    # "WPM" method doesn't seem to work for gamma or pearson3
    dist_and_methods = {"gamma": ["ML", "APP"], "fisk": ["ML", "APP"]}
    if dist not in dist_and_methods:
        raise NotImplementedError(f"The distribution `{dist}` is not supported.")
    if method not in dist_and_methods[dist]:
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
    if dist in ["gamma", "fisk"]:
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
    wb_cal: Quantified,
    freq: str = "MS",
    window: int = 1,
    dist: str = "gamma",
    method: str = "APP",
) -> xarray.DataArray:
    r"""Standardized Precipitation Evapotranspiration Index (SPEI).

    Precipitation minus potential evapotranspiration data (PET) fitted to a statistical distribution (dist), transformed
    to a cdf,  and inverted back to a gaussian normal pdf. The potential evapotranspiration is calculated with a given
    method (`method`).

    Parameters
    ----------
    wb : xarray.DataArray
        Daily water budget (pr - pet).
    wb_cal : xarray.DataArray
        Daily water budget used for calibration.
    freq : str
        Resampling frequency. A monthly or daily frequency is expected.
    window : int
        Averaging window length relative to the resampling frequency. For example, if `freq="MS"`, i.e. a monthly
        resampling, the window is an integer number of months.
    dist : {'gamma', 'fisk'}
        Name of the univariate distribution. (see :py:mod:`scipy.stats`).
    method : {'APP', 'ML'}
        Name of the fitting method, such as `ML` (maximum likelihood), `APP` (approximate). The approximate method
        uses a deterministic function that doesn't involve any optimization. Available methods
        vary with the distribution: 'gamma':{'APP', 'ML'}, 'fisk':{'ML'}

    Returns
    -------
    xarray.DataArray
        Standardized Precipitation Evapotranspiration Index.

    See Also
    --------
    standardized_precipitation_index

    Notes
    -----
    See Standardized Precipitation Index (SPI) for more details on usage.
    """
    # Allowed distributions are constrained by the SPI function
    if dist in ["gamma", "fisk"]:
        # Distributions bounded by zero: Water budget must be shifted, only positive values
        # are allowed. The offset choice is arbitrary and the same offset as the monocongo
        # library is taken
        offset = convert_units_to("1 mm/d", wb.units, context="hydro")
        with xarray.set_options(keep_attrs=True):
            wb, wb_cal = wb + offset, wb_cal + offset

    spei = standardized_precipitation_index(wb, wb_cal, freq, window, dist, method)

    return spei


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


@declare_units(tas="[temperature]")
def qian_weighted_mean_average(
    tas: xarray.DataArray, dim: str = "time"
) -> xarray.DataArray:
    r"""Binomial smoothed, five-day weighted mean average temperature.

    Calculates a five-day weighted moving average with emphasis on temperatures closer to day of interest.

    Parameters
    ----------
    tas : xr.DataArray
        Daily mean temperature.
    dim : str
        Time dimension.

    Returns
    -------
    xr.DataArray, [same as tas]
        Binomial smoothed, five-day weighted mean average temperature.

    Notes
    -----
    Qian Modified Weighted Mean Indice originally proposed in :cite:p:`qian_observed_2010`,
    based on :cite:p:`bootsma_impacts_2005`.

    Let :math:`X_{n}` be the average temperature for day :math:`n` and :math:`X_{t}` be the daily mean temperature
    on day :math:`t`. Then the weighted mean average can be calculated as follows:

    .. math::

        \overline{X}_{n} = \frac{X_{n-2} + 4X_{n-1} + 6X_{n} + 4X_{n+1} + X_{n+2}}{16}

    References
    ----------
    :cite:cts:`bootsma_impacts_2005,qian_observed_2010`
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
    thresh: Quantified = "5 degC",
    method: str = "bootsma",
    after_date: DayOfYearStr = "07-01",
    dim: str = "time",
    freq: str = "YS",
) -> xarray.DataArray:
    r"""Effective growing degree days.

    Growing degree days based on a dynamic start and end of the growing season, as defined in :cite:p:`bootsma_impacts_2005`.

    Parameters
    ----------
    tasmax : xr.DataArray
        Daily mean temperature.
    tasmin : xr.DataArray
        Daily minimum temperature.
    thresh : Quantified
        The minimum temperature threshold.
    method : {"bootsma", "qian"}
        The window method used to determine the temperature-based start date.
        For "bootsma", the start date is defined as 10 days after the average temperature exceeds a threshold.
        For "qian", the start date is based on a weighted 5-day rolling average,
        based on :py:func`qian_weighted_mean_average`.
    after_date : str
        Date of the year after which to look for the first frost event. Should have the format '%m-%d'.
    dim : str
        Time dimension.
    freq : str
        Resampling frequency.

    Returns
    -------
    xarray.DataArray, [K days]
        Effective growing degree days (EGDD).

    Notes
    -----
    The effective growing degree days for a given year :math:`EGDD_i` can be calculated as follows:

    .. math::

        EGDD_i = \sum_{i=\text{j_{start}}^{\text{j_{end}}} max\left(TG - Thresh, 0 \right)

    Where :math:`TG` is the mean daly temperature, and :math:`j_{start}` and :math:`j_{end}` are the start and end dates
    of the growing season. The growing season start date methodology is determined via the `method` flag.
    For "bootsma", the start date is defined as 10 days after the average temperature exceeds a threshold (5 degC).
    For "qian", the start date is based on a weighted 5-day rolling average, based on :py:func:`qian_weighted_mean_average`.

    The end date is determined as the day preceding the first day with minimum temperature below 0 degC.

    References
    ----------
    :cite:cts:`bootsma_impacts_2005`
    """
    tasmax = convert_units_to(tasmax, "degC")
    tasmin = convert_units_to(tasmin, "degC")
    thresh = convert_units_to(thresh, "degC")

    tas = (tasmin + tasmax) / 2
    tas.attrs["units"] = "degC"

    if method.lower() == "bootsma":
        fda = first_day_temperature_above(tas=tas, thresh=thresh, window=1, freq=freq)
        start = fda + 10
    elif method.lower() == "qian":
        tas_weighted = qian_weighted_mean_average(tas=tas, dim=dim)
        start = first_day_temperature_above(
            tas_weighted, thresh=thresh, window=5, freq=freq
        )
    else:
        raise NotImplementedError(f"Method: {method}.")

    # The day before the first day below 0 degC
    end = (
        first_day_temperature_below(
            tasmin,
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

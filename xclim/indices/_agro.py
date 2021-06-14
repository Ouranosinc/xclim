# noqa: D100

from typing import Optional

import xarray

import xclim.indices as xci
from xclim.core.units import convert_units_to, declare_units
from xclim.core.utils import DayOfYearStr
from xclim.indices.generic import aggregate_between_dates

# Frequencies : YS: year start, QS-DEC: seasons starting in december, MS: month start
# See http://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases

# -------------------------------------------------- #
# ATTENTION: ASSUME ALL INDICES WRONG UNTIL TESTED ! #
# -------------------------------------------------- #

__all__ = [
    "corn_heat_units",
    "biologically_effective_degree_days",
    "cool_night_index",
    "water_budget",
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
    tasmin="[temperature]",
    tasmax="[temperature]",
    thresh_tasmin="[temperature]",
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
    method: {"gladstones", "icclim"}
      The formula to use for the calculation.
      The "gladstones" integrates a daily temperature range and latitude coefficient. End_date should be "11-01".
      The "icclim" method ignores daily temperature range and latitude coefficient. End date should be "10-01".
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
        TR_{adj} = f(TX_{i}, TN_{i}) = \left\{ \begin{array}{cl}
                                0.25(TX_{i} - TN_{i} - 13), & \text{if } (TX_{i} - TN_{i}) > 13 \\
                                0, & \text{if } 10 < (TX_{i} - TN_{i}) < 13\\
                                0.25(TX_{i} - TN_{i} - 10), & \text{if } (TX_{i} - TN_{i}) < 10 \\
                            \end{array} \right\}

    .. math::
        k = f(lat) = 1 + \left(\frac{\left| lat  \right|}{50} * 0.06,  \text{if }40 < |lat| <50, \text{else } 0\right)

    A second version of the BEDD (`method="icclim") does not consider :math:`TR_{adj}` and :math:`k` and employs a
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

    if method.lower() == "gladstones" and lat is not None:
        low_dtr = convert_units_to(low_dtr, "degC")
        high_dtr = convert_units_to(high_dtr, "degC")
        dtr = tasmax - tasmin
        tr_adj = 0.25 * xarray.where(
            dtr > high_dtr,
            dtr - high_dtr,
            xarray.where(dtr < low_dtr, dtr - low_dtr, 0),
        )

        lat_mask = (abs(lat) >= 40) & (abs(lat) <= 50)
        k = 1 + xarray.where(lat_mask, (abs(lat) / 50) * 0.06, 0)
    elif method.lower() == "icclim":
        k = 1
        tr_adj = 0
    else:
        raise NotImplementedError()

    bedd = ((((tasmin + tasmax) / 2) - thresh_tasmin).clip(min=0) * k + tr_adj).clip(
        max=max_daily_degree_days
    )

    bedd = aggregate_between_dates(bedd, start=start_date, end=end_date, freq=freq)

    bedd.attrs["units"] = "K days"
    return bedd


@declare_units(tasmin="[temperature]")
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

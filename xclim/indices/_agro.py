# noqa: D100

import xarray

from xclim.core.units import convert_units_to, declare_units
from xclim.core.utils import DayOfYearStr
from xclim.indices.generic import aggregate_between_dates

# Frequencies : YS: year start, QS-DEC: seasons starting in december, MS: month start
# See http://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases

# -------------------------------------------------- #
# ATTENTION: ASSUME ALL INDICES WRONG UNTIL TESTED ! #
# -------------------------------------------------- #

__all__ = ["corn_heat_units", "biologically_effective_degree_days"]


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
    thresh_tasmax="[temperature]",
)
def biologically_effective_degree_days(
    tasmin: xarray.DataArray,
    tasmax: xarray.DataArray,
    lat: xarray.DataArray,
    thresh_tasmin: str = "10 degC",
    thresh_tasmax: str = "19 degC",
    start_date: DayOfYearStr = "04-01",
    end_date: DayOfYearStr = "11-01",
    freq: str = "YS",
) -> xarray.DataArray:
    """Biologically effective growing degree days.

    Growing-degree days with a base of 10°C and an upper limit of 19°C and adjusted for latitudes between 40°N and 50°N
    for April to October (Northern Hemisphere; October to April in Southern Hemisphere). Used as a heat-summation metric
    in viticulture climatology.

    Parameters
    ----------
    tasmin: xarray.DataArray
      Minimum daily temperature.
    tasmax: xarray.DataArray
      Maximum daily temperature.
    lat: xarray.DataArray
      Latitude coordinate.
    thresh_tasmin: str
      The minimum temperature threshold.
    thresh_tasmax: str
      The maximum temperature threshold.
    start_date: DayOfYearStr
      The hemisphere-based start date to consider (north = April, south = October).
    end_date: DayOfYearStr
      The hemisphere-based start date to consider (north = October, south = April). This date is non-inclusive.
    freq : str
      Resampling frequency.

    Returns
    -------
    xarray.DataArray
      Biologically effective growing degree days (BEDD).



    References
    ----------
    Indice originally from Gladstones, J. S.  (1992).  Viticulture and environment : a study of the effects of
    environment on grapegrowing and wine qualities, with emphasis on present and future areas for growing winegrapes
    in Australia.  Adelaide :  Winetitles.
    """
    tasmin = convert_units_to(tasmin, "degC")
    tasmax = convert_units_to(tasmax, "degC")
    thresh_tasmin = convert_units_to(thresh_tasmin, "degC")
    thresh_tasmax = convert_units_to(thresh_tasmax, "degC")

    mask_tasmin = tasmin > thresh_tasmin
    tasmin = tasmin.where(mask_tasmin)
    tasmax = tasmax.where(mask_tasmin)

    lat_mask = (abs(lat) >= 40) & (abs(lat) <= 50)
    lat_constant = xarray.where(lat_mask, (abs(lat) / 50) * 0.06, 0)

    def tas_range_adjust(tmax, tmin):
        if (tmax - tmin) > 13:
            return 0.25 * (tmax - tmin - 13)
        elif (tmax - tmin) < 10:
            return 0.25 * (tmax - tmin - 10)
        else:
            return 0

    bedd = (
        (
            (tasmin.clip(max=thresh_tasmax) - thresh_tasmin)
            + (tasmax.clip(max=thresh_tasmax) - thresh_tasmin)
        )
        / 2
    ) * (1 + lat_constant) + tas_range_adjust(tasmax, tasmin)

    bedd = aggregate_between_dates(bedd, start=start_date, end=end_date, freq=freq)

    bedd.attrs["units"] = "degC"
    return bedd

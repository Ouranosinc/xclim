# noqa: D205,D400
"""
Helper functions submodule
==========================

Functions that encapsulate some geophysical logic but could be shared by many indices.
"""
from __future__ import annotations

import numpy as np
import xarray as xr

from xclim.core.calendar import datetime_to_decimal_year
from xclim.core.units import convert_units_to


def solar_declination(day_angle: xr.DataArray, method="spencer") -> xr.DataArray:
    """Solar declination

    The angle between the sun rays and the earth's equator, in radians, as approximated
    by [Spencer1971]_ or assuming the orbit is a cirle.

    Parameters
    ----------
    day_angle: xr.DataArray
      Assuming the earth makes a full circle in a year, this is the angle covered from
      the beginning of the year up to that timestep. Also called the "julian day fraction".
      See :py:func:`~xclim.core.calendar.datetime_to_decimal_year`.
    method: {'spencer', 'simple'}
      Which approximation to use. The default ("spencer") uses the first 7 terms of the
      Fourier series representing the observed declination, while "simple" assumes
      the orbit is a circle with a fixed obliquity and that the solstice/equinox happen
      at fixed angles on the orbit (the exact calendar date changes for leap years).

    Returns
    -------
    Solar declination angle, [rad]

    References
    ----------
    .. [Spencer1971] Spencer JW (1971) Fourier series representation of the position of the sun. Search 2(5):172
    """
    # julian day fraction
    da = convert_units_to(day_angle, "rad")
    if method == "simple":
        # This assumes the orbit is a perfect circle, the obliquity is 0.4091 rad (23.43°)
        # and the equinox is on the March 21st 17:20 UTC (March 20th 23:14 UTC on leap years)
        return 0.4091 * np.sin(da - 1.39)

    if method == "spencer":
        return (
            0.006918
            - 0.399912 * np.cos(da)
            + 0.070257 * np.sin(da)
            - 0.006758 * np.cos(2 * da)
            + 0.000907 * np.sin(2 * da)
            - 0.002697 * np.cos(3 * da)
            + 0.001480 * np.sin(3 * da)
        )
    raise NotImplementedError(f"Method {method} must be one of 'simple' or 'spencer'")


def eccentricity_correction_factor(day_angle: xr.DataArray, method="spencer"):
    """Eccentricity correction factor of the Earth's orbit

    The squared ratio of the mean distance Earth-Sun to the distance at a specific moment.
    As approximated by [Spencer1971]_.

    Parameters
    ----------
    day_angle: xr.DataArray
      Assuming the earth makes a full circle in a year, this is the angle covered from
      the beginning of the year up to that timestep. Also called the "julian day fraction".
      See :py:func:`~xclim.core.calendar.datetime_to_decimal_year`.
    method:
      Which approximation to use. The default ("spencer") uses the first five terms of
      the fourier series of the eccentrencity, while "simple" approximates with only
      the first two.

    Returns
    -------
    Eccentricity correction factor, [dimensionless]

    References
    ----------
    Spencer JW (1971) Fourier series representation of the position of the sun. Search 2(5):172
    """
    # julian day fraction
    da = convert_units_to(day_angle, "rad")
    if method == "simple":
        # It is quite used, I think the source is (not available online) : Perrin de Brichambaut, C. (1975). Estimation des ressources énergétiques solaires en France. Ed. Européennes thermique et industrie.
        return 1 + 0.033 * np.cos(da)
    if method == "spencer":
        return (
            1.0001100
            + 0.034221 * np.cos(da)
            + 0.001280 * np.sin(da)
            + 0.000719 * np.cos(2 * da)
            + 0.000077 * np.sin(2 * da)
        )


def cosine_of_solar_zenith_angle(
    declination: xr.DataArray, lat: xr.DataArray, stat: str = "integral"
) -> xr.DataArray:
    """Cosine of the solar zenith angle.

    The solar zenith angle is the angle between a vertical line (perpendicular to the ground)
    and the sun rays. This function computes a daily statistic of its cosine : its integral
    from sunrise to sunset or the average over the same period. Based on [Kalogirou14]_.

    Parameters
    ----------
    declination : xr.DataArray
      Daily solar declination. See :py:func:`solar_declination`.
    lat : xr.DataArray
      Latitude.
    stat : {'integral', 'average'}
      Which daily statistic to return. If "integral", this returns the integral of the
      cosine of the zenith angle from sunrise to sunset. If "average", the integral is
      divided by the "duration" from sunrise to sunset.

    Returns
    -------
    Cosine of the solar zenith angle, [rad] or [dimensionless]
      If stat is "integral", dimensions can be said to be "time" as the integral is on
      the hour angle. For seconds, multiply by the number of seconds in a comple day
      cycle (24*60*60) and divide by 2π.

    References
    ----------
    Kalogirou, S. A. (2014). Chapter 2 — Environmental Characteristics. In S. A. Kalogirou (Ed.), Solar Energy Engineering (Second Edition) (pp. 51–123). Academic Press. https://doi.org/10.1016/B978-0-12-397270-5.00002-9
    """
    lat = convert_units_to(lat, "rad")
    h_s = np.arccos(
        -np.tan(lat) * np.tan(declination)
    )  # hour angle of sunset (eq. 2.15)
    # The following equation is not explictely stated in the reference but it can easily be derived.
    if stat == "integral":
        return 2 * (
            h_s * np.sin(declination) * np.sin(lat)
            + np.cos(declination) * np.cos(lat) * np.sin(h_s)
        )
    if stat == "average":
        return (
            np.sin(declination) * np.sin(lat)
            + np.cos(declination) * np.cos(lat) * np.sin(h_s) / h_s
        )
    raise NotImplementedError("Argument 'stat' must be one of 'integral' or 'average.")


def extraterrestrial_solar_radiation(
    times: xr.DataArray,
    lat: xr.DataArray,
    solar_constant: str = "1361 W m-2",
    method="spencer",
) -> xr.DataArray:
    """Extraterrestrial solar radiation

    This is the daily energy received on a surface parallel to the ground at the mean
    distance of the earth to the sun. It neglects the effect of the atmosphere. Computation
    is based on [Kalogirou14]_ and the default solar constant is taken from [Matthes17]_.

    Parameters
    ----------
    times: xr.DataArray
      Daily datetime data. This function makes no sense with data of other frequency.
    lat : xr.DataArray
      Latitude.
    solar_constant : str
      The solar constant, the energy received on earth from the sun per surface per time.
    method : {'spencer', 'simple'}
      Which method to use when computing the solar declination and the eccentricity
      correction factor. See :py:func:`solar_declination` and :py:func:`eccentricity_correction_factor`.

    Returns
    -------
    Extraterrestrial solar radiation, [J m-2 d-1]

    References
    ----------
    .. [Matthes17] Matthes, K. et al. (2017). Solar forcing for CMIP6 (v3.2). Geoscientific Model Development, 10(6), 2247–2302. https://doi.org/10.5194/gmd-10-2247-2017
    .. [Kalogirou14] Kalogirou, S. A. (2014). Chapter 2 — Environmental Characteristics. In S. A. Kalogirou (Ed.), Solar Energy Engineering (Second Edition) (pp. 51–123). Academic Press. https://doi.org/10.1016/B978-0-12-397270-5.00002-9
    """
    da = ((datetime_to_decimal_year(times) % 1) * 2 * np.pi).assign_attrs(units="rad")
    dr = eccentricity_correction_factor(da, method=method)
    ds = solar_declination(da, method=method)
    gsc = convert_units_to(solar_constant, "J m-2 d-1")
    rad_to_day = 1 / (2 * np.pi)  # convert radians of the "day circle" to day
    return (
        gsc * rad_to_day * cosine_of_solar_zenith_angle(ds, lat, stat="integral") * dr
    ).assign_attrs(units="J m-2 d-1")


def day_lengths(
    dates: xr.DataArray,
    lat: xr.DataArray,
    method: str = "spencer",
) -> xr.DataArray:
    r"""Day-lengths according to latitude and day of year.

    See :py:func:`solar_declination` for the approximation used to compute the solar
    declination angle. Based on [Kalogirou14]_.

    Parameters
    ----------
    dates: xr.DataArray
    lat: xarray.DataArray
      Latitude coordinate.
    method : {'spencer', 'simple'}
      Which approximation to use when computing the solar declination angle.
      See :py:func:`solar_declination`.

    Returns
    -------
    xarray.DataArray, [hours]
      Day-lengths in hours per individual day.

    References
    ----------
    Kalogirou, S. A. (2014). Chapter 2 — Environmental Characteristics. In S. A. Kalogirou (Ed.), Solar Energy Engineering (Second Edition) (pp. 51–123). Academic Press. https://doi.org/10.1016/B978-0-12-397270-5.00002-9
    """
    day_angle = ((datetime_to_decimal_year(dates.time) % 1) * 2 * np.pi).assign_attrs(
        units="rad"
    )
    declination = solar_declination(day_angle, method=method)
    lat = convert_units_to(lat, "rad")
    # arccos gives the hour-angle at sunset, multiply by 24 / 2π to get hours.
    # The day length is twice that.
    day_length_hours = (
        (24 / np.pi) * np.arccos(-np.tan(lat) * np.tan(declination))
    ).assign_attrs(units="h")

    return day_length_hours

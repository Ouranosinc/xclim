# noqa: D205,D400
"""
Helper functions submodule
==========================

Functions that encapsulate some geophysical logic but could be shared by many indices.
"""
from __future__ import annotations

from inspect import stack

import cf_xarray  # noqa
import cftime
import numpy as np
import xarray as xr

from xclim.core.calendar import (
    datetime_to_decimal_year,
    ensure_cftime_array,
    get_calendar,
)
from xclim.core.units import convert_units_to
from xclim.core.utils import Quantified


def distance_from_sun(dates: xr.DataArray) -> xr.DataArray:
    """
    Sun-earth distance.

    The distance from sun to earth in astronomical units.

    Parameters
    ----------
    dates : xr.DataArray
        Series of dates and time of days.

    Returns
    -------
    xr.DataArray, [astronomical units]
        Sun-earth distance.

    References
    ----------
    # TODO: Find a way to reference this
    U.S. Naval Observatory:Astronomical Almanac. Washington, D.C.: U.S. Government Printing Office (1985).
    """
    cal = get_calendar(dates)
    if cal == "default":
        cal = "standard"
    days_since = cftime.date2num(
        ensure_cftime_array(dates), "days since 2000-01-01 12:00:00", calendar=cal
    )
    g = ((357.528 + 0.9856003 * days_since) % 360) * np.pi / 180
    sun_earth = 1.00014 - 0.01671 * np.cos(g) - 0.00014 * np.cos(2.0 * g)
    return xr.DataArray(sun_earth, coords=dates.coords, dims=dates.dims)


def solar_declination(day_angle: xr.DataArray, method="spencer") -> xr.DataArray:
    """Solar declination.

    The angle between the sun rays and the earth's equator, in radians, as approximated
    by :cite:t:`spencer_fourier_1971` or assuming the orbit is a circle.

    Parameters
    ----------
    day_angle : xr.DataArray
      Assuming the earth makes a full circle in a year, this is the angle covered from
      the beginning of the year up to that timestep. Also called the "julian day fraction".
      See :py:func:`~xclim.core.calendar.datetime_to_decimal_year`.
    method : {'spencer', 'simple'}
      Which approximation to use. The default ("spencer") uses the first 7 terms of the
      Fourier series representing the observed declination, while "simple" assumes
      the orbit is a circle with a fixed obliquity and that the solstice/equinox happen
      at fixed angles on the orbit (the exact calendar date changes for leap years).

    Returns
    -------
    xr.DataArray, [rad]
        Solar declination angle.

    References
    ----------
    :cite:cts:`spencer_fourier_1971`
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


def time_correction_for_solar_angle(day_angle: xr.DataArray) -> xr.DataArray:
    """Time correction for solar angle.

    Every 1° of angular rotation on earth is equal to 4 minutes of time.
    The time correction is needed to adjust local watch time to solar time.

    Parameters
    ----------
    day_angle : xr.DataArray
        Assuming the earth makes a full circle in a year, this is the angle covered from
        the beginning of the year up to that timestep. Also called the "julian day fraction".
        See :py:func:`~xclim.core.calendar.datetime_to_decimal_year`.

    Returns
    -------
    xr.DataArray, [rad]
        Time correction of solar angle.

    References
    ----------
    :cite:cts:`di_napoli_mean_2020`
    """
    da = convert_units_to(day_angle, "rad")
    tc = (
        0.004297
        + 0.107029 * np.cos(da)
        - 1.837877 * np.sin(da)
        - 0.837378 * np.cos(2 * da)
        - 2.340475 * np.sin(2 * da)
    )
    tc = tc.assign_attrs(units="degrees")
    return convert_units_to(tc, "rad")


def eccentricity_correction_factor(day_angle: xr.DataArray, method="spencer"):
    """Eccentricity correction factor of the Earth's orbit.

    The squared ratio of the mean distance Earth-Sun to the distance at a specific moment.
    As approximated by :cite:t:`spencer_fourier_1971`.

    Parameters
    ----------
    day_angle : xr.DataArray
        Assuming the earth makes a full circle in a year, this is the angle covered from the beginning of the year up to
        that timestep. Also called the "julian day fraction".
        See :py:func:`~xclim.core.calendar.datetime_to_decimal_year`.
    method : str
        Which approximation to use. The default ("spencer") uses the first five terms of the fourier series of the
        eccentricity, while "simple" approximates with only the first two.

    Returns
    -------
    xr.DataArray, [dimensionless]
        Eccentricity correction factor.

    References
    ----------
    :cite:cts:`spencer_fourier_1971,perrin_estimation_1975`
    """
    # julian day fraction
    da = convert_units_to(day_angle, "rad")
    if method == "simple":
        # It is quite used, I think the source is (not available online):
        # Perrin de Brichambaut, C. (1975).
        # Estimation des ressources énergétiques solaires en France. Ed. Européennes thermique et industrie.
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
    declination: xr.DataArray,
    lat: xr.DataArray,
    lon: xr.DataArray = None,
    time_correction: xr.DataArray = None,
    hours: xr.DataArray = None,
    interval: int = None,
    stat: str = "integral",
) -> xr.DataArray:
    """Cosine of the solar zenith angle.

    The solar zenith angle is the angle between a vertical line (perpendicular to the ground) and the sun rays.
    This function computes a daily statistic of its cosine : its integral from sunrise to sunset or the average over
    the same period. Based on :cite:t:`kalogirou_chapter_2014`. In addition, it computes instantaneous values of its
    cosine. Based on :cite:t:`di_napoli_mean_2020`.

    Parameters
    ----------
    declination : xr.DataArray
        Solar declination. See :py:func:`solar_declination`.
    lat : xr.DataArray
        Latitude.
    lon : xr.DataArray, optional
        Longitude.
        This is necessary if stat is "instant", "interval" or "sunlit".
    time_correction : xr.DataArray, optional
        Time correction for solar angle. See :py:func:`time_correction_for_solar_angle`
        This is necessary if stat is "instant".
    hours : xr.DataArray, optional
        Watch time hours.
        This is necessary if stat is "instant", "interval" or "sunlit".
    interval : int, optional
        Time interval between two time steps in hours
        This is necessary if stat is "interval" or "sunlit".
    stat : {'integral', 'average', 'instant', 'interval', 'sunlit'}
        Which daily statistic to return. If "integral", this returns the integral of the cosine of the zenith angle from
        sunrise to sunset. If "average", the integral is divided by the "duration" from sunrise to sunset. If "instant",
        this returns the instantaneous cosine of the zenith angle. If "interval", this returns the cosine of the zenith
        angle during each interval. If "sunlit", this returns the cosine of the zenith angle during the sunlit period of
        each interval.

    Returns
    -------
    xr.DataArray, [rad] or [dimensionless]
        Cosine of the solar zenith angle. If stat is "integral", dimensions can be said to be "time" as the integral
        is on the hour angle.
        For seconds, multiply by the number of seconds in a complete day cycle (24*60*60) and divide by 2π.

    Notes
    -----
    This code was inspired by the `thermofeel` and `PyWBGT` package.

    References
    ----------
    :cite:cts:`kalogirou_chapter_2014,di_napoli_mean_2020`
    """
    lat = convert_units_to(lat, "rad")
    if lon is not None:
        lon = convert_units_to(lon, "rad")
    if hours is not None:
        sha = (hours - 12) * 15 / 180 * np.pi + lon
    if interval is not None:
        k = interval / 2.0
        h_s = sha - k * 15 * np.pi / 180
        h_e = sha + k * 15 * np.pi / 180
    h_sr = -np.arccos(-np.tan(lat) * np.tan(declination))
    h_ss = np.arccos(
        -np.tan(lat) * np.tan(declination)
    )  # hour angle of sunset (eq. 2.15)
    # The following equation is not explicitly stated in the reference, but it can easily be derived.
    if stat == "integral":
        csza = 2 * (
            h_ss * np.sin(declination) * np.sin(lat)
            + np.cos(declination) * np.cos(lat) * np.sin(h_ss)
        )
        return xr.where(np.isnan(csza), 0, csza)
    if stat == "average":
        csza = (
            np.sin(declination) * np.sin(lat)
            + np.cos(declination) * np.cos(lat) * np.sin(h_ss) / h_ss
        )
        return xr.where(np.isnan(csza), 0, csza)
    if stat == "instant":
        sha = sha + time_correction
        csza = np.sin(declination) * np.sin(lat) + np.cos(declination) * np.cos(
            lat
        ) * np.cos(sha)
        return csza.clip(0, None)
    if stat == "interval":
        csza = np.sin(declination) * np.sin(lat) + np.cos(declination) * np.cos(lat) * (
            np.sin(h_e) - np.sin(h_s)
        ) / (h_e - h_s)
        return csza.clip(0, None)
    if stat == "sunlit":
        h_min = xr.where(h_s >= h_sr, h_s, h_sr)
        h_max = xr.where(h_e <= h_ss, h_e, h_ss)
        csza = np.sin(declination) * np.sin(lat) + np.cos(declination) * np.cos(lat) * (
            np.sin(h_max) - np.sin(h_min)
        ) / (h_max - h_min)
        csza = xr.where(np.isnan(csza), 0, csza)
        return csza.clip(0, None)
    raise NotImplementedError(
        "Argument 'stat' must be one of 'integral', 'average', 'instant', 'interval' or 'sunlit'."
    )


def extraterrestrial_solar_radiation(
    times: xr.DataArray,
    lat: xr.DataArray,
    solar_constant: Quantified = "1361 W m-2",
    method="spencer",
) -> xr.DataArray:
    """Extraterrestrial solar radiation.

    This is the daily energy received on a surface parallel to the ground at the mean distance of the earth to the sun.
    It neglects the effect of the atmosphere. Computation is based on :cite:t:`kalogirou_chapter_2014` and the default
    solar constant is taken from :cite:t:`matthes_solar_2017`.

    Parameters
    ----------
    times : xr.DataArray
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
    :cite:cts:`kalogirou_chapter_2014,matthes_solar_2017`
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

    See :py:func:`solar_declination` for the approximation used to compute the solar declination angle.
    Based on :cite:t:`kalogirou_chapter_2014`.

    Parameters
    ----------
    dates: xr.DataArray
        Daily datetime data. This function makes no sense with data of other frequency.
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
    :cite:cts:`kalogirou_chapter_2014`
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


def wind_speed_height_conversion(
    ua: xr.DataArray,
    h_source: str,
    h_target: str,
    method: str = "log",
) -> xr.DataArray:
    r"""Wind speed at two meters.

    Parameters
    ----------
    ua : xarray.DataArray
        Wind speed at height h
    h_source : str
        Height of the input wind speed `ua` (e.g. `h == "10 m"` for a wind speed at `10 meters`)
    h_target : str
        Height of the output wind speed
    method : {"log"}
        Method used to convert wind speed from one height to another

    Returns
    -------
    xarray.DataArray
        Wind speed at height `h_target`

    References
    ----------
    :cite:cts:`allen_crop_1998`
    """
    h_source = convert_units_to(h_source, "m")
    h_target = convert_units_to(h_target, "m")
    if method == "log":
        if min(h_source, h_target) < 1 + 5.42 / 67.8:
            raise ValueError(
                f"The height {min(h_source, h_target)}m is too small for method {method}. Heights must be greater than {1 + 5.42 / 67.8}"
            )
        with xr.set_options(keep_attrs=True):
            return ua * np.log(67.8 * h_target - 5.42) / np.log(67.8 * h_source - 5.42)
    else:
        raise NotImplementedError(f"'{method}' method is not implemented.")


def _gather_lat(da: xr.DataArray) -> xr.DataArray:
    """Gather latitude coordinate using cf-xarray.

    Parameters
    ----------
    da : xarray.DataArray
        CF-conformant DataArray with a "latitude" coordinate.

    Returns
    -------
    xarray.DataArray
        Latitude coordinate.
    """
    try:
        lat = da.cf["latitude"]
        return lat
    except KeyError as err:
        n_func = stack()[1].function
        msg = (
            f"{n_func} could not find latitude coordinate in DataArray. "
            "Try passing it explicitly (`lat=ds.lat`)."
        )
        raise ValueError(msg) from err


def _gather_lon(da: xr.DataArray) -> xr.DataArray:
    """Gather longitude coordinate using cf-xarray.

    Parameters
    ----------
    da : xarray.DataArray
        CF-conformant DataArray with a "longitude" coordinate.

    Returns
    -------
    xarray.DataArray
        Longitude coordinate.
    """
    try:
        lat = da.cf["longitude"]
        return lat
    except KeyError as err:
        n_func = stack()[1].function
        msg = (
            f"{n_func} could not find longitude coordinate in DataArray. "
            "Try passing it explicitly (`lon=ds.lon`)."
        )
        raise ValueError(msg) from err

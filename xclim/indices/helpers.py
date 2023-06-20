"""
Indices Helper Functions Submodule
==================================

Functions that encapsulate some geophysical logic but could be shared by many indices.
"""
from __future__ import annotations

from inspect import stack

import cf_xarray  # noqa: F401, pylint: disable=unused-import
import cftime
import numba as nb
import numpy as np
import xarray as xr

from xclim.core.calendar import (
    datetime_to_decimal_year,
    ensure_cftime_array,
    get_calendar,
)
from xclim.core.units import convert_units_to
from xclim.core.utils import Quantified


def _wrap_radians(da):
    with xr.set_options(keep_attrs=True):
        return ((da + np.pi) % (2 * np.pi)) - np.pi


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


def day_angle(time: xr.DataArray):
    """Day of year as an angle.

    Assuming the earth makes a full circle in a year, this is the angle covered from
    the beginning of the year up to that timestep. Also called the "julian day fraction".
    See :py:func:`~xclim.core.calendar.datetime_to_decimal_year`.
    """
    decimal_year = datetime_to_decimal_year(times=time, calendar=time.dt.calendar)
    return ((decimal_year % 1) * 2 * np.pi).assign_attrs(units="rad")


def solar_declination(time: xr.DataArray, method="spencer") -> xr.DataArray:
    """Solar declination.

    The angle between the sun rays and the earth's equator, in radians, as approximated
    by :cite:t:`spencer_fourier_1971` or assuming the orbit is a circle.

    Parameters
    ----------
    time: xr.DataArray
      Time coordinate.
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
    da = convert_units_to(day_angle(time), "rad")
    if method == "simple":
        # This assumes the orbit is a perfect circle, the obliquity is 0.4091 rad (23.43°)
        # and the equinox is on the March 21st 17:20 UTC (March 20th 23:14 UTC on leap years)
        sd = 0.4091 * np.sin(da - 1.39)
    elif method == "spencer":
        sd = (
            0.006918
            - 0.399912 * np.cos(da)
            + 0.070257 * np.sin(da)
            - 0.006758 * np.cos(2 * da)
            + 0.000907 * np.sin(2 * da)
            - 0.002697 * np.cos(3 * da)
            + 0.001480 * np.sin(3 * da)
        )
    else:
        raise NotImplementedError(
            f"Method {method} must be one of 'simple' or 'spencer'"
        )
    return _wrap_radians(sd).assign_attrs(units="rad").rename("declination")


def time_correction_for_solar_angle(time: xr.DataArray) -> xr.DataArray:
    """Time correction for solar angle.

    Every 1° of angular rotation on earth is equal to 4 minutes of time.
    The time correction is needed to adjust local watch time to solar time.

    Parameters
    ----------
    time: xr.DataArray
      Time coordinate.

    Returns
    -------
    xr.DataArray, [rad]
        Time correction of solar angle.

    References
    ----------
    :cite:cts:`di_napoli_mean_2020`
    """
    da = convert_units_to(day_angle(time), "rad")
    tc = (
        0.004297
        + 0.107029 * np.cos(da)
        - 1.837877 * np.sin(da)
        - 0.837378 * np.cos(2 * da)
        - 2.340475 * np.sin(2 * da)
    )
    tc = tc.assign_attrs(units="degrees")
    return _wrap_radians(convert_units_to(tc, "rad"))


def eccentricity_correction_factor(time: xr.DataArray, method="spencer"):
    """Eccentricity correction factor of the Earth's orbit.

    The squared ratio of the mean distance Earth-Sun to the distance at a specific moment.
    As approximated by :cite:t:`spencer_fourier_1971`.

    Parameters
    ----------
    time: xr.DataArray
        Time coordinate
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
    da = convert_units_to(day_angle(time), "rad")
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
    time: xr.DataArray,
    declination: xr.DataArray,
    lat: Quantified,
    lon: Quantified = "0 °",
    time_correction: xr.DataArray = None,
    stat: str = "integral",
    sunlit: bool = False,
) -> xr.DataArray:
    """Cosine of the solar zenith angle.

    The solar zenith angle is the angle between a vertical line (perpendicular to the ground) and the sun rays.
    This function computes a statistic of its cosine : its instantaneous value, the integral from sunrise to sunset or the average over
    the same period or over a subdaily interval.
    Based on :cite:t:`kalogirou_chapter_2014` and :cite:t:`di_napoli_mean_2020`.

    Parameters
    ----------
    time: xr.DataArray
        The UTC time. If not daily and `stat` is "integral" or "average", the timestamp is taken as the start of interval.
        If daily, the interval is assumed to be centered on Noon.
        If fewer than three timesteps are given, a daily frequency is assumed.
    declination : xr.DataArray
        Solar declination. See :py:func:`solar_declination`.
    lat : Quantified
        Latitude.
    lon : Quantified
        Longitude. Needed if the input timeseries is subdaily.
    time_correction : xr.DataArray, optional
        Time correction for solar angle. See :py:func:`time_correction_for_solar_angle`
        This is necessary if stat is "instant".
    stat : {'integral', 'average', 'instant'}
        Which daily statistic to return.
        If "integral", this returns the integral of the cosine of the zenith angle
        If "average", this returns the average of the cosine of the zenith angle
        If "instant", this returns the instantaneous cosine of the zenith angle
    sunlit: bool
        If True, only the sunlit part of the interval is considered in the integral or average.
        Does nothing if stat is "instant".

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
    declination = convert_units_to(declination, "rad")
    lat = _wrap_radians(convert_units_to(lat, "rad"))
    lon = convert_units_to(lon, "rad")
    S_IN_D = 24 * 3600

    if len(time) < 3 or xr.infer_freq(time) == "D":
        h_s = -np.pi if stat != "instant" else 0
        h_e = np.pi - 1e-9  # just below pi
    else:
        if time.dtype == "O":  # cftime
            time_as_s = time.copy(data=xr.CFTimeIndex(time.values).asi8 / 1e6)
        else:  # numpy
            time_as_s = time.copy(data=time.astype(float) / 1e9)
        h_s_utc = (((time_as_s % S_IN_D) / S_IN_D) * 2 * np.pi + np.pi).assign_attrs(
            units="rad"
        )
        h_s = h_s_utc + lon

        interval_as_s = time.diff("time").dt.seconds.reindex(
            time=time.time, method="bfill"
        )
        h_e = h_s + 2 * np.pi * interval_as_s / S_IN_D

    if stat == "instant":
        h_s = h_s + time_correction
        return (
            np.sin(declination) * np.sin(lat)
            + np.cos(declination) * np.cos(lat) * np.cos(h_s)
        ).clip(0, None)
    elif stat not in {"average", "integral"}:
        raise NotImplementedError(
            "Argument 'stat' must be one of 'integral', 'average' or 'instant'."
        )
    if sunlit:
        # hour angle of sunset (eq. 2.15), with NaNs inside the polar day/night
        tantan = -np.tan(lat) * np.tan(declination)
        h_ss = np.arccos(tantan.where(abs(tantan) <= 1))
    else:
        # Whole period, so we put sunset at midnight
        h_ss = np.pi - 1e-9

    return xr.apply_ufunc(
        _sunlit_integral_of_cosine_of_solar_zenith_angle,
        declination,
        lat,
        _wrap_radians(h_ss),
        _wrap_radians(h_s),
        _wrap_radians(h_e),
        stat == "average",
        input_core_dims=[[]] * 6,
        dask="parallel",
    )


@nb.vectorize
def _sunlit_integral_of_cosine_of_solar_zenith_angle(
    declination, lat, h_sunset, h_start, h_end, average
):
    """Integral of the cosine of the the solar zenith angle over the sunlit part of the interval."""
    # Code inspired by PyWBGT
    h_sunrise = -h_sunset
    # Polar day
    if np.isnan(h_sunset) & ((declination * lat) > 0):
        num = np.sin(h_end) - np.sin(h_start)
        # Polar day with interval crossing midnight
        if h_end < h_start:
            denum = h_end + 2 * np.pi - h_start
        else:
            denum = h_end - h_start
    # Polar night:
    elif np.isnan(h_sunset) & ((declination * lat) < 0):
        return 0
    # No sunlit interval (at night) 1) crossing midnight and 2) between 0h and sunrise 3) between sunset and 0h
    elif (
        (h_start > h_sunset and h_end < h_sunrise)
        or (h_start < h_sunrise and h_end < h_sunrise)
        or (h_start > h_sunset and h_end > h_sunset)
    ):
        return 0
    # Interval crossing midnight, starting after sunset (before midnight), finishing after sunrise
    elif h_end < h_start and h_start >= h_sunset and h_end >= h_sunrise:
        num = np.sin(h_end) - np.sin(h_sunrise)
        denum = h_end - h_sunrise
    # Interval crossing midnight, starting after sunrise, finishing after sunset (after midnight)
    elif h_end < h_start and h_start >= h_sunrise and h_end <= h_sunrise:
        num = np.sin(h_sunset) - np.sin(h_start)
        denum = h_sunset - h_start
    # Interval crossing midnight, starting before sunset, finsing after sunrise (2 sunlit parts)
    elif h_end < h_start and h_start <= h_sunset and h_end >= h_sunrise:
        num = np.sin(h_sunset) - np.sin(h_start) + np.sin(h_end) - np.sin(h_sunrise)
        denum = h_sunset - h_start + h_end - h_sunrise
    # All other cases : interval not crossing midnight, overlapping with the sunlit part
    else:
        h1 = max(h_sunrise, h_start)
        h2 = min(h_sunset, h_end)
        num = np.sin(h2) - np.sin(h1)
        denum = h2 - h1
    out = (
        np.sin(declination) * np.sin(lat) * denum
        + np.cos(declination) * np.cos(lat) * num
    )
    if average:
        out = out / denum
    return out


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
    dr = eccentricity_correction_factor(times, method=method)
    ds = solar_declination(times, method=method)
    gsc = convert_units_to(solar_constant, "J m-2 d-1")
    rad_to_day = 1 / (2 * np.pi)  # convert radians of the "day circle" to day
    return (
        gsc
        * rad_to_day
        * cosine_of_solar_zenith_angle(times, ds, lat, stat="integral", sunlit=True)
        * dr
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
    declination = solar_declination(dates.time, method=method)
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

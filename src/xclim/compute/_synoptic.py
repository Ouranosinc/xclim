"""Synoptic indice definitions."""

from __future__ import annotations

import cf_xarray  # noqa: F401, pylint: disable=unused-import
import numpy as np
import xarray

from xclim.core.units import convert_units_to, declare_units

# Frequencies : YS: year start, QS-DEC: seasons starting in december, MS: month start
# See https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html

# -------------------------------------------------- #
# ATTENTION: ASSUME ALL INDICES WRONG UNTIL TESTED ! #
# -------------------------------------------------- #

__all__ = [
    "jetstream_metric_woollings",
]


@declare_units(ua="[speed]")
def jetstream_metric_woollings(
    ua: xarray.DataArray,
) -> tuple[xarray.DataArray, xarray.DataArray]:
    """
    Strength and latitude of jetstream.

    Identify latitude and strength of maximum smoothed zonal wind speed in the region from 15 to 75°N and -60 to 0°E,
    using the formula outlined in :cite:p:`woollings_variability_2010`. Wind is smoothened using a Lanczos filter
    approach.

    Parameters
    ----------
    ua : xarray.DataArray
        Eastward wind component (u) at between 750 and 950 hPa.

    Returns
    -------
    (xarray.DataArray, xarray.DataArray)
        Daily time series of latitude of jetstream and daily time series of strength of jetstream.

    Warnings
    --------
    This metric expects eastward wind component (u) to be on a regular grid (i.e. Plate Carree, 1D lat and lon)

    References
    ----------
    :cite:cts:`woollings_variability_2010`
    """
    # Select longitudes in the -60 to 0°E range
    lon = ua.cf["longitude"]
    ilon = (lon >= 300) * (lon <= 360) + (lon >= -60) * (lon <= 0)
    if not ilon.any():
        raise ValueError("Make sure the grid includes longitude values in a range between -60 and 0°E.")

    # Select latitudes in the 15 to 75°N range
    lat = ua.cf["latitude"]
    ilat = (lat >= 15) * (lat <= 75)
    if not ilat.any():
        raise ValueError("Make sure the grid includes latitude values in a range between 15 and 75°N.")

    # Select levels between 750 and 950 hPa
    pmin = convert_units_to("750 hPa", ua.cf["vertical"])
    pmax = convert_units_to("950 hPa", ua.cf["vertical"])

    p = ua.cf["vertical"]
    ip = (p >= pmin) * (p <= pmax)
    if not ip.any():
        raise ValueError("Make sure the grid includes pressure values in a range between 750 and 950 hPa.")

    ua = ua.cf.sel(
        vertical=ip,
        latitude=ilat,
        longitude=ilon,
    )

    zonal_mean = ua.cf.mean(["vertical", "longitude"])

    # apply Lanczos filter - parameters are hard-coded following those used in Woollings (2010)
    filter_freq = 10
    window_size = 61
    cutoff = 1 / filter_freq
    if ua.time.size <= filter_freq or ua.time.size <= window_size:
        raise ValueError(f"Time series is too short to apply 61-day Lanczos filter (got a length of  {ua.time.size})")

    # compute low-pass filter weights
    lanczos_weights = _compute_low_pass_filter_weights(window_size=window_size, cutoff=cutoff)
    # apply the filter
    ua_lf = zonal_mean.rolling(time=window_size, center=True).construct("window").dot(lanczos_weights)

    # Get latitude & eastward wind component units
    lat_name = ua.cf["latitude"].name
    lat_units = ua.cf["latitude"].units
    ua_units = ua.units

    jetlat = ua_lf.cf.idxmax(lat_name).rename("jetlat").assign_attrs(units=lat_units)
    jetstr = ua_lf.cf.max(lat_name).rename("jetstr").assign_attrs(units=ua_units)
    return jetlat, jetstr


def _compute_low_pass_filter_weights(window_size: int, cutoff: float) -> xarray.DataArray:
    order = ((window_size - 1) // 2) + 1
    nwts = 2 * order + 1
    w = np.zeros([nwts])
    n = nwts // 2
    w[n] = 2 * cutoff
    k = np.arange(1.0, n)
    sigma = np.sin(np.pi * k / n) * n / (np.pi * k)
    firstfactor = np.sin(2.0 * np.pi * cutoff * k) / (np.pi * k)
    w[n - 1 : 0 : -1] = firstfactor * sigma
    w[n + 1 : -1] = firstfactor * sigma

    lanczos_weights = xarray.DataArray(w[0 + (window_size % 2) : -1], dims=["window"])
    return lanczos_weights

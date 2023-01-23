# noqa: D100
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
    """Strength and latitude of jetstream.

    Identify latitude and strength of maximum smoothed zonal wind speed in the region from 15 to 75°N and -60 to 0°E,
    using the formula outlined in :cite:p:`woollings_variability_2010`. Wind is smoothened using a Lanczos filter
    approach.

    Warnings
    --------
    This metric expects eastward wind component (u) to be on a regular grid (i.e. Plate Carree, 1D lat and lon)

    Parameters
    ----------
    ua : xarray.DataArray
        Eastward wind component (u) at between 750 and 950 hPa.

    Returns
    -------
    (xarray.DataArray, xarray.DataArray)
        Daily time series of latitude of jetstream and Daily time series of strength of jetstream.

    References
    ----------
    :cite:cts:`woollings_variability_2010`
    """
    lon_min = -60
    lon_max = 0
    lons_within_range = any(
        (ua.cf["longitude"] >= lon_min) & (ua.cf["longitude"] <= lon_max)
    )
    if not lons_within_range:
        raise ValueError(
            f"Longitude values need to be in a range between {lon_min}-{lon_max}. "
            "Consider changing the longitude coordinates to between -180 degrees E – 180 degrees W."
        )

    # get latitude & eastward wind component units
    lat_units = ua.cf["latitude"].units
    ua_units = ua.units
    lat_name = ua.cf["latitude"].name

    # select only relevant hPa levels, compute zonal mean wind speed
    pmin = convert_units_to("750 hPa", ua.cf["pressure"])
    pmax = convert_units_to("950 hPa", ua.cf["pressure"])

    ua = ua.cf.sel(
        pressure=slice(pmin, pmax),
        latitude=slice(15, 75),
        longitude=slice(lon_min, lon_max),
    )

    zonal_mean = ua.cf.mean(["pressure", "longitude"])

    # apply Lanczos filter - parameters are hard-coded following those used in Woollings (2010)
    filter_freq = 10
    window_size = 61
    cutoff = 1 / filter_freq
    if ua.time.size <= filter_freq or ua.time.size <= window_size:
        raise ValueError(
            f"Time series is too short to apply 61-day Lanczos filter (got a length of  {ua.time.size})"
        )

    # compute low-pass filter weights
    lanczos_weights = _compute_low_pass_filter_weights(
        window_size=window_size, cutoff=cutoff
    )
    # apply the filter
    ua_lf = (
        zonal_mean.rolling(time=window_size, center=True)
        .construct("window")
        .dot(lanczos_weights)
    )
    jetlat = ua_lf.cf.idxmax(lat_name).rename("jetlat").assign_attrs(units=lat_units)
    jetstr = ua_lf.cf.max(lat_name).rename("jetstr").assign_attrs(units=ua_units)
    return jetlat, jetstr


def _compute_low_pass_filter_weights(
    window_size: int, cutoff: float
) -> xarray.DataArray:
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

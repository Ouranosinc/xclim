import numpy as np
import xarray

from xclim.core.units import convert_units_to, declare_units

# Frequencies : YS: year start, QS-DEC: seasons starting in december, MS: month start
# See https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html

# -------------------------------------------------- #
# ATTENTION: ASSUME ALL INDICES WRONG UNTIL TESTED ! #
# -------------------------------------------------- #

__all__ = [
    "jetstream_metric_woolings",
]


@declare_units(ua="[speed]")
def jetstream_metric_woolings(
    ua: xarray.DataArray,
) -> xarray.DataArray:
    """Strength and latitude of jetstream.

    Identify latitude and strength of maximum smoothed zonal wind speed in the region from 15 to 75°N and -60 to 0°E.

    Parameters
    ----------
    ua : xarray.DataArray
      eastward wind component (u) at between 750 and 950 hPa.

    Returns
    -------
    xarray.DataArray
      Daily time series of latitude of jetstream.
    xarray.DataArray
      Daily time series of strength of jetstream.

    References
    ----------
    .. [woollings2010] Woollings, T., Hannachi, A., & Hoskins, B. (2010). Variability of the North Atlantic eddy‐driven jet stream. Quarterly Journal of the Royal Meteorological Society, 136(649), 856-868.

    """

    # get latitude & eastward wind component units
    lat_units = ua["lat"].units
    ua_units = ua.units

    # select only relevant hPa levels, compute zonal mean windspeed
    pmin = convert_units_to("750 hPa", ua.plev)
    pmax = convert_units_to("950 hPa", ua.plev)

    ua = ua.sel(plev=slice(pmin, pmax), lat=slice(15, 75), lon=slice(-60, 0))

    zonal_mean = ua.mean(["plev", "lon"])

    # apply Lanczos filter - parameters are hard-coded following those used in Woollings (2010)
    filter_freq = 10
    window_size = 61
    cutoff = 1 / filter_freq

    if ua["time"].count() <= filter_freq or ua["time"].count() <= window_size:
        print("Time series is too short to apply 61-day Lanczos filter")
        return

    # compute low-pass filter weights
    lanczos_weights = compute_low_pass_filter_weights(
        window_size=window_size, cutoff=cutoff
    )
    # apply the filter
    ua_lf = (
        zonal_mean.rolling(time=window_size, center=True)
        .construct("window")
        .dot(lanczos_weights)
    )

    jetlat = ua_lf.idxmax("lat").rename("jetlat").assign_attrs(units=lat_units)
    jetstr = ua_lf.max("lat").rename("jetstr").assign_attrs(units=ua_units)

    return jetlat, jetstr


def compute_low_pass_filter_weights(
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

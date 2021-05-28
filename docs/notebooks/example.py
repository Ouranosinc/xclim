import xarray as xr

from xclim.core.units import declare_units, rate2amount


@declare_units(pr="[precipitation]")
def extreme_precip_accumulation(pr: xr.DataArray, perc: float = 95, freq: str = "YS"):
    """Total precipitation accumulation during extreme events.

    The `perc` percentile of the precipitation (including all values, not in a day-of-year manner)
    is computed. Then, for each period, the days where `pr` is above the threshold are accumulated,
    to get the total precip related to those extreme events.

    Parameters
    ----------
    pr: xr.DataArray
      Precipitation flux (both phases).
    perc: float
      Percentile corresponding to "extreme" precipitation, [0-100].
    freq: str
      Resampling frequency.

    Returns
    -------
    xarray.DataArray
      Precipitation accumulated during events where pr was above the {perc}th percentile of the whole series.
    """
    pr_thresh = pr.quantile(perc / 100, dim="time")

    pr_extreme = rate2amount(pr).where(pr >= pr_thresh)

    out = pr_extreme.resample(time=freq).sum()
    out.attrs["units"] = pr_extreme.units
    return out

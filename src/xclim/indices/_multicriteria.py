"""Multicriteria indices for climate data analysis."""

import numpy as np
import xarray

from xclim.core.calendar import select_time
from xclim.core.units import convert_units_to
from xclim.indices._simple import tn_mean, tx_mean
from xclim.indices._threshold import frost_free_season_length

__all__ = ["canadian_hardiness_zones"]


def canadian_hardiness_zones(
    tasmin: xarray.DataArray,
    tasmax: xarray.DataArray,
    pr: xarray.DataArray,
    snd: xarray.DataArray,
    sfcWindmax: xarray.DataArray,
    freq: str = "YS",
) -> xarray.DataArray:
    """
    Canadian hardiness zones.

    Parameters
    ----------
    tasmin : xarray.DataArray
        Minimum daily temperature.
    tasmax : xarray.DataArray
        Maximum daily temperature.
    pr : xarray.DataArray
        Precipitation.
    snd : xarray.DataArray
        Snow depth.
    sfcWindmax : xarray.DataArray
        Maximum surface wind speed.
    freq : str
        Resampling frequency for the input data, by default "YS" (yearly start).

    Returns
    -------
    xarray.DataArray
        Canadian hardiness zones.

    Notes
    -----
    This index is based on the Canadian hardiness zones, which are used to classify regions based on their climate
    suitability for growing plants. The index is calculated using a combination of temperature, precipitation,
    snow depth, and wind speed data. The formula is adapted from the Canadian hardiness zones index as described in
    :cite:t:`ouellet_hardiness_1967b` and :cite:t:`ouellet_hardiness_1967c`.

    References
    ----------
    :cite:cts:`ouellet_hardiness_1967b,ouellet_hardiness_1967c`
    """
    # FIXME: This index expects the input data to be 30-year climatological averages.
    if not isinstance(freq, str):
        raise TypeError("Freq must be a string.")

    _tasmin = convert_units_to(tasmin, "degC")
    _tasmax = convert_units_to(tasmax, "degC")
    _pr = convert_units_to(pr, "mm")
    _snd = convert_units_to(snd, "mm")
    _sfcWindmax = convert_units_to(sfcWindmax, "km h-1")

    # Monthly mean of minimum temperatures of the coldest month
    x1 = tn_mean(tasmin=_tasmin, freq="MS").resample(time=freq).min()

    # Length of the frost free period (FFP)
    x2 = frost_free_season_length(tasmin=_tasmin, op=">", mid_date=None, freq=freq)

    # Precipitation in the period from June to November, inclusive
    _pr_constrained = (
        select_time(_pr, date_bounds=("06-01", "12-01"), include_bounds=(True, False)).resample(time=freq).sum()
    )
    # Empirical adjustment for millimeters of precipitation
    x3 = _pr_constrained / (_pr_constrained + 25.4)

    # Monthly mean of maximum temperatures of the warmest month
    x4 = tx_mean(tasmax=_tasmax, freq="MS").resample(time=freq).max()

    # Winter factor
    x5 = (0 - x1) * _pr.sel(time=_pr["time"].dt.month == 1).resample(time=freq).sum()

    # Mean maximum snow depth
    snd_above_0 = _snd.where(_snd > 0, np.nan).resample(time=freq).mean()
    x6 = snd_above_0 / (snd_above_0 + 25.4)

    # Maximum wind gust in experienced in a 30-year period
    x7 = sfcWindmax.resample(time=freq).max()

    suitability_index = (
        -67.62
        + (1.734 * x1)
        + (0.1868 * x2)
        + (69.77 * x3)
        + (1.256 * x4)
        + (0.006119 * x5)
        + (22.37 * x6)
        - (0.01832 * x7)
    )
    suitability_index.attrs["long_name"] = "Canadian hardiness zones."
    suitability_index.attrs["units"] = ""

    return suitability_index

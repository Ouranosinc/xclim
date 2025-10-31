"""Simple indice definitions."""

from __future__ import annotations

import xarray

from xclim.core import Quantified
from xclim.core.units import declare_units
from xclim.indices.generic import count_occurrences, statistics

# Frequencies : YS: year start, QS-DEC: seasons starting in december, MS: month start
# See http://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases

# -------------------------------------------------- #
# ATTENTION: ASSUME ALL INDICES WRONG UNTIL TESTED ! #
# -------------------------------------------------- #

__all__ = [
    "frost_days",
    "hot_days",
    "ice_days",
    "sfcWind_max",
    "sfcWind_mean",
    "sfcWind_min",
    "sfcWindmax_max",
    "sfcWindmax_mean",
    "sfcWindmax_min",
    "snow_depth",
    "tg_max",
    "tg_mean",
    "tg_min",
    "tn_max",
    "tn_mean",
    "tn_min",
    "tx_max",
    "tx_mean",
    "tx_min",
]


@declare_units(tas="[temperature]")
def tg_max(tas: xarray.DataArray, freq: str = "YS") -> xarray.DataArray:
    r"""
    Highest mean temperature.

    The maximum of daily mean temperature.

    Parameters
    ----------
    tas : xarray.DataArray
        Mean daily temperature.
    freq : str
        Resampling frequency.

    Returns
    -------
    xarray.DataArray, [same units as tas]
        Maximum of daily mean temperature.

    Notes
    -----
    Let :math:`TN_{ij}` be the mean temperature at day :math:`i` of period :math:`j`. Then the maximum
    daily mean temperature for period :math:`j` is:

    .. math::

       TNx_j = max(TN_{ij})
    """
    return tas.resample(time=freq).max(dim="time").assign_attrs(units=tas.units)


@declare_units(tas="[temperature]")
def tg_mean(tas: xarray.DataArray, freq: str = "YS") -> xarray.DataArray:
    r"""
    Mean of daily average temperature.

    Resample the original daily mean temperature series by taking the mean over each period.

    Parameters
    ----------
    tas : xarray.DataArray
        Mean daily temperature.
    freq : str
        Resampling frequency.

    Returns
    -------
    xarray.DataArray, [same units as tas]
        The mean daily temperature at the given time frequency.

    Notes
    -----
    Let :math:`TN_i` be the mean daily temperature of day :math:`i`, then for a period :math:`p` starting at
    day :math:`a` and finishing on day :math:`b`:

    .. math::

       TG_p = \frac{\sum_{i=a}^{b} TN_i}{b - a + 1}

    Examples
    --------
    The following would compute for each grid cell of file `tas.day.nc` the mean temperature
    at the seasonal frequency, i.e. DJF, MAM, JJA, SON, DJF, etc.:

    >>> from xclim.indices import tg_mean
    >>> t = xr.open_dataset(path_to_tas_file).tas
    >>> tg = tg_mean(t, freq="QS-DEC")
    """
    return statistics(tas, statistic="mean", freq=freq)


@declare_units(tas="[temperature]")
def tg_min(tas: xarray.DataArray, freq: str = "YS") -> xarray.DataArray:
    r"""
    Lowest mean temperature.

    Minimum of daily mean temperature.

    Parameters
    ----------
    tas : xarray.DataArray
        Mean daily temperature.
    freq : str
        Resampling frequency.

    Returns
    -------
    xarray.DataArray, [same units as tas]
        Minimum of daily minimum temperature.

    Notes
    -----
    Let :math:`TG_{ij}` be the mean temperature at day :math:`i` of period :math:`j`. Then the minimum
    daily mean temperature for period :math:`j` is:

    .. math::

       TGn_j = min(TG_{ij})
    """
    return statistics(tas, statistic="min", freq=freq)


@declare_units(tasmin="[temperature]")
def tn_max(tasmin: xarray.DataArray, freq: str = "YS") -> xarray.DataArray:
    r"""
    Highest minimum temperature.

    The maximum of daily minimum temperature.

    Parameters
    ----------
    tasmin : xarray.DataArray
        Minimum daily temperature.
    freq : str
        Resampling frequency.

    Returns
    -------
    xarray.DataArray, [same units as tasmin]
        Maximum of daily minimum temperature.

    Notes
    -----
    Let :math:`TN_{ij}` be the minimum temperature at day :math:`i` of period :math:`j`. Then the maximum
    daily minimum temperature for period :math:`j` is:

    .. math::

       TNx_j = max(TN_{ij})
    """
    return statistics(tasmin, statistic="max", freq=freq)


@declare_units(tasmin="[temperature]")
def tn_mean(tasmin: xarray.DataArray, freq: str = "YS") -> xarray.DataArray:
    r"""
    Mean minimum temperature.

    Mean of daily minimum temperature.

    Parameters
    ----------
    tasmin : xarray.DataArray
        Minimum daily temperature.
    freq : str
        Resampling frequency.

    Returns
    -------
    xarray.DataArray, [same units as tasmin]
        Mean of daily minimum temperature.

    Notes
    -----
    Let :math:`TN_{ij}` be the minimum temperature at day :math:`i` of period :math:`j`. Then mean
    values in period :math:`j` are given by:

    .. math::

       TN_{ij} = \frac{ \sum_{i=1}^{I} TN_{ij} }{I}
    """
    return statistics(tasmin, statistic="mean", freq=freq)


@declare_units(tasmin="[temperature]")
def tn_min(tasmin: xarray.DataArray, freq: str = "YS") -> xarray.DataArray:
    r"""
    Lowest minimum temperature.

    Minimum of daily minimum temperature.

    Parameters
    ----------
    tasmin : xarray.DataArray
        Minimum daily temperature.
    freq : str
        Resampling frequency.

    Returns
    -------
    xarray.DataArray, [same units as tasmin]
        Minimum of daily minimum temperature.

    Notes
    -----
    Let :math:`TN_{ij}` be the minimum temperature at day :math:`i` of period :math:`j`. Then the minimum
    daily minimum temperature for period :math:`j` is:

    .. math::

       TNn_j = min(TN_{ij})
    """
    return statistics(tasmin, statistic="min", freq=freq)


@declare_units(tasmax="[temperature]")
def tx_max(tasmax: xarray.DataArray, freq: str = "YS") -> xarray.DataArray:
    r"""
    Highest max temperature.

    The maximum value of daily maximum temperature.

    Parameters
    ----------
    tasmax : xarray.DataArray
        Maximum daily temperature.
    freq : str
        Resampling frequency.

    Returns
    -------
    xarray.DataArray, [same units as tasmax]
        Maximum value of daily maximum temperature.

    Notes
    -----
    Let :math:`TX_{ij}` be the maximum temperature at day :math:`i` of period :math:`j`. Then the maximum
    daily maximum temperature for period :math:`j` is:

    .. math::

       TXx_j = max(TX_{ij})
    """
    return statistics(tasmax, statistic="max", freq=freq)


@declare_units(tasmax="[temperature]")
def tx_mean(tasmax: xarray.DataArray, freq: str = "YS") -> xarray.DataArray:
    r"""
    Mean max temperature.

    The mean of daily maximum temperature.

    Parameters
    ----------
    tasmax : xarray.DataArray
        Maximum daily temperature.
    freq : str
        Resampling frequency.

    Returns
    -------
    xarray.DataArray, [same units as tasmax]
        Mean of daily maximum temperature.

    Notes
    -----
    Let :math:`TX_{ij}` be the maximum temperature at day :math:`i` of period :math:`j`. Then mean
    values in period :math:`j` are given by:

    .. math::

       TX_{ij} = \frac{ \sum_{i=1}^{I} TX_{ij} }{I}
    """
    return statistics(tasmax, statistic="mean", freq=freq)


@declare_units(tasmax="[temperature]")
def tx_min(tasmax: xarray.DataArray, freq: str = "YS") -> xarray.DataArray:
    r"""
    Lowest max temperature.

    The minimum of daily maximum temperature.

    Parameters
    ----------
    tasmax : xarray.DataArray
        Maximum daily temperature.
    freq : str
        Resampling frequency.

    Returns
    -------
    xarray.DataArray, [same units as tasmax]
      Minimum of daily maximum temperature.

    Notes
    -----
    Let :math:`TX_{ij}` be the maximum temperature at day :math:`i` of period :math:`j`. Then the minimum
    daily maximum temperature for period :math:`j` is:

    .. math::

       TXn_j = min(TX_{ij})
    """
    return statistics(tasmax, statistic="min", freq=freq)


@declare_units(tasmax="[temperature]", thresh="[temperature]")
def hot_days(
    tasmax: xarray.DataArray,
    thresh: Quantified = "25 degC",
    freq: str = "YS",
) -> xarray.DataArray:
    r"""
    Hot days index.

    Number of days where daily maximum temperatures are above a threshold temperature.

    Parameters
    ----------
    tasmax : xarray.DataArray
        Maximum daily temperature.
    thresh : Quantified
        Threshold temperature.
    freq : str
        Resampling frequency.

    Returns
    -------
    xarray.DataArray, [time]
        Hot days index.

    Notes
    -----
    Let :math:`TX_{ij}` be the daily maximum temperature at day :math:`i` of period :math:`j`
    and :math`TT` the threshold. Then counted is the number of days where:

    .. math::

       TX_{ij} > TT
    """
    return count_occurrences(tasmax, condition=">", thresh=thresh, freq=freq)


@declare_units(tasmin="[temperature]", thresh="[temperature]")
def frost_days(
    tasmin: xarray.DataArray,
    thresh: Quantified = "0 degC",
    freq: str = "YS",
) -> xarray.DataArray:
    r"""
    Frost days index.

    Number of days where daily minimum temperatures are below a threshold temperature.

    Parameters
    ----------
    tasmin : xarray.DataArray
        Minimum daily temperature.
    thresh : Quantified
        Freezing temperature.
    freq : str
        Resampling frequency.

    Returns
    -------
    xarray.DataArray, [time]
        Frost days index.

    Notes
    -----
    Let :math:`TN_{ij}` be the daily minimum temperature at day :math:`i` of period :math:`j`
    and :math`TT` the threshold. Then counted is the number of days where:

    .. math::

       TN_{ij} < TT
    """
    return count_occurrences(tasmin, condition="<", thresh=thresh, freq=freq)


@declare_units(tasmax="[temperature]", thresh="[temperature]")
def ice_days(tasmax: xarray.DataArray, thresh: Quantified = "0 degC", freq: str = "YS") -> xarray.DataArray:
    r"""
    Number of ice/freezing days.

    Number of days when daily maximum temperatures are below a threshold.

    Parameters
    ----------
    tasmax : xarray.DataArray
        Maximum daily temperature.
    thresh : Quantified
        Freezing temperature.
    freq : str
        Resampling frequency.

    Returns
    -------
    xarray.DataArray, [time]
        Number of ice/freezing days.

    Notes
    -----
    Let :math:`TX_{ij}` be the daily maximum temperature at day :math:`i` of period :math:`j`,
    and :math`TT` the threshold. Then counted is the number of days where:

    .. math::

       TX_{ij} < TT
    """
    return count_occurrences(tasmax, condition="<", thresh=thresh, freq=freq)


@declare_units(snd="[length]")
def snow_depth(
    snd: xarray.DataArray,
    freq: str = "YS",
) -> xarray.DataArray:
    """
    Mean of daily average snow depth.

    Resample the original daily mean snow depth series by taking the mean over each period.

    Parameters
    ----------
    snd : xarray.DataArray
        Mean daily snow depth.
    freq : str
        Resampling frequency.

    Returns
    -------
    xarray.DataArray, [same units as snd]
        The mean daily snow depth at the given time frequency.
    """
    return snd.resample(time=freq).mean(dim="time").assign_attrs(units=snd.units)


@declare_units(sfcWind="[speed]")
def sfcWind_max(  # noqa: N802
    sfcWind: xarray.DataArray, freq: str = "YS"
) -> xarray.DataArray:
    r"""
    Highest daily mean wind speed.

    The maximum of daily mean wind speed.

    Parameters
    ----------
    sfcWind : xarray.DataArray
        Mean daily wind speed.
    freq : str
        Resampling frequency.

    Returns
    -------
    xarray.DataArray, [same units as sfcWind]
        Maximum of daily mean wind speed.

    Notes
    -----
    Let :math:`FG_{ij}` be the mean wind speed at day :math:`i` of period :math:`j`. Then the maximum
    daily mean wind speed for period :math:`j` is:

    .. math::

       FGx_j = max(FG_{ij})

    Examples
    --------
    The following would compute for each grid cell the maximum wind speed
    at the seasonal frequency, i.e. DJF, MAM, JJA, SON, DJF, etc.:

    >>> from xclim.indices import sfcWind_max
    >>> fg = xr.open_dataset(path_to_sfcWind_file).sfcWind
    >>> fg_max = sfcWind_max(fg, freq="QS-DEC")
    """
    return sfcWind.resample(time=freq).max(dim="time").assign_attrs(units=sfcWind.units)


@declare_units(sfcWind="[speed]")
def sfcWind_mean(  # noqa: N802
    sfcWind: xarray.DataArray, freq: str = "YS"
) -> xarray.DataArray:
    r"""
    Mean of daily mean wind speed.

    Resample the original daily mean wind speed series by taking the mean over each period.

    Parameters
    ----------
    sfcWind : xarray.DataArray
        Mean daily wind speed.
    freq : str
        Resampling frequency.

    Returns
    -------
    xarray.DataArray, [same units as sfcWind]
        The mean daily wind speed at the given time frequency.

    Notes
    -----
    Let :math:`FG_i` be the mean wind speed of day :math:`i`, then for a period :math:`p` starting at
    day :math:`a` and finishing on day :math:`b`:

    .. math::

       FG_m = \frac{\sum_{i=a}^{b} FG_i}{b - a + 1}

    Examples
    --------
    The following would compute for each grid cell the mean wind speed
    at the seasonal frequency, i.e. DJF, MAM, JJA, SON, DJF, etc.:

    >>> from xclim.indices import sfcWind_mean
    >>> fg = xr.open_dataset(path_to_sfcWind_file).sfcWind
    >>> fg_mean = sfcWind_mean(fg, freq="QS-DEC")
    """
    return sfcWind.resample(time=freq).mean(dim="time").assign_attrs(units=sfcWind.units)


@declare_units(sfcWind="[speed]")
def sfcWind_min(  # noqa: N802
    sfcWind: xarray.DataArray, freq: str = "YS"
) -> xarray.DataArray:
    r"""
    Lowest daily mean wind speed.

    The minimum of daily mean wind speed.

    Parameters
    ----------
    sfcWind : xarray.DataArray
        Mean daily wind speed.
    freq : str
        Resampling frequency.

    Returns
    -------
    xarray.DataArray, [same units as sfcWind]
        Minimum of daily mean wind speed.

    Notes
    -----
    Let :math:`FG_{ij}` be the mean wind speed at day :math:`i` of period :math:`j`. Then the minimum
    daily mean wind speed for period :math:`j` is:

    .. math::

       FGn_j = min(FG_{ij})

    Examples
    --------
    The following would compute for each grid cell the minimum wind speed
    at the seasonal frequency, i.e. DJF, MAM, JJA, SON, DJF, etc.:

    >>> from xclim.indices import sfcWind_min
    >>> fg = xr.open_dataset(path_to_sfcWind_file).sfcWind
    >>> fg_min = sfcWind_min(fg, freq="QS-DEC")
    """
    return sfcWind.resample(time=freq).min(dim="time").assign_attrs(units=sfcWind.units)


@declare_units(sfcWindmax="[speed]")
def sfcWindmax_max(  # noqa: N802
    sfcWindmax: xarray.DataArray, freq: str = "YS"
) -> xarray.DataArray:
    r"""
    Highest maximum wind speed.

    The maximum of daily maximum wind speed.

    Parameters
    ----------
    sfcWindmax : xarray.DataArray
        Maximum daily wind speed.
    freq : str
        Resampling frequency.

    Returns
    -------
    xarray.DataArray, [same units as sfcWindmax]
        Maximum value of daily maximum wind speed.

    Notes
    -----
    Let :math:`FX_{ij}` be the maximum wind speed at day :math:`i` of period :math:`j`. Then the maximum
    daily maximum wind speed for period :math:`j` is:

    .. math::

       FXx_j = max(FX_{ij})

    Examples
    --------
    The following would compute for each grid cell of the dataset the extreme maximum wind speed
    at the seasonal frequency, i.e. DJF, MAM, JJA, SON, DJF, etc.:

    >>> from xclim.indices import sfcWindmax_max
    >>> max_sfcWindmax = sfcWindmax_max(sfcWindmax_dataset, freq="QS-DEC")
    """
    return sfcWindmax.resample(time=freq).max(dim="time").assign_attrs(units=sfcWindmax.units)


@declare_units(sfcWindmax="[speed]")
def sfcWindmax_mean(  # noqa: N802
    sfcWindmax: xarray.DataArray, freq: str = "YS"
) -> xarray.DataArray:
    r"""
    Mean of daily maximum wind speed.

    Resample the original daily maximum wind speed series by taking the mean over each period.

    Parameters
    ----------
    sfcWindmax : xarray.DataArray
        Maximum daily wind speed.
    freq : str
        Resampling frequency.

    Returns
    -------
    xarray.DataArray, [same units as sfcWindmax]
        The mean daily maximum wind speed at the given time frequency.

    Notes
    -----
    Let :math:`FX_i` be the maximum wind speed of day :math:`i`, then for a period :math:`p` starting at
    day :math:`a` and finishing on day :math:`b`:

    .. math::

       FX_m = \frac{\sum_{i=a}^{b} FX_i}{b - a + 1}

    Examples
    --------
    The following would compute for each grid cell of the dataset the mean of maximum wind speed
    at the seasonal frequency, i.e. DJF, MAM, JJA, SON, DJF, etc.:

    >>> from xclim.indices import sfcWindmax_mean
    >>> mean_sfcWindmax = sfcWindmax_mean(sfcWindmax_dataset, freq="QS-DEC")
    """
    return sfcWindmax.resample(time=freq).mean(dim="time").assign_attrs(units=sfcWindmax.units)


@declare_units(sfcWindmax="[speed]")
def sfcWindmax_min(  # noqa: N802
    sfcWindmax: xarray.DataArray, freq: str = "YS"
) -> xarray.DataArray:
    r"""
    Lowest daily maximum wind speed.

    The minimum of daily maximum wind speed.

    Parameters
    ----------
    sfcWindmax : xarray.DataArray
        Maximum daily wind speed.
    freq : str
        Resampling frequency.

    Returns
    -------
    xarray.DataArray, [same units as sfcWindmax]
        Minimum of daily maximum wind speed.

    Notes
    -----
    Let :math:`FX_{ij}` be the maximum wind speed at day :math:`i` of period :math:`j`.
    Then the minimum daily maximum wind speed for period :math:`j` is:

    .. math::

       FXn_j = min(FX_{ij})

    Examples
    --------
    The following would compute for each grid cell of the dataset the minimum of maximum wind speed
    at the seasonal frequency, i.e. DJF, MAM, JJA, SON, DJF, etc.:

    >>> from xclim.indices import sfcWindmax_min
    >>> min_sfcWindmax = sfcWindmax_min(sfcWindmax_dataset, freq="QS-DEC")
    """
    return sfcWindmax.resample(time=freq).min(dim="time").assign_attrs(units=sfcWindmax.units)

# noqa: D100
from __future__ import annotations

import xarray

from xclim.core.units import convert_units_to, declare_units, rate2amount, to_agg_units
from xclim.core.utils import Quantified

from .generic import select_time, threshold_count

# Frequencies : YS: year start, QS-DEC: seasons starting in december, MS: month start
# See http://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases

# -------------------------------------------------- #
# ATTENTION: ASSUME ALL INDICES WRONG UNTIL TESTED ! #
# -------------------------------------------------- #

__all__ = [
    "frost_days",
    "ice_days",
    "max_1day_precipitation_amount",
    "max_n_day_precipitation_amount",
    "max_pr_intensity",
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
    r"""Highest mean temperature.

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
    r"""Mean of daily average temperature.

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
        The mean daily temperature at the given time frequency

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
    return tas.resample(time=freq).mean(dim="time").assign_attrs(units=tas.units)


@declare_units(tas="[temperature]")
def tg_min(tas: xarray.DataArray, freq: str = "YS") -> xarray.DataArray:
    r"""Lowest mean temperature.

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
    return tas.resample(time=freq).min(dim="time").assign_attrs(units=tas.units)


@declare_units(tasmin="[temperature]")
def tn_max(tasmin: xarray.DataArray, freq: str = "YS") -> xarray.DataArray:
    r"""Highest minimum temperature.

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
    return tasmin.resample(time=freq).max(dim="time").assign_attrs(units=tasmin.units)


@declare_units(tasmin="[temperature]")
def tn_mean(tasmin: xarray.DataArray, freq: str = "YS") -> xarray.DataArray:
    r"""Mean minimum temperature.

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
    return tasmin.resample(time=freq).mean(dim="time").assign_attrs(units=tasmin.units)


@declare_units(tasmin="[temperature]")
def tn_min(tasmin: xarray.DataArray, freq: str = "YS") -> xarray.DataArray:
    r"""Lowest minimum temperature.

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
    return tasmin.resample(time=freq).min(dim="time").assign_attrs(units=tasmin.units)


@declare_units(tasmax="[temperature]")
def tx_max(tasmax: xarray.DataArray, freq: str = "YS") -> xarray.DataArray:
    r"""Highest max temperature.

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
    return tasmax.resample(time=freq).max(dim="time").assign_attrs(units=tasmax.units)


@declare_units(tasmax="[temperature]")
def tx_mean(tasmax: xarray.DataArray, freq: str = "YS") -> xarray.DataArray:
    r"""Mean max temperature.

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
    return tasmax.resample(time=freq).mean(dim="time").assign_attrs(units=tasmax.units)


@declare_units(tasmax="[temperature]")
def tx_min(tasmax: xarray.DataArray, freq: str = "YS") -> xarray.DataArray:
    r"""Lowest max temperature.

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
    return tasmax.resample(time=freq).min(dim="time").assign_attrs(units=tasmax.units)


@declare_units(tasmin="[temperature]", thresh="[temperature]")
def frost_days(
    tasmin: xarray.DataArray, thresh: Quantified = "0 degC", freq: str = "YS", **indexer
) -> xarray.DataArray:
    r"""Frost days index.

    Number of days where daily minimum temperatures are below a threshold temperature.

    Parameters
    ----------
    tasmin : xarray.DataArray
        Minimum daily temperature.
    thresh : Quantified
        Freezing temperature.
    freq : str
        Resampling frequency.
    indexer
        Indexing parameters to compute the frost days on a temporal subset of the data.
        It accepts the same arguments as :py:func:`xclim.indices.generic.select_time`.

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
    frz = convert_units_to(thresh, tasmin)
    sel = select_time(tasmin, **indexer)
    out = threshold_count(sel, "<", frz, freq)
    return to_agg_units(out, tasmin, "count")


@declare_units(tasmax="[temperature]", thresh="[temperature]")
def ice_days(
    tasmax: xarray.DataArray, thresh: Quantified = "0 degC", freq: str = "YS"
) -> xarray.DataArray:
    r"""Number of ice/freezing days.

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
    frz = convert_units_to(thresh, tasmax)
    out = threshold_count(tasmax, "<", frz, freq)
    return to_agg_units(out, tasmax, "count")


@declare_units(pr="[precipitation]")
def max_1day_precipitation_amount(
    pr: xarray.DataArray, freq: str = "YS"
) -> xarray.DataArray:
    r"""Highest 1-day precipitation amount for a period (frequency).

    Resample the original daily total precipitation temperature series by taking the max over each period.

    Parameters
    ----------
    pr : xarray.DataArray
        Daily precipitation values.
    freq : str
        Resampling frequency.

    Returns
    -------
    xarray.DataArray, [same units as pr]
        The highest 1-period precipitation flux value at the given time frequency.

    Notes
    -----
    Let :math:`PR_i` be the mean daily precipitation of day `i`, then for a period `j`:

    .. math::

       PRx_{ij} = max(PR_{ij})

    Examples
    --------
    The following would compute for each grid cell the highest 1-day total at an annual frequency:

    >>> from xclim.indices import max_1day_precipitation_amount
    >>> pr = xr.open_dataset(path_to_pr_file).pr
    >>> rx1day = max_1day_precipitation_amount(pr, freq="YS")
    """
    return pr.resample(time=freq).max(dim="time").assign_attrs(units=pr.units)


@declare_units(pr="[precipitation]")
def max_n_day_precipitation_amount(
    pr: xarray.DataArray, window: int = 1, freq: str = "YS"
) -> xarray.DataArray:
    r"""Highest precipitation amount cumulated over a n-day moving window.

    Calculate the n-day rolling sum of the original daily total precipitation series
    and determine the maximum value over each period.

    Parameters
    ----------
    pr : xarray.DataArray
        Daily precipitation values.
    window : int
        Window size in days.
    freq : str
        Resampling frequency.

    Returns
    -------
    xarray.DataArray, [length]
        The highest cumulated n-period precipitation value at the given time frequency.

    Examples
    --------
    The following would compute for each grid cell the highest 5-day total precipitation at an annual frequency:

    >>> from xclim.indices import max_n_day_precipitation_amount
    >>> pr = xr.open_dataset(path_to_pr_file).pr
    >>> out = max_n_day_precipitation_amount(pr, window=5, freq="YS")
    """
    # Rolling sum of the values
    pram = rate2amount(pr)
    arr = pram.rolling(time=window).sum(skipna=False)
    return arr.resample(time=freq).max(dim="time").assign_attrs(units=pram.units)


@declare_units(pr="[precipitation]")
def max_pr_intensity(
    pr: xarray.DataArray, window: int = 1, freq: str = "YS"
) -> xarray.DataArray:
    r"""Highest precipitation intensity over a n-hour moving window.

    Calculate the n-hour rolling average of the original hourly total precipitation series
    and determine the maximum value over each period.

    Parameters
    ----------
    pr : xarray.DataArray
        Hourly precipitation values.
    window : int
        Window size in hours.
    freq : str
        Resampling frequency.

    Returns
    -------
    xarray.DataArray, [same units as pr]
        The highest cumulated n-hour precipitation intensity at the given time frequency.

    Examples
    --------
    The following would compute the maximum 6-hour precipitation intensity at an annual frequency:

    >>> from xclim.indices import max_pr_intensity
    >>> pr = xr.open_dataset(path_to_pr_file).pr
    >>> out = max_pr_intensity(pr, window=5, freq="YS")
    """
    # Rolling sum of the values
    arr = pr.rolling(time=window).mean(skipna=False)
    out = arr.resample(time=freq).max(dim="time")

    out.attrs["units"] = pr.units
    return out


@declare_units(snd="[length]")
def snow_depth(
    snd: xarray.DataArray,
    freq: str = "YS",
) -> xarray.DataArray:
    """Mean of daily average snow depth.

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
        The mean daily snow depth at the given time frequency
    """
    return snd.resample(time=freq).mean(dim="time").assign_attrs(units=snd.units)


@declare_units(sfcWind="[speed]")
def sfcWind_max(sfcWind: xarray.DataArray, freq: str = "YS") -> xarray.DataArray:
    r"""Highest daily mean wind speed.

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
def sfcWind_mean(sfcWind: xarray.DataArray, freq: str = "YS") -> xarray.DataArray:
    r"""Mean of daily mean wind speed.

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
        The mean daily wind speed at the given time frequency

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
    return (
        sfcWind.resample(time=freq).mean(dim="time").assign_attrs(units=sfcWind.units)
    )


@declare_units(sfcWind="[speed]")
def sfcWind_min(sfcWind: xarray.DataArray, freq: str = "YS") -> xarray.DataArray:
    r"""Lowest daily mean wind speed.

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
def sfcWindmax_max(sfcWindmax: xarray.DataArray, freq: str = "YS") -> xarray.DataArray:
    r"""Highest maximum wind speed.

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
    return (
        sfcWindmax.resample(time=freq)
        .max(dim="time")
        .assign_attrs(units=sfcWindmax.units)
    )


@declare_units(sfcWindmax="[speed]")
def sfcWindmax_mean(sfcWindmax: xarray.DataArray, freq: str = "YS") -> xarray.DataArray:
    r"""Mean of daily maximum wind speed.

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
        The mean daily maximum wind speed at the given time frequency

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
    return (
        sfcWindmax.resample(time=freq)
        .mean(dim="time")
        .assign_attrs(units=sfcWindmax.units)
    )


@declare_units(sfcWindmax="[speed]")
def sfcWindmax_min(sfcWindmax: xarray.DataArray, freq: str = "YS") -> xarray.DataArray:
    r"""Lowest daily maxium wind speed.

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
    Let :math:`FX_{ij}` be the maximum wind speed at day :math:`i` of period :math:`j`. Then the minimum
    daily maximum wind speed for period :math:`j` is:

    .. math::

        FXn_j = min(FX_{ij})

    Examples
    --------
    The following would compute for each grid cell of the dataset the minimum of maximum wind speed
    at the seasonal frequency, i.e. DJF, MAM, JJA, SON, DJF, etc.:

    >>> from xclim.indices import sfcWindmax_min
    >>> min_sfcWindmax = sfcWindmax_min(sfcWindmax_dataset, freq="QS-DEC")
    """
    return (
        sfcWindmax.resample(time=freq)
        .min(dim="time")
        .assign_attrs(units=sfcWindmax.units)
    )

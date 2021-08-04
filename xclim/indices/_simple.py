# noqa: D100
import xarray

from xclim.core.units import convert_units_to, declare_units, rate2amount, to_agg_units

from .generic import threshold_count

# Frequencies : YS: year start, QS-DEC: seasons starting in december, MS: month start
# See http://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases

# -------------------------------------------------- #
# ATTENTION: ASSUME ALL INDICES WRONG UNTIL TESTED ! #
# -------------------------------------------------- #

__all__ = [
    "tg_max",
    "tg_mean",
    "tg_min",
    "tn_max",
    "tn_mean",
    "tn_min",
    "tx_max",
    "tx_mean",
    "tx_min",
    "frost_days",
    "ice_days",
    "max_1day_precipitation_amount",
    "max_n_day_precipitation_amount",
    "max_pr_intensity",
    "snow_depth",
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
      Maximum of daily minimum temperature.

    Notes
    -----
    Let :math:`TN_{ij}` be the mean temperature at day :math:`i` of period :math:`j`. Then the maximum
    daily mean temperature for period :math:`j` is:

    .. math::

        TNx_j = max(TN_{ij})
    """
    return tas.resample(time=freq).max(dim="time", keep_attrs=True)


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
    at the seasonal frequency, ie DJF, MAM, JJA, SON, DJF, etc.:

    >>> from xclim.indices import tg_mean
    >>> t = xr.open_dataset(path_to_tas_file).tas
    >>> tg = tg_mean(t, freq="QS-DEC")
    """
    arr = tas.resample(time=freq)
    return arr.mean(dim="time", keep_attrs=True)


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
    return tas.resample(time=freq).min(dim="time", keep_attrs=True)


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
    return tasmin.resample(time=freq).max(dim="time", keep_attrs=True)


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
    arr = tasmin.resample(time=freq)
    return arr.mean(dim="time", keep_attrs=True)


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
    return tasmin.resample(time=freq).min(dim="time", keep_attrs=True)


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
    return tasmax.resample(time=freq).max(dim="time", keep_attrs=True)


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
    arr = tasmax.resample(time=freq)
    return arr.mean(dim="time", keep_attrs=True)


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
    return tasmax.resample(time=freq).min(dim="time", keep_attrs=True)


@declare_units(tasmin="[temperature]", thresh="[temperature]")
def frost_days(
    tasmin: xarray.DataArray, thresh: str = "0 degC", freq: str = "YS"
) -> xarray.DataArray:
    r"""Frost days index.

    Number of days where daily minimum temperatures are below 0℃.

    Parameters
    ----------
    tasmin : xarray.DataArray
      Minimum daily temperature.
    thresh : str
      Freezing temperature.
    freq : str
      Resampling frequency.

    Returns
    -------
    xarray.DataArray, [time]
      Frost days index.

    Notes
    -----
    Let :math:`TN_{ij}` be the daily minimum temperature at day :math:`i` of period :math:`j`. Then
    counted is the number of days where:

    .. math::

        TN_{ij} < 0℃
    """
    frz = convert_units_to(thresh, tasmin)
    out = threshold_count(tasmin, "<", frz, freq)
    return to_agg_units(out, tasmin, "count")


@declare_units(tasmax="[temperature]", thresh="[temperature]")
def ice_days(
    tasmax: xarray.DataArray, thresh: str = "0 degC", freq: str = "YS"
) -> xarray.DataArray:  # noqa: D401
    r"""Number of ice/freezing days.

    Number of days where daily maximum temperatures are below 0℃.

    Parameters
    ----------
    tasmax : xarray.DataArray
      Maximum daily temperature.
    thresh : str
      Freezing temperature.
    freq : str
      Resampling frequency.

    Returns
    -------
    xarray.DataArray, [time]
      Number of ice/freezing days.

    Notes
    -----
    Let :math:`TX_{ij}` be the daily maximum temperature at day :math:`i` of period :math:`j`. Then
    counted is the number of days where:

    .. math::

        TX_{ij} < 0℃
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
    >>> from xclim.indices import max_1day_precipitation_amount

    # The following would compute for each grid cell the highest 1-day total
    # at an annual frequency:
    >>> pr = xr.open_dataset(path_to_pr_file).pr
    >>> rx1day = max_1day_precipitation_amount(pr, freq="YS")
    """
    return pr.resample(time=freq).max(dim="time", keep_attrs=True)


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
    >>> from xclim.indices import max_n_day_precipitation_amount

    # The following would compute for each grid cell the highest 5-day total precipitation
    #at an annual frequency:
    >>> pr = xr.open_dataset(path_to_pr_file).pr
    >>> out = max_n_day_precipitation_amount(pr, window=5, freq="YS")
    """
    # Rolling sum of the values
    pram = rate2amount(pr)
    arr = pram.rolling(time=window).sum(skipna=False)
    out = arr.resample(time=freq).max(dim="time", keep_attrs=True)

    out.attrs["units"] = pram.units
    return out


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
    >>> from xclim.indices import max_pr_intensity

    # The following would compute the maximum 6-hour precipitation intensity.
    # at an annual frequency:
    # TODO
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
    tas : xarray.DataArray
      Mean daily snow depth.
    freq : str
      Resampling frequency.

    Returns
    -------
    xarray.DataArray, [same units as snd]
      The mean daily snow depth at the given time frequency

    """
    arr = snd.resample(time=freq)
    return arr.mean(dim="time", keep_attrs=True)

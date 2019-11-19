import xarray

from xclim import run_length as rl
from xclim import utils
from xclim.utils import declare_units
from xclim.utils import units

xarray.set_options(enable_cftimeindex=True)  # Set xarray to use cftimeindex


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
    "base_flow_index",
    "consecutive_frost_days",
    "frost_days",
    "ice_days",
    "max_1day_precipitation_amount",
    "max_n_day_precipitation_amount",
]


@declare_units("[temperature]", tas="[temperature]")
def tg_max(tas: xarray.DataArray, freq: str = "YS"):
    r"""Highest mean temperature.

    The maximum of daily mean temperature.

    Parameters
    ----------
    tas : xarray.DataArray
      Mean daily temperature [℃] or [K]
    freq : str
      Resampling frequency; Defaults to "YS" (yearly).

    Returns
    -------
    xarray.DataArray
      Maximum of daily minimum temperature.

    Notes
    -----
    Let :math:`TN_{ij}` be the mean temperature at day :math:`i` of period :math:`j`. Then the maximum
    daily mean temperature for period :math:`j` is:

    .. math::

        TNx_j = max(TN_{ij})
    """

    return tas.resample(time=freq).max(dim="time", keep_attrs=True)


@declare_units("[temperature]", tas="[temperature]")
def tg_mean(tas: xarray.DataArray, freq: str = "YS"):
    r"""Mean of daily average temperature.

    Resample the original daily mean temperature series by taking the mean over each period.

    Parameters
    ----------
    tas : xarray.DataArray
      Mean daily temperature [℃] or [K]
    freq : str
      Resampling frequency; Defaults to "YS" (yearly).

    Returns
    -------
    xarray.DataArray
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
    >>> import xarray as xr
    >>> t = xr.open_dataset('tas.day.nc')
    >>> tg = tm_mean(t, freq="QS-DEC")
    """

    arr = tas.resample(time=freq) if freq else tas
    return arr.mean(dim="time", keep_attrs=True)


@declare_units("[temperature]", tas="[temperature]")
def tg_min(tas: xarray.DataArray, freq: str = "YS"):
    r"""Lowest mean temperature

    Minimum of daily mean temperature.

    Parameters
    ----------
    tas : xarray.DataArray
      Mean daily temperature [℃] or [K]
    freq : str
      Resampling frequency; Defaults to "YS" (yearly).

    Returns
    -------
    xarray.DataArray
      Minimum of daily minimum temperature.

    Notes
    -----
    Let :math:`TG_{ij}` be the mean temperature at day :math:`i` of period :math:`j`. Then the minimum
    daily mean temperature for period :math:`j` is:

    .. math::

        TGn_j = min(TG_{ij})
    """

    return tas.resample(time=freq).min(dim="time", keep_attrs=True)


@declare_units("[temperature]", tasmin="[temperature]")
def tn_max(tasmin: xarray.DataArray, freq: str = "YS"):
    r"""Highest minimum temperature.

    The maximum of daily minimum temperature.

    Parameters
    ----------
    tasmin : xarray.DataArray
      Minimum daily temperature [℃] or [K]
    freq : str
      Resampling frequency; Defaults to "YS" (yearly).

    Returns
    -------
    xarray.DataArray
      Maximum of daily minimum temperature.

    Notes
    -----
    Let :math:`TN_{ij}` be the minimum temperature at day :math:`i` of period :math:`j`. Then the maximum
    daily minimum temperature for period :math:`j` is:

    .. math::

        TNx_j = max(TN_{ij})
    """

    return tasmin.resample(time=freq).max(dim="time", keep_attrs=True)


@declare_units("[temperature]", tasmin="[temperature]")
def tn_mean(tasmin: xarray.DataArray, freq: str = "YS"):
    r"""Mean minimum temperature.

    Mean of daily minimum temperature.

    Parameters
    ----------
    tasmin : xarray.DataArray
      Minimum daily temperature [℃] or [K]
    freq : str
      Resampling frequency; Defaults to "YS" (yearly).

    Returns
    -------
    xarray.DataArray
      Mean of daily minimum temperature.

    Notes
    -----
    Let :math:`TN_{ij}` be the minimum temperature at day :math:`i` of period :math:`j`. Then mean
    values in period :math:`j` are given by:

    .. math::

        TN_{ij} = \frac{ \sum_{i=1}^{I} TN_{ij} }{I}
    """

    arr = tasmin.resample(time=freq) if freq else tasmin
    return arr.mean(dim="time", keep_attrs=True)


@declare_units("[temperature]", tasmin="[temperature]")
def tn_min(tasmin: xarray.DataArray, freq: str = "YS"):
    r"""Lowest minimum temperature

    Minimum of daily minimum temperature.

    Parameters
    ----------
    tasmin : xarray.DataArray
      Minimum daily temperature [℃] or [K]
    freq : str
      Resampling frequency; Defaults to "YS" (yearly).

    Returns
    -------
    xarray.DataArray
      Minimum of daily minimum temperature.

    Notes
    -----
    Let :math:`TN_{ij}` be the minimum temperature at day :math:`i` of period :math:`j`. Then the minimum
    daily minimum temperature for period :math:`j` is:

    .. math::

        TNn_j = min(TN_{ij})
    """

    return tasmin.resample(time=freq).min(dim="time", keep_attrs=True)


@declare_units("[temperature]", tasmax="[temperature]")
def tx_max(tasmax: xarray.DataArray, freq: str = " YS"):
    r"""Highest max temperature

    The maximum value of daily maximum temperature.

    Parameters
    ----------
    tasmax : xarray.DataArray
      Maximum daily temperature [℃] or [K]
    freq : str
      Resampling frequency; Defaults to "YS" (yearly).

    Returns
    -------
    xarray.DataArray
      Maximum value of daily maximum temperature.

    Notes
    -----
    Let :math:`TX_{ij}` be the maximum temperature at day :math:`i` of period :math:`j`. Then the maximum
    daily maximum temperature for period :math:`j` is:

    .. math::

        TXx_j = max(TX_{ij})
    """

    return tasmax.resample(time=freq).max(dim="time", keep_attrs=True)


@declare_units("[temperature]", tasmax="[temperature]")
def tx_mean(tasmax: xarray.DataArray, freq: str = " YS"):
    r"""Mean max temperature

    The mean of daily maximum temperature.

    Parameters
    ----------
    tasmax : xarray.DataArray
      Maximum daily temperature [℃] or [K]
    freq : str
      Resampling frequency; Defaults to "YS" (yearly).

    Returns
    -------
    xarray.DataArray
      Mean of daily maximum temperature.

    Notes
    -----
    Let :math:`TX_{ij}` be the maximum temperature at day :math:`i` of period :math:`j`. Then mean
    values in period :math:`j` are given by:

    .. math::

        TX_{ij} = \frac{ \sum_{i=1}^{I} TX_{ij} }{I}
    """

    arr = tasmax.resample(time=freq) if freq else tasmax
    return arr.mean(dim="time", keep_attrs=True)


@declare_units("[temperature]", tasmax="[temperature]")
def tx_min(tasmax: xarray.DataArray, freq: str = "YS"):
    r"""Lowest max temperature

    The minimum of daily maximum temperature.

    Parameters
    ----------
    tasmax : xarray.DataArray
      Maximum daily temperature [℃] or [K]
    freq : str
      Resampling frequency; Defaults to "YS" (yearly).

    Returns
    -------
    xarray.DataArray
      Minimum of daily maximum temperature.

    Notes
    -----
    Let :math:`TX_{ij}` be the maximum temperature at day :math:`i` of period :math:`j`. Then the minimum
    daily maximum temperature for period :math:`j` is:

    .. math::

        TXn_j = min(TX_{ij})
    """

    return tasmax.resample(time=freq).min(dim="time", keep_attrs=True)


@declare_units("", q="[discharge]")
def base_flow_index(q: xarray.DataArray, freq: str = "YS"):
    r"""Base flow index

    Return the base flow index, defined as the minimum 7-day average flow divided by the mean flow.

    Parameters
    ----------
    q : xarray.DataArray
      Rate of river discharge [m³/s]
    freq : str
      Resampling frequency; Defaults to "YS" (yearly).

    Returns
    -------
    xarray.DataArrray
      Base flow index.

    Notes
    -----
    Let :math:`\mathbf{q}=q_0, q_1, \ldots, q_n` be the sequence of daily discharge and :math:`\overline{\mathbf{q}}`
    the mean flow over the period. The base flow index is given by:

    .. math::

       \frac{\min(\mathrm{CMA}_7(\mathbf{q}))}{\overline{\mathbf{q}}}


    where :math:`\mathrm{CMA}_7` is the seven days moving average of the daily flow:

    .. math::

       \mathrm{CMA}_7(q_i) = \frac{\sum_{j=i-3}^{i+3} q_j}{7}

    """

    m7 = q.rolling(time=7, center=True).mean().resample(time=freq)
    mq = q.resample(time=freq)

    m7m = m7.min(dim="time")
    return m7m / mq.mean(dim="time")


@declare_units("days", tasmin="[temperature]")
def consecutive_frost_days(tasmin: xarray.DataArray, freq="AS-JUL"):
    r"""Maximum number of consecutive frost days (Tmin < 0℃).

    Resample the daily minimum temperature series by computing the maximum number
    of days below the freezing point over each period.

    Parameters
    ----------
    tasmin : xarray.DataArray
      Minimum daily temperature values [℃] or [K]
    freq : str
      Resampling frequency; Defaults to "YS" (yearly).

    Returns
    -------
    xarray.DataArray
      The maximum number of consecutive days below the freezing point.

    Notes
    -----
    Let :math:`\mathbf{x}=x_0, x_1, \ldots, x_n` be a daily minimum temperature series and
    :math:`\mathbf{s}` be the sorted vector of indices :math:`i` where :math:`[p_i < 0\celsius] \neq [p_{i+1} <
    0\celsius]`, that is, the days when the temperature crosses the freezing point.
    Then the maximum number of consecutive frost days is given by

    .. math::

       \max(\mathbf{d}) \quad \mathrm{where} \quad d_j = (s_j - s_{j-1}) [x_{s_j} > 0\celsius]

    where :math:`[P]` is 1 if :math:`P` is true, and 0 if false. Note that this formula does not handle sequences at
    the start and end of the series, but the numerical algorithm does.
    """
    tu = units.parse_units(tasmin.attrs["units"].replace("-", "**-"))
    fu = "degC"
    frz = 0
    if fu != tu:
        frz = units.convert(frz, fu, tu)
    group = (tasmin < frz).resample(time=freq)
    return group.apply(rl.longest_run, dim="time")


@declare_units("days", tasmin="[temperature]")
def frost_days(tasmin: xarray.DataArray, freq: str = "YS"):
    r"""Frost days index

    Number of days where daily minimum temperatures are below 0℃.

    Parameters
    ----------
    tasmin : xarray.DataArray
      Minimum daily temperature [℃] or [K]
    freq : str
      Resampling frequency; Defaults to "YS" (yearly).

    Returns
    -------
    xarray.DataArray
      Frost days index.

    Notes
    -----
    Let :math:`TN_{ij}` be the daily minimum temperature at day :math:`i` of period :math:`j`. Then
    counted is the number of days where:

    .. math::

        TN_{ij} < 0℃
    """
    tu = units.parse_units(tasmin.attrs["units"].replace("-", "**-"))
    fu = "degC"
    frz = 0
    if fu != tu:
        frz = units.convert(frz, fu, tu)
    f = (tasmin < frz) * 1
    return f.resample(time=freq).sum(dim="time")


@declare_units("days", tasmax="[temperature]")
def ice_days(tasmax: xarray.DataArray, freq: str = "YS"):
    r"""Number of ice/freezing days

    Number of days where daily maximum temperatures are below 0℃.

    Parameters
    ----------
    tasmax : xarrray.DataArray
      Maximum daily temperature [℃] or [K]
    freq : str
      Resampling frequency; Defaults to "YS" (yearly).

    Returns
    -------
    xarray.DataArray
      Number of ice/freezing days.

    Notes
    -----
    Let :math:`TX_{ij}` be the daily maximum temperature at day :math:`i` of period :math:`j`. Then
    counted is the number of days where:

    .. math::

        TX_{ij} < 0℃
    """
    tu = units.parse_units(tasmax.attrs["units"].replace("-", "**-"))
    fu = "degC"
    frz = 0
    if fu != tu:
        frz = units.convert(frz, fu, tu)
    f = (tasmax < frz) * 1
    return f.resample(time=freq).sum(dim="time")


@declare_units("mm/day", pr="[precipitation]")
def max_1day_precipitation_amount(pr: xarray.DataArray, freq: str = "YS"):
    r"""Highest 1-day precipitation amount for a period (frequency).

    Resample the original daily total precipitation temperature series by taking the max over each period.

    Parameters
    ----------
    pr : xarray.DataArray
      Daily precipitation values [Kg m-2 s-1] or [mm]
    freq : str
      Resampling frequency; Defaults to "YS" (yearly).

    Returns
    -------
    xarray.DataArray
      The highest 1-day precipitation value at the given time frequency.

    Notes
    -----
    Let :math:`PR_i` be the mean daily precipitation of day `i`, then for a period `j`:

    .. math::

       PRx_{ij} = max(PR_{ij})

    Examples
    --------
    The following would compute for each grid cell of file `pr.day.nc` the highest 1-day total
    at an annual frequency:
    >>> import xarray as xr
    >>> pr = xr.open_dataset('pr.day.nc').pr
    >>> rx1day = max_1day_precipitation_amount(pr, freq="YS")
    """

    out = pr.resample(time=freq).max(dim="time", keep_attrs=True)
    return utils.convert_units_to(out, "mm/day", "hydro")


@declare_units("mm", pr="[precipitation]")
def max_n_day_precipitation_amount(pr, window: int = 1, freq: str = "YS"):
    r"""Highest precipitation amount cumulated over a n-day moving window.

    Calculate the n-day rolling sum of the original daily total precipitation series
    and determine the maximum value over each period.

    Parameters
    ----------
    pr : xarray.DataArray
      Daily precipitation values [Kg m-2 s-1] or [mm]
    window : int
      Window size in days.
    freq : str
      Resampling frequency; Defaults to "YS" (yearly).

    Returns
    -------
    xarray.DataArray
      The highest cumulated n-day precipitation value at the given time frequency.

    Examples
    --------
    The following would compute for each grid cell of file `pr.day.nc` the highest 5-day total precipitation
    at an annual frequency:
    >>> import xarray as xr
    >>> da = xr.open_dataset('pr.day.nc').pr
    >>> window = 5
    >>> output = max_n_day_precipitation_amount(da, window, freq="YS")
    """

    # rolling sum of the values
    arr = pr.rolling(time=window, center=False).sum()
    out = arr.resample(time=freq).max(dim="time", keep_attrs=True)

    out.attrs["units"] = pr.units
    # Adjust values and units to make sure they are daily
    return utils.pint_multiply(out, 1 * units.day, "mm")

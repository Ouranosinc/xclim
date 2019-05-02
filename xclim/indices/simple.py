from xclim.utils import declare_units


@declare_units('[temperature]', tas='[temperature]')
def tg_max(tas, freq='YS'):
    r"""Highest mean temperature.

    The maximum of daily mean temperature.

    Parameters
    ----------
    tas : xarray.DataArray
      Mean daily temperature [℃] or [K]
    freq : str, optional
      Resampling frequency

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

    return tas.resample(time=freq).max(dim='time', keep_attrs=True)


@declare_units('[temperature]', tas='[temperature]')
def tg_mean(tas, freq='YS'):
    r"""Mean of daily average temperature.

    Resample the original daily mean temperature series by taking the mean over each period.

    Parameters
    ----------
    tas : xarray.DataArray
      Mean daily temperature [℃] or [K]
    freq : str, optional
      Resampling frequency

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

    >>> t = xr.open_dataset('tas.day.nc')
    >>> tg = tm_mean(t, freq="QS-DEC")
    """

    arr = tas.resample(time=freq) if freq else tas
    return arr.mean(dim='time', keep_attrs=True)


@declare_units('[temperature]', tas='[temperature]')
def tg_min(tas, freq='YS'):
    r"""Lowest mean temperature

    Minimum of daily mean temperature.

    Parameters
    ----------
    tas : xarray.DataArray
      Mean daily temperature [℃] or [K]
    freq : str, optional
      Resampling frequency

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

    return tas.resample(time=freq).min(dim='time', keep_attrs=True)


@declare_units('[temperature]', tasmin='[temperature]')
def tn_max(tasmin, freq='YS'):
    r"""Highest minimum temperature.

    The maximum of daily minimum temperature.

    Parameters
    ----------
    tasmin : xarray.DataArray
      Minimum daily temperature [℃] or [K]
    freq : str, optional
      Resampling frequency

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

    return tasmin.resample(time=freq).max(dim='time', keep_attrs=True)


@declare_units('[temperature]', tasmin='[temperature]')
def tn_mean(tasmin, freq='YS'):
    r"""Mean minimum temperature.

    Mean of daily minimum temperature.

    Parameters
    ----------
    tasmin : xarray.DataArray
      Minimum daily temperature [℃] or [K]
    freq : str, optional
      Resampling frequency

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
    return arr.mean(dim='time', keep_attrs=True)


@declare_units('[temperature]', tasmin='[temperature]')
def tn_min(tasmin, freq='YS'):
    r"""Lowest minimum temperature

    Minimum of daily minimum temperature.

    Parameters
    ----------
    tasmin : xarray.DataArray
      Minimum daily temperature [℃] or [K]
    freq : str, optional
      Resampling frequency

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

    return tasmin.resample(time=freq).min(dim='time', keep_attrs=True)


@declare_units('[temperature]', tasmax='[temperature]')
def tx_max(tasmax, freq='YS'):
    r"""Highest max temperature

    The maximum value of daily maximum temperature.

    Parameters
    ----------
    tasmax : xarray.DataArray
      Maximum daily temperature [℃] or [K]
    freq : str, optional
      Resampling frequency

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

    return tasmax.resample(time=freq).max(dim='time', keep_attrs=True)


@declare_units('[temperature]', tasmax='[temperature]')
def tx_mean(tasmax, freq='YS'):
    r"""Mean max temperature

    The mean of daily maximum temperature.

    Parameters
    ----------
    tasmax : xarray.DataArray
      Maximum daily temperature [℃] or [K]
    freq : str, optional
      Resampling frequency

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
    return arr.mean(dim='time', keep_attrs=True)


@declare_units('[temperature]', tasmax='[temperature]')
def tx_min(tasmax, freq='YS'):
    r"""Lowest max temperature

    The minimum of daily maximum temperature.

    Parameters
    ----------
    tasmax : xarray.DataArray
      Maximum daily temperature [℃] or [K]
    freq : str, optional
      Resampling frequency

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

    return tasmax.resample(time=freq).min(dim='time', keep_attrs=True)

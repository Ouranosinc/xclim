import xarray

from ._multivariate import daily_temperature_range
from ._multivariate import extreme_temperature_range
from ._multivariate import precip_accumulation
from ._simple import tg_mean
from .run_length import lazy_indexing
from xclim.core.units import convert_units_to
from xclim.core.units import declare_units
from xclim.core.units import pint_multiply
from xclim.core.units import units
from xclim.core.units import units2pint

xarray.set_options(enable_cftimeindex=True)  # Set xarray to use cftimeindex

# Frequencies : YS: year start, QS-DEC: seasons starting in december, MS: month start
# See http://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases

# -------------------------------------------------- #
# ATTENTION: ASSUME ALL INDICES WRONG UNTIL TESTED ! #
# -------------------------------------------------- #

__all__ = [
    "temperature_seasonality",
    "precip_seasonality",
    "tg_mean_warmcold_quarter",
    "tg_mean_wetdry_quarter",
    "prcptot_wetdry_quarter",
    "prcptot_warmcold_quarter",
    "prcptot",
    "prcptot_wetdry_period",
    "isothermality",
]


@declare_units("%", tasmin="[temperature]", tasmax="[temperature]")
def isothermality(tasmin: xarray.DataArray, tasmax: xarray.DataArray, freq: str = "YS"):
    r"""Isothermality

    The mean diurnal range divided by the annual temperature range.

    Parameters
    ----------
    tasmin : xarray.DataArray
      Average daily minimum temperature [℃] or [K] at daily, weekly, or monthly frequency.
    tasmax : xarray.DataArray
      Average daily maximum temperature [℃] or [K] at daily, weekly, or monthly frequency.
    freq : str
      Resampling frequency; Defaults to "YS".

    Returns
    -------
    xarray.DataArray
      The isothermality value expressed as a percent.

    Notes
    -----
    According to the ANUCLIM user-guide https://fennerschool.anu.edu.au/files/anuclim61.pdf (ch. 6), input
    values should be at a weekly (or monthly) frequency.  However, the xclim.indices implementation here will calculate
    the output with input data with daily frequency as well. As such weekly or monthly input values, if desired, should
    be calculated prior to calling the function.
    """

    dtr = daily_temperature_range(tasmin=tasmin, tasmax=tasmax, freq=freq)
    etr = extreme_temperature_range(tasmin=tasmin, tasmax=tasmax, freq=freq)
    with xarray.set_options(keep_attrs=True):
        iso = dtr / etr * 100
        iso.attrs["units"] = "%"
    return iso


@declare_units("%", tas="[temperature]")
def temperature_seasonality(tas: xarray.DataArray):
    r"""ANUCLIM temperature seasonality (coefficient of variation)

    The annual temperature coefficient of variation expressed in percent. Calculated as the standard deviation
    of temperature values for a given year expressed as a percentage of the mean of those temperatures.

    Parameters
    ----------
    tas : xarray.DataArray
      Mean temperature [℃] or [K] at daily, weekly, or monthly frequency.

    Returns
    -------
    xarray.DataArray
      The Coefficient of Variation of mean temperature values expressed in percent.

    Examples
    --------
    The following would compute for each grid cell of file `tas.day.nc` the annual temperature
    temperature seasonality:

    >>> import xclim.indices as xci
    >>> t = xr.open_dataset('tas.day.nc').tas
    >>> tday_seasonality = xci.temperature_seasonality(t)

    >>> t_weekly = xci.tg_mean(t, freq='7D')
    >>> tweek_seasonality = xci.temperature_seasonality(t_weekly)

    Notes
    -----
    For this calculation, the mean in degrees Kelvin is used. This avoids the possibility of having to
    divide by zero, but it does mean that the values are usually quite small.
    See : https://fennerschool.anu.edu.au/files/anuclim61.pdf ch. 6

    According to the ANUCLIM user-guide https://fennerschool.anu.edu.au/files/anuclim61.pdf (ch. 6), input
    values should be at a weekly (or monthly) frequency.  However, the xclim.indices implementation here will calculate
    the result with input data with daily frequency as well. As such weekly or monthly input values, if desired, should be
    calculated prior to calling the function.
    """
    tas = convert_units_to(tas, "K")

    with xarray.set_options(keep_attrs=True):
        seas = 100 * _anuclim_coeff_var(tas)

    seas.attrs["units"] = "%"
    return seas


@declare_units("percent", pr="[precipitation]")
def precip_seasonality(pr: xarray.DataArray,):
    r""" ANUCLIM Precipitation Seasonality (C of V)

    The annual precipitation Coefficient of Variation (C of V) expressed in percent. Calculated as the standard deviation
    of precipitation values for a given year expressed as a percentage of the mean of those values.
    See : https://fennerschool.anu.edu.au/files/anuclim61.pdf ch. 6

    Parameters
    ----------
    pr : xarray.DataArray
      Total precipitation rate at daily, weekly, or monthly frequency.
      Units need to be defined as a rate (e.g. mm d-1, mm week-1).

    Returns
    -------
    xarray.DataArray
      The coefficient of variation of precipitation values.

    Examples
    --------
    The following would compute for each grid cell of file `pr.day.nc` the annual precipitation seasonality:

    >>> import xclim.indices as xci
    >>> p = xr.open_dataset('pr.day.nc')
    >>> pday_seasonality = xci.precip_seasonality(p)

    >>> p_weekly = xci.precip_accumulation(p, freq='7D')
    >>> p_weekly.attrs['units'] = "mm/week" # input units need to be a rate
    >>> pweek_seasonality = xci.precip_seasonality(p_weekly)

    Notes
    -----
    According to the ANUCLIM user-guide https://fennerschool.anu.edu.au/files/anuclim61.pdf (ch. 6), input
    values should be at a weekly (or monthly) frequency.  However, the xclim.indices implementation here will calculate
    the result with input data with daily frequency as well. As such weekly or monthly input values, if desired,
    should be calculated prior to calling the function.

    If input units are in mm s-1 or equivalent values are converted to mm/day to avoid potentially small denominator
    values.

    """

    # If units in mm/sec convert to mm/days to avoid potentially small denominator
    if units2pint(pr) == units("mm / s"):
        pr = convert_units_to(pr, "mm d-1")

    with xarray.set_options(keep_attrs=True):
        seas = 100 * _anuclim_coeff_var(pr)

    seas.attrs["units"] = "percent"
    return seas


@declare_units("[temperature]", tas="[temperature]")
def tg_mean_warmcold_quarter(
    tas: xarray.DataArray, op: str = None, input_freq: str = None, freq: str = "YS",
):
    r"""ANUCLIM Mean temperature of warmest/coldest quarter

    The warmest (or coldest) quarter of the year is determined, and the mean temperature of this period is
    calculated.  If the input data frequency is "daily" or "weekly", quarters are defined as 13 week periods,
    otherwise as 3 months.

    Parameters
    ----------
    tas : xarray.DataArray
      Mean temperature [℃] or [K] at daily, weekly, or monthly frequency.
    op : str
      Operation to perform:  'warmest' calculate warmest quarter; 'coldest' calculate coldest quarter.
    input_freq : str
      Input data time frequency - One of 'daily', 'weekly' or 'monthly'.
    freq : str
      Resampling frequency; Defaults to "YS".

    Returns
    -------
    xarray.DataArray
       mean temperature values of the warmest/coldest quearter of each year.

    Examples
    --------

    The following would compute for each grid cell of file `tas.day.nc` the annual temperature
    warmest quarter mean temperature:

    >>> import xarray as xr
    >>> import xclim.indices as xci
    >>> t = xr.open_dataset('tas.day.nc')
    >>> t_warm_qrt = xci.tg_mean_warmest_quarter(tas=t.tas, op='warmest', input_freq='daily')

    Notes
    -----
    According to the ANUCLIM user-guide https://fennerschool.anu.edu.au/files/anuclim61.pdf (ch. 6), input
    values should be at a weekly (or monthly) frequency.  However, the xclim.indices implementation here will calculate
    the result with input data with daily frequency as well. As such weekly or monthly input values, if desired,
    should be calculated prior to calling the function.

    """
    out = _to_quarter(input_freq, tas=tas)

    with xarray.set_options(keep_attrs=True):
        if op == "warmest":
            out = out.resample(time=freq).max(dim="time")
        elif op == "coldest":
            out = out.resample(time=freq).min(dim="time")
        else:
            raise NotImplementedError(
                f'Unknown operation "{op}" ; op parameter but be one of "warmest" or "coldest"'
            )
        return out


@declare_units("[temperature]", tas="[temperature]", pr="[precipitation]")
def tg_mean_wetdry_quarter(
    tas: xarray.DataArray,
    pr: xarray.DataArray,
    op: str = None,
    input_freq: str = None,
    freq: str = "YS",
):
    r""" ANUCLIM Mean temperature of wettest/dryest quarter

    The wettest (or dryest) quarter of the year is determined, and the mean temperature of this period is calculated.
    If the input data frequency is "daily" or "weekly" quarters are defined as 13 week periods, otherwise are 3 months.

    Parameters
    ----------
    tas : xarray.DataArray
      Mean temperature [℃] or [K] at daily, weekly, or monthly frequency.
    pr : xarray.DataArray
      Total precipitation rate at daily, weekly, or monthly frequency.
    op : str
      Operation to perform: 'wettest' calculate for the wettest quarter; 'dryest' calculate for the dryest quarter.
    input_freq : str
      Input data time frequency - One of 'daily', 'weekly' or 'monthly'.
    freq : str
      Resampling frequency; Defaults to "YS".

    Returns
    -------
    xarray.DataArray
       Mean temperature values of the wettest/dryest quarter of each year.

    Notes
    -----
    According to the ANUCLIM user-guide https://fennerschool.anu.edu.au/files/anuclim61.pdf (ch. 6), input
    values should be at a weekly (or monthly) frequency.  However, the xclim.indices implementation here will calculate
    the result with input data with daily frequency as well. As such weekly or monthly input values, if desired,
    should be calculated prior to calling the function.

    """
    tas_qrt = _to_quarter(input_freq, tas=tas)
    pr_qrt = _to_quarter(input_freq, pr=pr)

    with xarray.set_options(keep_attrs=True):

        if op == "wettest":

            def get_at_extreme(ds):
                return lazy_indexing(ds.tas, ds.pr.argmax(dim="time"), dim="time")

        elif op == "dryest":

            def get_at_extreme(ds):
                return lazy_indexing(ds.tas, ds.pr.argmin(dim="time"), dim="time")

        else:
            raise NotImplementedError(
                f'Unknown operation "{op}" ; op parameter but be one of "wettest" or "dryest"'
            )
        out = (
            xarray.Dataset(data_vars={"tas": tas_qrt, "pr": pr_qrt})
            .resample(time=freq)
            .map(get_at_extreme)
        )
        out.attrs = tas.attrs
        return out


@declare_units("mm", pr="[precipitation]")
def prcptot_wetdry_quarter(
    pr: xarray.DataArray, op: str = None, input_freq: str = None, freq: str = "YS"
):
    r""" ANUCLIM Total precipitation of wettest/dryest quarter

    The wettest (or dryest) quarter of the year is determined, and the total precipitation of this
    period is calculated. If the input data frequency is "daily" or "weekly" quarters
    are defined as 13 week periods, otherwise are 3 months.

    Parameters
    ----------
    pr : xarray.DataArray
      Total precipitation rate at daily, weekly, or monthly frequency.
    op : str
      Operation to perform :  'wettest' calculate wettest quarter ; 'dryest' calculate dryest quarter.
    input_freq : str
      Input data time frequency - One of 'daily', 'weekly' or 'monthly'.
    freq : str
      Resampling frequency; Defaults to "YS".

    Returns
    -------
    xarray.DataArray
       Total precipitation values of the wettest/dryest quarter of each year.

    Examples
    --------

    The following would compute for each grid cell of file `pr.day.nc` the annual wettest quarter total precipitation:

    >>> import xarray as xr
    >>> import xclim.indices as xci
    >>> p = xr.open_dataset('pr.day.nc')
    >>> pr_warm_qrt = xci.prcptot_wetdry_quarter(pr=p.pr, op='wettest', input_freq='daily')

    Notes
    -----
    According to the ANUCLIM user-guide https://fennerschool.anu.edu.au/files/anuclim61.pdf (ch. 6), input
    values should be at a weekly (or monthly) frequency.  However, the xclim.indices implementation here will calculate
    the result with input data with daily frequency as well. As such weekly or monthly input values, if desired,
    should be calculated prior to calling the function.

    """
    # determine input data frequency

    if input_freq == "monthly":
        data1 = pint_multiply(pr, 1 * units.month, "mm")
        wind = 3
    elif input_freq == "weekly":
        data1 = pint_multiply(pr, 1 * units.week, "mm")
        wind = 13
    elif input_freq == "daily":
        data1 = precip_accumulation(pr, freq="7D")
        wind = 13
    else:
        raise NotImplementedError(
            f'Unknown input time frequency "{input_freq}" : input_freq parameter must be '
            f'one of "daily", "weekly" or "monthly"'
        )

    with xarray.set_options(keep_attrs=True):
        out = data1.rolling(time=wind, center=False,).sum(allow_lazy=True, skipna=False)
        out.attrs = data1.attrs
        # out.attrs["units"] = "mm"
        if op == "wettest":
            out = out.resample(time=freq).max(dim="time")
        elif op == "dryest":
            out = out.resample(time=freq).min(dim="time")
        else:
            raise NotImplementedError(
                f'Unknown operation "{op}" : op parameter must be one of "wettest" or "dryest"'
            )
        return out


@declare_units("mm", pr="[precipitation]", tas="[temperature]")
def prcptot_warmcold_quarter(
    pr: xarray.DataArray,
    tas: xarray.DataArray,
    op: str = None,
    input_freq: str = None,
    freq="YS",
):
    r""" ANUCLIM Total precipitation of warmest/coldest quarter

    The warmest (or coldest) quarter of the year is determined, and the total
    precipitation of this period is calculated.  If the input data frequency is "daily" or "weekly" quarters
    are defined as 13 week periods, otherwise are 3 months.

    Parameters
    ----------
    pr : xarray.DataArray
      Total precipitation rate at daily, weekly, or monthly frequency.
    tas : xarray.DataArray
      Mean temperature [℃] or [K] at daily, weekly, or monthly frequency.
    op : str
      Operation to perform: 'warmest' calculate for the warmest quarter ; 'coldest' calculate for the coldest quarter.
    input_freq : str
      Input data time frequency - One of 'daily', 'weekly' or 'monthly'.
    freq : str
      Resampling frequency; Defaults to "YS".

    Returns
    -------
    xarray.DataArray
       Total precipitation values of the warmest/coldest quarter of each year.

    Notes
    -----
    According to the ANUCLIM user-guide https://fennerschool.anu.edu.au/files/anuclim61.pdf (ch. 6), input
    values should be at a weekly (or monthly) frequency.  However, the xclim.indices implementation here will calculate
    the result with input data with daily frequency as well. As such weekly or monthly input values, if desired,
    should be calculated prior to calling the function.

    """
    # determine input data frequency
    if input_freq == "monthly":
        wind = 3
        pr = pint_multiply(pr, 1 * units.month, "mm")
    elif input_freq == "weekly":
        pr = pint_multiply(pr, 1 * units.week, "mm")
        wind = 13
    elif input_freq == "daily":
        tas = tg_mean(tas, freq="7D")
        pr = precip_accumulation(pr, freq="7D")
        wind = 13
    else:
        raise NotImplementedError(
            f'Unknown input time frequency "{input_freq}" : input_freq parameter must be '
            f'one of "daily", "weekly" or "monthly"'
        )

    with xarray.set_options(keep_attrs=True):
        tas_qrt = tas.rolling(time=wind, center=False).mean()
        pr_qrt = pr.rolling(time=wind, center=False).sum()

        if op == "warmest":

            def get_at_extreme(ds):
                return lazy_indexing(ds.pr, ds.tas.argmax(dim="time"), dim="time")

        elif op == "coldest":

            def get_at_extreme(ds):
                return lazy_indexing(ds.pr, ds.tas.argmin(dim="time"), dim="time")

        else:
            raise NotImplementedError(
                f'Unknown operation "{op}" ; op parameter but be one of "warmest" or "coldest"'
            )

        out = (
            xarray.Dataset(data_vars={"tas": tas_qrt, "pr": pr_qrt})
            .resample(time=freq)
            .map(get_at_extreme)
        )
        out.attrs = pr.attrs
        out.attrs["units"] = "mm"
        return out


@declare_units("mm", pr="[precipitation]")
def prcptot(pr: xarray.DataArray, input_freq: str = None, freq: str = "YS"):
    r"""ANUCLIM Accumulated total precipitation.

    Parameters
    ----------
    pr : xarray.DataArray
      Total precipitation flux [mm d-1], [mm week-1], [mm month-1] or similar.
    input_freq : str
      Input data time frequency - One of 'daily', 'weekly' or 'monthly'.
    freq : str
      Resampling frequency; Defaults to "YS".

    Returns
    -------
    xarray.DataArray
       Total precipitation [mm].

    Notes
    -----
    According to the ANUCLIM user-guide https://fennerschool.anu.edu.au/files/anuclim61.pdf (ch. 6), input
    values should be at a weekly (or monthly) frequency.  However, the xclim.indices implementation here will calculate
    the result with input data with daily frequency as well.

    """

    if input_freq == "monthly":
        pr = pint_multiply(pr, 1 * units.month, "mm")
    elif input_freq == "weekly":
        pr = pint_multiply(pr, 1 * units.week, "mm")
    elif input_freq == "daily":
        pr = pint_multiply(pr, 1 * units.day, "mm")
    else:
        raise NotImplementedError(
            f'Unknown input time frequency "{input_freq}" : input_freq parameter must be '
            f'one of "daily", "weekly" or "monthly"'
        )
    return pr.resample(time=freq).sum(dim="time", keep_attrs=True)


@declare_units("mm", pr="[precipitation]")
def prcptot_wetdry_period(
    pr: xarray.DataArray, op: str = None, input_freq: str = None, freq: str = "YS"
):
    r"""ANUCLIM precipitation of the wettest/dryest day, week or month, depending on the time step

    Parameters
    ----------
    pr : xarray.DataArray
      Total precipitation flux [mm d-1], [mm week-1], [mm month-1] or similar.
    op : str
      Operation to perform :  'wettest' calculate wettest period ; 'dryest' calculate dryest period.
    input_freq : str
      Input data time frequency - One of 'daily', 'weekly' or 'monthly'.
    freq : str
      Resampling frequency; Defaults to "YS".

    Returns
    -------
    xarray.DataArray
       Total precipitation [mm] of the wettest / dryest period.

    Notes
    -----
    According to the ANUCLIM user-guide https://fennerschool.anu.edu.au/files/anuclim61.pdf (ch. 6), input
    values should be at a weekly (or monthly) frequency.  However, the xclim.indices implementation here will calculate
    the result with input data with daily frequency as well. As such weekly or monthly input values, if desired,
    should be calculated prior to calling the function.
    """

    if input_freq == "monthly":
        pr = pint_multiply(pr, 1 * units.month, "mm")
    elif input_freq == "weekly":
        pr = pint_multiply(pr, 1 * units.week, "mm")
    elif input_freq == "daily":
        pr = pint_multiply(pr, 1 * units.day, "mm")
    else:
        raise NotImplementedError(
            f'Unknown input time frequency "{input_freq}" : input_freq parameter must be '
            f'one of "daily", "weekly" or "monthly"'
        )

    if op == "wettest":
        return pr.resample(time=freq).max(dim="time", keep_attrs=True)
    elif op == "dryest":
        return pr.resample(time=freq).min(dim="time", keep_attrs=True)
    raise NotImplementedError(
        f'Unknown operation "{op}" ; op parameter but be one of "wettest" or "dryest"'
    )


def _anuclim_coeff_var(arr: xarray.DataArray):
    r""" calculate the annual coefficient of variation for anuclim indices"""
    std = arr.resample(time="YS").std(dim="time")
    mu = arr.resample(time="YS").mean(dim="time")
    return std / mu


def _to_quarter(freq, pr=None, tas=None):
    """Convert daily, weekly or monthly time series to quarterly time series according to ANUCLIM specifications."""

    if freq.upper().startswith("D"):
        if tas is not None:
            tas = tg_mean(tas, freq="7D")

        if pr is not None:
            pr = precip_accumulation(pr, freq="7D")
            pr.attrs["units"] = "mm/week"

        freq = "W"

    if freq.upper().startswith("W"):
        window = 13
        u = units.week

    elif freq.upper().startswith("M"):
        window = 3
        u = units.month

    else:
        raise NotImplementedError(
            f'Unknown input time frequency "{freq}": must be one of "daily", "weekly" or "monthly".'
        )

    with xarray.set_options(keep_attrs=True):
        if pr is not None:
            pr = pint_multiply(pr, 1 * u, "mm")
            out = pr.rolling(time=window, center=False).sum()
            out.attrs = pr.attrs

        if tas is not None:
            out = tas.rolling(time=window, center=False).mean(
                allow_lazy=True, skipna=False
            )
            out.attrs = tas.attrs

    return out

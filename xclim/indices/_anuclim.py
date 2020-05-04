import xarray

from xclim.core.units import convert_units_to
from xclim.core.units import declare_units
from xclim.core.units import units

xarray.set_options(enable_cftimeindex=True)  # Set xarray to use cftimeindex


# Frequencies : YS: year start, QS-DEC: seasons starting in december, MS: month start
# See http://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases

# -------------------------------------------------- #
# ATTENTION: ASSUME ALL INDICES WRONG UNTIL TESTED ! #
# -------------------------------------------------- #

__all__ = ["temperature_seasonality", "precip_seasonality"]


@declare_units("percent", tas="[temperature]")
def temperature_seasonality(tas: xarray.DataArray,):
    r""" ANUCLIM Temperature Seasonality (C of V)
    The annual temperature Coefficient of Variation (C of V) expressed in percent. Calculated as the standard deviation
    of temperature values for a given year expressed as a percentage of the mean of those temperatures. For this
    calculation, the mean in degrees Kelvin is used. This avoids the possibility of having to
    divide by zero, but it does mean that the values are usually quite small.
    See : https://fennerschool.anu.edu.au/files/anuclim61.pdf ch. 6


    Parameters
    ----------
    tas : xarray.DataArray
      Mean temperature [℃] or [K] at daily, weekly, or monthly frequency

    Returns
    -------
    xarray.DataArray
      The Coefficient of Variation of mean temperature values expressed in percent.

    Examples
    --------

    The following would compute for each grid cell of file `tas.day.nc` the annual temperature
    temperature seasonality:

    >>> import xarray as xr
    >>> import xclim.indices as xci
    >>> t = xr.open_dataset('tas.day.nc')
    >>> tday_seasonality = xci.temperature_seasonality(t)

    >>> t_weekly = xci.tg_mean(t, freq='7D')
    >>> tweek_seasonality = xci.temperature_seasonality(t_weekly)

    Notes
    -----
    According to the ANUCLIM user-guide https://fennerschool.anu.edu.au/files/anuclim61.pdf (ch. 6), input
    values should be at a weekly (or monthly) frequency.  However, the xclim.indices implementation here will calculate
    the C of V on input data of any frequency. As such weekly or monthly averages, if desired, should be calculated prior
    to calling the function

    """

    # Ensure temperature data is in Kelvin
    tas = convert_units_to(tas, "K")

    with xarray.set_options(keep_attrs=True):
        seas = 100 * (_anuclim_coeff_var(tas))

    seas.attrs["units"] = "percent"
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
       total precipitation rate at daily, weekly, or monthly frequency.
       pr units need to be defined as a rate (e.g. mm d-1, mm week-1)

    Returns
    -------
    xarray.DataArray
      The Coefficient of Variation of precipitation values

    Examples
    --------

    The following would compute for each grid cell of file `pr.day.nc` the annual precipitation seasonality:

    >>> import xarray as xr
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
    the C of V on input data of any frequency. As such weekly or monthly averages, if desired, should be calculated prior
    to calling the function.

    If input units are in mm s-1 or equivalent values are converted to mm/day to avoid potentially small denominator values

    """
    pr_units = units.parse_units(pr.attrs["units"].replace("-", "**-"))
    mm_s = units.parse_units("mm s-1".replace("-", "**-"))
    # if units in mm/sec convert to mm/days to avoid potentially small denominator
    if 1 * mm_s == 1 * pr_units:
        pr = convert_units_to(pr, "mm d-1")
    with xarray.set_options(keep_attrs=True):
        seas = 100 * (_anuclim_coeff_var(pr))

    seas.attrs["units"] = "percent"
    return seas


@declare_units("percent", tas="[temperature]")
def tg_mean_warmest_quarter(tas: xarray.DataArray):
    r""" ANUCLIM Mean Temperature of Warmest Quarter
    The warmest quarter of the year is determined (13 week or 3 month period), and the mean
    temperature of this period is calculated.


    Parameters
    ----------
    tas : xarray.DataArray
      Mean temperature [℃] or [K] at daily, weekly, or monthly frequency

    Returns
    -------
    xarray.DataArray
      The Coefficient of Variation of mean temperature values expressed in percent.

    Examples
    --------

    The following would compute for each grid cell of file `tas.day.nc` the annual temperature
    temperature seasonality:

    >>> import xarray as xr
    >>> import xclim.indices as xci
    >>> t = xr.open_dataset('tas.day.nc')
    >>> tday_seasonality = xci.temperature_seasonality(t)

    >>> t_weekly = xci.tg_mean(t, freq='7D')
    >>> tweek_seasonality = xci.temperature_seasonality(t_weekly)

    Notes
    -----
    According to the ANUCLIM user-guide https://fennerschool.anu.edu.au/files/anuclim61.pdf (ch. 6), input
    values should be at a weekly (or monthly) frequency.  However, the xclim.indices implementation here will calculate
    the C of V on input data of any frequency. As such weekly or monthly averages, if desired, should be calculated prior
    to calling the function

    """


def _anuclim_coeff_var(arr: xarray.DataArray):
    r""" calculate the annual coefficient of variation for anuclim indices"""
    cv = arr.resample(time="YS").std(dim="time") / arr.resample(time="YS").mean(
        dim="time"
    )
    return cv


def _anuclim_max_min_quarter(arr: xarray.DataArray, op=None):
    r"""indentify the quarter period during which we find the annuale max or min """
    arr

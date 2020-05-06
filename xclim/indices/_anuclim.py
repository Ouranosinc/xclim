import xarray

import xclim.indices as xci
from xclim.core.units import convert_units_to
from xclim.core.units import declare_units
from xclim.core.units import units

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
    "prcptot_wetdry_quarter",
]


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


@declare_units("[temperature]", tas="[temperature]")
def tg_mean_warmcold_quarter(
    tas: xarray.DataArray, op: str = None, input_freq: str = None
):
    r""" ANUCLIM Mean temperature of warmest/coldest quarter
    The warmest (or coldest) quarter of the year is determined, and the mean
    temperature of this period is calculated.  If the input data frequency is "daily" or "weekly" quarters
    are defined as 13 week periods, otherwise are 3 months.

    Parameters
    ----------
    tas : xarray.DataArray
      Mean temperature [℃] or [K] at daily, weekly, or monthly frequency

    op : str
        Operation to perform :  'warmest' calculate warmest quarter ; 'coldest' calculate coldest quarter

    input_freq : str
        input data time frequency - One of 'daily', 'weekly' or 'monthly'

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
    >>> t_warm_qrt = xci.tg_mean_warmest_quarter(tas=t, op='warmest', input_freq='daily')

    Notes
    -----
    According to the ANUCLIM user-guide https://fennerschool.anu.edu.au/files/anuclim61.pdf (ch. 6), input
    values should be at a weekly (or monthly) frequency.  However, the xclim.indices implementation here will calculate
    the result with input data of any frequency.

    """
    # determine input data frequency
    # determine input data frequency
    if input_freq == "monthly":
        data1 = tas
        wind = 3
    elif input_freq == "weekly":
        data1 = tas
        wind = 13
    elif input_freq == "daily":
        data1 = xci.tg_mean(tas, freq="7D")
        wind = 13
    else:
        raise NotImplementedError(
            f'Unknown input time frequency "{input_freq}" : input_freq parameter must be '
            f'one of "daily", "weekly" or "monthly"'
        )

    with xarray.set_options(keep_attrs=True):
        out = data1.rolling(time=wind, center=False,).mean(
            allow_lazy=True, skipna=False
        )
        out.attrs = data1.attrs
        if op == "warmest":
            out = out.resample(time="YS").max(dim="time")
        elif op == "coldest":
            out = out.resample(time="YS").min(dim="time")
        else:
            raise NotImplementedError(
                f'Unknown operation "{op}" ; op parameter but be one of "warmest" or "coldest"'
            )
        return out


@declare_units("mm", pr="[precipitation]")
def prcptot_wetdry_quarter(
    pr: xarray.DataArray, op: str = None, input_freq: str = None
):
    r""" ANUCLIM Total precipitation of wettest/driest quarter
    The wettest (or driest) quarter of the year is determined, and the total precipitation of this
    period is calculated. If the input data frequency is "daily" or "weekly" quarters
    are defined as 13 week periods, otherwise are 3 months.

    Parameters
    ----------
    pr : xarray.DataArray
      Total precipitation rate at daily, weekly, or monthly frequency

    op : str
        Operation to perform :  'wettest' calculate wettest quarter ; 'driest' calculate driest quarter

    input_freq : str
        input data time frequency - One of 'daily', 'weekly' or 'monthly'

    Returns
    -------
    xarray.DataArray
       Total precipitation values of the wettest/driest quarter of each year.

    Examples
    --------

    The following would compute for each grid cell of file `pr.day.nc` the annual wettest quarter total precipitation:

    >>> import xarray as xr
    >>> import xclim.indices as xci
    >>> t = xr.open_dataset('pr.day.nc')
    >>> pr_warm_qrt = xci.prcptot_wetdry_quarter(tas=t, op='wettest', input_freq='daily')

    Notes
    -----
    According to the ANUCLIM user-guide https://fennerschool.anu.edu.au/files/anuclim61.pdf (ch. 6), input
    values should be at a weekly (or monthly) frequency.  However, the xclim.indices implementation here will calculate
    the result with input data of any frequency.

    """
    # determine input data frequency

    if input_freq == "monthly":
        data1 = pr
        wind = 3
    elif input_freq == "weekly":
        data1 = pr
        wind = 13
    elif input_freq == "daily":
        data1 = xci.precip_accumulation(pr, freq="7D")
        wind = 13
    else:
        raise NotImplementedError(
            f'Unknown input time frequency "{input_freq}" : input_freq parameter must be '
            f'one of "daily", "weekly" or "monthly"'
        )

    with xarray.set_options(keep_attrs=True):
        out = data1.rolling(time=wind, center=False,).sum(allow_lazy=True, skipna=False)
        out.attrs = data1.attrs
        out.attrs["units"] = "mm"
        if op == "wettest":
            out = out.resample(time="YS").max(dim="time")
        elif op == "driest":
            out = out.resample(time="YS").min(dim="time")
        else:
            raise NotImplementedError(
                f'Unknown operation "{op}" : op parameter must be one of "wettest" or "driest"'
            )
        return out


def _anuclim_coeff_var(arr: xarray.DataArray):
    r""" calculate the annual coefficient of variation for anuclim indices"""
    cv = arr.resample(time="YS").std(dim="time") / arr.resample(time="YS").mean(
        dim="time"
    )
    return cv

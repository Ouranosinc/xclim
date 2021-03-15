# noqa: D100
from typing import Optional

import numpy as np
import xarray

from xclim.core.units import (
    convert_units_to,
    declare_units,
    pint_multiply,
    rate2amount,
    units,
    units2pint,
)
from xclim.core.utils import ensure_chunk_size

from ._multivariate import (
    daily_temperature_range,
    extreme_temperature_range,
    precip_accumulation,
)
from ._simple import tg_mean
from .generic import select_resample_op
from .run_length import lazy_indexing

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

_xr_argops = {
    "wettest": xarray.DataArray.argmax,
    "warmest": xarray.DataArray.argmax,
    "dryest": xarray.DataArray.argmin,
    "driest": xarray.DataArray.argmin,
    "coldest": xarray.DataArray.argmin,
}

_np_ops = {
    "wettest": "max",
    "warmest": "max",
    "dryest": "min",
    "driest": "min",
    "coldest": "min",
}


@declare_units(tasmin="[temperature]", tasmax="[temperature]")
def isothermality(
    tasmin: xarray.DataArray, tasmax: xarray.DataArray, freq: str = "YS"
) -> xarray.DataArray:
    r"""Isothermality.

    The mean diurnal range divided by the annual temperature range.

    Parameters
    ----------
    tasmin : xarray.DataArray
      Average daily minimum temperature at daily, weekly, or monthly frequency.
    tasmax : xarray.DataArray
      Average daily maximum temperature at daily, weekly, or monthly frequency.
    freq : str
      Resampling frequency.

    Returns
    -------
    xarray.DataArray, [%]
      The isothermality value.

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


@declare_units(tas="[temperature]")
def temperature_seasonality(tas: xarray.DataArray) -> xarray.DataArray:
    r"""ANUCLIM temperature seasonality (coefficient of variation).

    The annual temperature coefficient of variation expressed in percent. Calculated as the standard deviation
    of temperature values for a given year expressed as a percentage of the mean of those temperatures.

    Parameters
    ----------
    tas : xarray.DataArray
      Mean temperature at daily, weekly, or monthly frequency.

    Returns
    -------
    xarray.DataArray, [%]
      The Coefficient of Variation of mean temperature values.

    Examples
    --------
    The following would compute for each grid cell of file `tas.day.nc` the annual temperature seasonality:

    >>> import xclim.indices as xci
    >>> t = xr.open_dataset(path_to_tas_file).tas
    >>> tday_seasonality = xci.temperature_seasonality(t)
    >>> t_weekly = xci.tg_mean(t, freq='7D')
    >>> tweek_seasonality = xci.temperature_seasonality(t_weekly)

    Notes
    -----
    For this calculation, the mean in degrees Kelvin is used. This avoids the possibility of having to
    divide by zero, but it does mean that the values are usually quite small.

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


@declare_units(pr="[precipitation]")
def precip_seasonality(
    pr: xarray.DataArray,
) -> xarray.DataArray:
    r"""ANUCLIM Precipitation Seasonality (C of V).

    The annual precipitation Coefficient of Variation (C of V) expressed in percent. Calculated as the standard deviation
    of precipitation values for a given year expressed as a percentage of the mean of those values.

    Parameters
    ----------
    pr : xarray.DataArray
      Total precipitation rate at daily, weekly, or monthly frequency.
      Units need to be defined as a rate (e.g. mm d-1, mm week-1).

    Returns
    -------
    xarray.DataArray, [%]
      The coefficient of variation of precipitation values.

    Examples
    --------
    The following would compute for each grid cell of file `pr.day.nc` the annual precipitation seasonality:

    >>> import xclim.indices as xci
    >>> p = xr.open_dataset(path_to_pr_file).pr
    >>> pday_seasonality = xci.precip_seasonality(p)
    >>> p_weekly = xci.precip_accumulation(p, freq='7D')

    # Input units need to be a rate
    >>> p_weekly.attrs['units'] = "mm/week"
    >>> pweek_seasonality = xci.precip_seasonality(p_weekly)

    Notes
    -----
    According to the ANUCLIM user-guide https://fennerschool.anu.edu.au/files/anuclim61.pdf (ch. 6), input
    values should be at a weekly (or monthly) frequency.  However, the xclim.indices implementation here will calculate
    the result with input data with daily frequency as well. As such weekly or monthly input values, if desired,
    should be calculated prior to calling the function.

    If input units are in mm s-1 (or equivalent) values are converted to mm/day to avoid potentially small denominator
    values.
    """
    # If units in mm/sec convert to mm/days to avoid potentially small denominator
    if units2pint(pr) == units("mm / s"):
        pr = convert_units_to(pr, "mm d-1")

    with xarray.set_options(keep_attrs=True):
        seas = 100 * _anuclim_coeff_var(pr)

    seas.attrs["units"] = "%"
    return seas


@declare_units(tas="[temperature]")
def tg_mean_warmcold_quarter(
    tas: xarray.DataArray,
    op: str = None,
    src_timestep: str = None,
    freq: str = "YS",
) -> xarray.DataArray:
    r"""ANUCLIM Mean temperature of warmest/coldest quarter.

    The warmest (or coldest) quarter of the year is determined, and the mean temperature of this period is
    calculated.  If the input data frequency is daily ("D") or weekly ("W"), quarters are defined as 13 week periods,
    otherwise as 3 months.

    Parameters
    ----------
    tas : xarray.DataArray
      Mean temperature at daily, weekly, or monthly frequency.
    op : str {'warmest', 'coldest'}
      Operation to perform:  'warmest' calculate warmest quarter; 'coldest' calculate coldest quarter.
    src_timestep : {'D', 'W', 'M'}
      Input data time frequency - One of daily, weekly or monthly.
    freq : str
      Resampling frequency.

    Returns
    -------
    xarray.DataArray, [same as tas]
       Mean temperature values of the warmest/coldest quearter of each year.

    Examples
    --------
    The following would compute for each grid cell of file `tas.day.nc` the annual temperature
    warmest quarter mean temperature:

    >>> import xclim.indices as xci
    >>> t = xr.open_dataset(path_to_tas_file)
    >>> t_warm_qrt = xci.tg_mean_warmcold_quarter(tas=t.tas, op='warmest', src_timestep='daily')

    Notes
    -----
    According to the ANUCLIM user-guide https://fennerschool.anu.edu.au/files/anuclim61.pdf (ch. 6), input
    values should be at a weekly (or monthly) frequency.  However, the xclim.indices implementation here will calculate
    the result with input data with daily frequency as well. As such weekly or monthly input values, if desired,
    should be calculated prior to calling the function.
    """
    out = _to_quarter(src_timestep, tas=tas)

    oper = _np_ops[op]
    out = select_resample_op(out, oper, freq)
    out.attrs["units"] = tas.units
    return out


@declare_units(tas="[temperature]", pr="[precipitation]")
def tg_mean_wetdry_quarter(
    tas: xarray.DataArray,
    pr: xarray.DataArray,
    op: str = None,
    src_timestep: str = None,
    freq: str = "YS",
) -> xarray.DataArray:
    r"""ANUCLIM Mean temperature of wettest/driest quarter.

    The wettest (or driest) quarter of the year is determined, and the mean temperature of this period is calculated.
    If the input data frequency is daily ("D") or weekly ("W"), quarters are defined as 13 week periods, otherwise are 3 months.

    Parameters
    ----------
    tas : xarray.DataArray
      Mean temperature at daily, weekly, or monthly frequency.
    pr : xarray.DataArray
      Total precipitation rate at daily, weekly, or monthly frequency.
    op : {'wettest', 'driest'}
      Operation to perform: 'wettest' calculate for the wettest quarter; 'driest' calculate for the driest quarter.
    src_timestep : {'D', 'W', 'M'}
      Input data time frequency - One of daily, weekly or monthly.
    freq : str
      Resampling frequency.

    Returns
    -------
    xarray.DataArray, [same as tas]
       Mean temperature values of the wettest/driest quarter of each year.

    Notes
    -----
    According to the ANUCLIM user-guide https://fennerschool.anu.edu.au/files/anuclim61.pdf (ch. 6), input
    values should be at a weekly (or monthly) frequency.  However, the xclim.indices implementation here will calculate
    the result with input data with daily frequency as well. As such weekly or monthly input values, if desired,
    should be calculated prior to calling the function.
    """
    tas_qrt = _to_quarter(src_timestep, tas=tas)
    pr_qrt = _to_quarter(src_timestep, pr=pr)

    xr_op = _xr_argops[op]
    with xarray.set_options(keep_attrs=True):
        out = _from_other_arg(criteria=pr_qrt, output=tas_qrt, op=xr_op, freq=freq)
        out.attrs = tas.attrs
        return out


@declare_units(pr="[precipitation]")
def prcptot_wetdry_quarter(
    pr: xarray.DataArray, op: str = None, src_timestep: str = None, freq: str = "YS"
) -> xarray.DataArray:
    r"""ANUCLIM Total precipitation of wettest/driest quarter.

    The wettest (or driest) quarter of the year is determined, and the total precipitation of this
    period is calculated. If the input data frequency is daily ("D") or weekly ("W") quarters
    are defined as 13 week periods, otherwise are 3 months.

    Parameters
    ----------
    pr : xarray.DataArray
      Total precipitation rate at daily, weekly, or monthly frequency.
    op : {'wettest', 'driest'}
      Operation to perform :  'wettest' calculate wettest quarter ; 'driest' calculate driest quarter.
    src_timestep : {'D', 'W', 'M'}
      Input data time frequency - One of daily, weekly or monthly.
    freq : str
      Resampling frequency.

    Returns
    -------
    xarray.DataArray, [length]
       Total precipitation values of the wettest/driest quarter of each year.

    Examples
    --------
    The following would compute for each grid cell of file `pr.day.nc` the annual wettest quarter total precipitation:

    >>> from xclim.indices import prcptot_wetdry_quarter
    >>> p = xr.open_dataset(path_to_pr_file)
    >>> pr_warm_qrt = prcptot_wetdry_quarter(pr=p.pr, op='wettest', src_timestep='D')

    Notes
    -----
    According to the ANUCLIM user-guide https://fennerschool.anu.edu.au/files/anuclim61.pdf (ch. 6), input
    values should be at a weekly (or monthly) frequency.  However, the xclim.indices implementation here will calculate
    the result with input data with daily frequency as well. As such weekly or monthly input values, if desired,
    should be calculated prior to calling the function.
    """
    # returns mm values
    pr_qrt = _to_quarter(src_timestep, pr=pr)

    try:
        oper = _np_ops[op]
    except KeyError:
        raise NotImplementedError(
            f'Unknown operation "{op}" ; not one of "wettest" or "driest"'
        )

    out = select_resample_op(pr_qrt, oper, freq)
    out.attrs["units"] = pr_qrt.units
    return out


@declare_units(pr="[precipitation]", tas="[temperature]")
def prcptot_warmcold_quarter(
    pr: xarray.DataArray,
    tas: xarray.DataArray,
    op: str = None,
    src_timestep: str = None,
    freq: str = "YS",
) -> xarray.DataArray:
    r"""ANUCLIM Total precipitation of warmest/coldest quarter.

    The warmest (or coldest) quarter of the year is determined, and the total
    precipitation of this period is calculated.  If the input data frequency is daily ("D) or weekly ("W"), quarters
    are defined as 13 week periods, otherwise are 3 months.

    Parameters
    ----------
    pr : xarray.DataArray
      Total precipitation rate at daily, weekly, or monthly frequency.
    tas : xarray.DataArray
      Mean temperature at daily, weekly, or monthly frequency.
    op : {'warmest', 'coldest'}
      Operation to perform: 'warmest' calculate for the warmest quarter ; 'coldest' calculate for the coldest quarter.
    src_timestep : {'D', 'W', 'M'}
      Input data time frequency - One of daily, weekly or monthly.
    freq : str
      Resampling frequency.

    Returns
    -------
    xarray.DataArray : [mm]
       Total precipitation values of the warmest/coldest quarter of each year

    Notes
    -----
    According to the ANUCLIM user-guide https://fennerschool.anu.edu.au/files/anuclim61.pdf (ch. 6), input
    values should be at a weekly (or monthly) frequency.  However, the xclim.indices implementation here will calculate
    the result with input data with daily frequency as well. As such weekly or monthly input values, if desired,
    should be calculated prior to calling the function.
    """
    # determine input data frequency
    tas_qrt = _to_quarter(src_timestep, tas=tas)
    # returns mm values
    pr_qrt = _to_quarter(src_timestep, pr=pr)

    xr_op = _xr_argops[op]
    out = _from_other_arg(criteria=tas_qrt, output=pr_qrt, op=xr_op, freq=freq)
    out.attrs = pr_qrt.attrs
    return out


@declare_units(pr="[precipitation]")
def prcptot(
    pr: xarray.DataArray, src_timestep: str = None, freq: str = "YS"
) -> xarray.DataArray:
    r"""ANUCLIM Accumulated total precipitation.

    Parameters
    ----------
    pr : xarray.DataArray
      Total precipitation flux [mm d-1], [mm week-1], [mm month-1] or similar.
    src_timestep : {'D', 'W', 'M'}
      Input data time frequency - One of daily, weekly or monthly.
    freq : str
      Resampling frequency.

    Returns
    -------
    xarray.DataArray, [length]
       Total precipitation.

    Notes
    -----
    According to the ANUCLIM user-guide https://fennerschool.anu.edu.au/files/anuclim61.pdf (ch. 6), input
    values should be at a weekly (or monthly) frequency.  However, the xclim.indices implementation here will calculate
    the result with input data with daily frequency as well.
    """
    pram = rate2amount(pr)
    return pram.resample(time=freq).sum(dim="time", keep_attrs=True)


@declare_units(pr="[precipitation]")
def prcptot_wetdry_period(
    pr: xarray.DataArray, *, op: str, src_timestep: str, freq: str = "YS"
) -> xarray.DataArray:
    r"""ANUCLIM precipitation of the wettest/driest day, week, or month, depending on the time step.

    Parameters
    ----------
    pr : xarray.DataArray
      Total precipitation flux [mm d-1], [mm week-1], [mm month-1] or similar.
    op : {'wettest', 'driest'}
      Operation to perform :  'wettest' calculate wettest period ; 'driest' calculate driest period.
    src_timestep : {'D', 'W', 'M'}
      Input data time frequency - One of daily, weekly or monthly.
    freq : str
      Resampling frequency.

    Returns
    -------
    xarray.DataArray, [length]
       Total precipitation of the wettest / driest period.

    Notes
    -----
    According to the ANUCLIM user-guide https://fennerschool.anu.edu.au/files/anuclim61.pdf (ch. 6), input
    values should be at a weekly (or monthly) frequency.  However, the xclim.indices implementation here will calculate
    the result with input data with daily frequency as well. As such weekly or monthly input values, if desired,
    should be calculated prior to calling the function.
    """
    pram = rate2amount(pr)

    if op == "wettest":
        return pram.resample(time=freq).max(dim="time", keep_attrs=True)
    if op == "driest":
        return pram.resample(time=freq).min(dim="time", keep_attrs=True)
    raise NotImplementedError(
        f'Unknown operation "{op}" ; op parameter but be one of "wettest" or "driest"'
    )


def _anuclim_coeff_var(arr: xarray.DataArray) -> xarray.DataArray:
    r"""Calculate the annual coefficient of variation for ANUCLIM indices."""
    std = arr.resample(time="YS").std(dim="time")
    mu = arr.resample(time="YS").mean(dim="time")
    return std / mu


def _from_other_arg(
    criteria: xarray.DataArray, output: xarray.DataArray, op, freq
) -> xarray.DataArray:
    """Pick values from output based on operation returning an index from criteria.

    Parameters
    ----------
    criteria : DataArray
      Series on which operation returning index is applied.
    output : DataArray
      Series to be indexed.
    op : func
      Function returning an index, for example np.argmin, np.argmax, np.nanargmin, np.nanargmax.
    freq : str
      Temporal grouping.

    Returns
    -------
    DataArray
      Output values where criteria is met at the given frequency.
    """
    ds = xarray.Dataset(data_vars={"criteria": criteria, "output": output})
    dim = "time"

    def get_other_op(ds):
        all_nans = ds.criteria.isnull().all(dim=dim)
        index = op(ds.criteria.where(~all_nans, 0), dim=dim)
        return lazy_indexing(ds.output, index=index, dim=dim).where(~all_nans)

    return ds.resample(time=freq).map(get_other_op)


def _to_quarter(
    freq: str,
    pr: Optional[xarray.DataArray] = None,
    tas: Optional[xarray.DataArray] = None,
) -> xarray.DataArray:
    """Convert daily, weekly or monthly time series to quarterly time series according to ANUCLIM specifications."""
    if freq.upper().startswith("D"):
        if tas is not None:
            tas = tg_mean(tas, freq="7D")

        if pr is not None:
            # Accumulate on a week
            # Ensure units are back to a "rate" for rate2amount below
            pr = convert_units_to(precip_accumulation(pr, freq="7D"), "mm")
            pr.attrs["units"] = "mm/week"

        freq = "W"

    if freq.upper().startswith("W"):
        window = 13

    elif freq.upper().startswith("M"):
        window = 3

    else:
        raise NotImplementedError(
            f'Unknown input time frequency "{freq}": must be one of "D", "W" or "M".'
        )

    if tas is not None:
        tas = ensure_chunk_size(tas, time=np.ceil(window / 2))
    if pr is not None:
        pr = ensure_chunk_size(pr, time=np.ceil(window / 2))

    if pr is not None:
        pram = rate2amount(pr)
        out = pram.rolling(time=window, center=False).sum()
        out.attrs = pr.attrs
        out.attrs["units"] = pram.units

    if tas is not None:
        out = tas.rolling(time=window, center=False).mean(skipna=False)
        out.attrs = tas.attrs

    out = ensure_chunk_size(out, time=-1)
    return out

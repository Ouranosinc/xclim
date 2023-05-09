# noqa: D100
from __future__ import annotations

from typing import Callable

import numpy as np
import xarray

from xclim.core.units import (
    convert_units_to,
    declare_units,
    rate2amount,
    units,
    units2pint,
)
from xclim.core.utils import Quantified, ensure_chunk_size

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
    "isothermality",
    "prcptot",
    "prcptot_warmcold_quarter",
    "prcptot_wetdry_period",
    "prcptot_wetdry_quarter",
    "precip_seasonality",
    "temperature_seasonality",
    "tg_mean_warmcold_quarter",
    "tg_mean_wetdry_quarter",
]

_xr_argops = {
    "wettest": xarray.DataArray.argmax,
    "warmest": xarray.DataArray.argmax,
    "dryest": xarray.DataArray.argmin,  # "dryest" is a common enough spelling mistake
    "driest": xarray.DataArray.argmin,
    "coldest": xarray.DataArray.argmin,
}

_np_ops = {
    "wettest": "max",
    "warmest": "max",
    "dryest": "min",  # "dryest" is a common enough spelling mistake
    "driest": "min",
    "coldest": "min",
}


@declare_units(tasmin="[temperature]", tasmax="[temperature]")
def isothermality(
    tasmin: xarray.DataArray, tasmax: xarray.DataArray, freq: str = "YS"
) -> xarray.DataArray:
    r"""Isothermality.

    The mean diurnal temperature range divided by the annual temperature range.

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
       Isothermality

    Notes
    -----
    According to the ANUCLIM user-guide (:cite:t:`xu_anuclim_2010`, ch. 6), input values should be at a weekly
    (or monthly) frequency.  However, the xclim.indices implementation here will calculate the output with input data
    with daily frequency as well. As such weekly or monthly input values, if desired, should be calculated prior to
    calling the function.

    References
    ----------
    :cite:cts:`xu_anuclim_2010`
    """
    dtr = daily_temperature_range(tasmin=tasmin, tasmax=tasmax, freq=freq)
    etr = extreme_temperature_range(tasmin=tasmin, tasmax=tasmax, freq=freq)
    iso = dtr / etr * 100
    iso.attrs["units"] = "%"
    return iso


@declare_units(tas="[temperature]")
def temperature_seasonality(
    tas: xarray.DataArray, freq: str = "YS"
) -> xarray.DataArray:
    r"""Temperature seasonality (coefficient of variation).

    The annual temperature coefficient of variation expressed in percent. Calculated as the standard deviation
    of temperature values for a given year expressed as a percentage of the mean of those temperatures.

    Parameters
    ----------
    tas : xarray.DataArray
        Mean temperature at daily, weekly, or monthly frequency.
    freq : str
        Resampling frequency.

    Returns
    -------
    xarray.DataArray, [%]
      Mean temperature coefficient of variation
    freq : str
      Resampling frequency.

    Examples
    --------
    The following would compute for each grid cell of file `tas.day.nc` the annual temperature seasonality:

    >>> import xclim.indices as xci
    >>> t = xr.open_dataset(path_to_tas_file).tas
    >>> tday_seasonality = xci.temperature_seasonality(t)
    >>> t_weekly = xci.tg_mean(t, freq="7D")
    >>> tweek_seasonality = xci.temperature_seasonality(t_weekly)

    Notes
    -----
    For this calculation, the mean in degrees Kelvin is used. This avoids the possibility of having to
    divide by zero, but it does mean that the values are usually quite small.

    According to the ANUCLIM user-guide (:cite:t:`xu_anuclim_2010`, ch. 6), input values should be at a weekly
    (or monthly) frequency. However, the xclim.indices implementation here will calculate the result with input data
    with daily frequency as well. As such weekly or monthly input values, if desired, should be calculated prior to
    calling the function.

    References
    ----------
    :cite:cts:`xu_anuclim_2010`
    """
    tas = convert_units_to(tas, "K")

    seas = 100 * _anuclim_coeff_var(tas, freq=freq)
    seas.attrs["units"] = "%"
    return seas


@declare_units(pr="[precipitation]")
def precip_seasonality(pr: xarray.DataArray, freq: str = "YS") -> xarray.DataArray:
    r"""Precipitation Seasonality (C of V).

    The annual precipitation Coefficient of Variation (C of V) expressed in percent. Calculated as the standard
    deviation of precipitation values for a given year expressed as a percentage of the mean of those values.

    Parameters
    ----------
    pr : xarray.DataArray
        Total precipitation rate at daily, weekly, or monthly frequency.
        Units need to be defined as a rate (e.g. mm d-1, mm week-1).
    freq : str
        Resampling frequency.

    Returns
    -------
    xarray.DataArray, [%]
        Precipitation coefficient of variation

    Examples
    --------
    The following would compute for each grid cell of file `pr.day.nc` the annual precipitation seasonality:

    >>> import xclim.indices as xci
    >>> p = xr.open_dataset(path_to_pr_file).pr
    >>> pday_seasonality = xci.precip_seasonality(p)
    >>> p_weekly = xci.precip_accumulation(p, freq="7D")

    # Input units need to be a rate
    >>> p_weekly.attrs["units"] = "mm/week"
    >>> pweek_seasonality = xci.precip_seasonality(p_weekly)

    Notes
    -----
    According to the ANUCLIM user-guide (:cite:t:`xu_anuclim_2010`, ch. 6), input values should be at a weekly
    (or monthly) frequency. However, the xclim.indices implementation here will calculate the result with input data
    with daily frequency as well. As such weekly or monthly input values, if desired, should be calculated prior to
    calling the function.

    If input units are in mm s-1 (or equivalent), values are converted to mm/day to avoid potentially small denominator
    values.

    References
    ----------
    :cite:cts:`xu_anuclim_2010`
    """
    # If units in mm/sec convert to mm/days to avoid potentially small denominator
    if units2pint(pr) == units("mm / s"):
        pr = convert_units_to(pr, "mm d-1")

    seas = 100 * _anuclim_coeff_var(pr, freq=freq)
    seas.attrs["units"] = "%"
    return seas


@declare_units(tas="[temperature]")
def tg_mean_warmcold_quarter(
    tas: xarray.DataArray,
    op: str = None,
    freq: str = "YS",
) -> xarray.DataArray:
    r"""Mean temperature of warmest/coldest quarter.

    The warmest (or coldest) quarter of the year is determined, and the mean temperature of this period is calculated.
    If the input data frequency is daily ("D") or weekly ("W"), quarters are defined as 13-week periods, otherwise as
    three (3) months.

    Parameters
    ----------
    tas : xarray.DataArray
        Mean temperature at daily, weekly, or monthly frequency.
    op : str {'warmest', 'coldest'}
        Operation to perform: 'warmest' calculate the warmest quarter; 'coldest' calculate the coldest quarter.
    freq : str
        Resampling frequency.

    Returns
    -------
    xarray.DataArray, [same as tas]
       Mean temperature of {op} quarter

    Examples
    --------
    The following would compute for each grid cell of file `tas.day.nc` the annual temperature of the
    warmest quarter mean temperature:

    >>> from xclim.indices import tg_mean_warmcold_quarter
    >>> t = xr.open_dataset(path_to_tas_file)
    >>> t_warm_qrt = tg_mean_warmcold_quarter(tas=t.tas, op="warmest")

    Notes
    -----
    According to the ANUCLIM user-guide (:cite:t:`xu_anuclim_2010`, ch. 6), input values should be at a weekly
    (or monthly) frequency. However, the xclim.indices implementation here will calculate the result with input data
    with daily frequency as well. As such weekly or monthly input values, if desired, should be calculated prior to
    calling the function.

    References
    ----------
    :cite:cts:`xu_anuclim_2010`
    """
    out = _to_quarter(tas=tas)

    if op not in ["warmest", "coldest"]:
        raise NotImplementedError(
            f'op parameter ({op}) may only be one of "warmest", "coldest"'
        )
    oper = _np_ops[op]

    out = select_resample_op(out, oper, freq)
    out.attrs["units"] = tas.units
    return out


@declare_units(tas="[temperature]", pr="[precipitation]")
def tg_mean_wetdry_quarter(
    tas: xarray.DataArray,
    pr: xarray.DataArray,
    op: str = None,
    freq: str = "YS",
) -> xarray.DataArray:
    r"""Mean temperature of wettest/driest quarter.

    The wettest (or driest) quarter of the year is determined, and the mean temperature of this period is calculated.
    If the input data frequency is daily ("D") or weekly ("W"), quarters are defined as 13-week periods,
    otherwise are 3 months.

    Parameters
    ----------
    tas : xarray.DataArray
      Mean temperature at daily, weekly, or monthly frequency.
    pr : xarray.DataArray
      Total precipitation rate at daily, weekly, or monthly frequency.
    op : {'wettest', 'driest'}
      Operation to perform: 'wettest' calculate for the wettest quarter; 'driest' calculate for the driest quarter.
    freq : str
      Resampling frequency.

    Returns
    -------
    xarray.DataArray, [same as tas]
       Mean temperature of {op} quarter

    Notes
    -----
    According to the ANUCLIM user-guide (:cite:t:`xu_anuclim_2010`, ch. 6), input values should be at a weekly
    (or monthly) frequency. However, the xclim.indices implementation here will calculate the result with input data
    with daily frequency as well. As such weekly or monthly input values, if desired, should be calculated prior to
    calling the function.

    References
    ----------
    :cite:cts:`xu_anuclim_2010`
    """
    # determine input data frequency
    tas_qrt = _to_quarter(tas=tas)
    # returns mm values
    pr_qrt = _to_quarter(pr=pr)

    if op not in ["wettest", "driest", "dryest"]:
        raise NotImplementedError(
            f'op parameter ({op}) may only be one of "wettest" or "driest"'
        )
    xr_op = _xr_argops[op]

    out = _from_other_arg(criteria=pr_qrt, output=tas_qrt, op=xr_op, freq=freq)
    return out.assign_attrs(units=tas.units)


@declare_units(pr="[precipitation]")
def prcptot_wetdry_quarter(
    pr: xarray.DataArray, op: str = None, freq: str = "YS"
) -> xarray.DataArray:
    r"""Total precipitation of wettest/driest quarter.

    The wettest (or driest) quarter of the year is determined, and the total precipitation of this period is calculated.
    If the input data frequency is daily ("D") or weekly ("W") quarters are defined as 13-week periods, otherwise are
    three (3) months.

    Parameters
    ----------
    pr : xarray.DataArray
      Total precipitation rate at daily, weekly, or monthly frequency.
    op : {'wettest', 'driest'}
      Operation to perform :  'wettest' calculate the wettest quarter ; 'driest' calculate the driest quarter.
    freq : str
      Resampling frequency.

    Returns
    -------
    xarray.DataArray, [length]
       Precipitation of {op} quarter

    Examples
    --------
    The following would compute for each grid cell of file `pr.day.nc` the annual wettest quarter total precipitation:

    >>> from xclim.indices import prcptot_wetdry_quarter
    >>> p = xr.open_dataset(path_to_pr_file)
    >>> pr_warm_qrt = prcptot_wetdry_quarter(pr=p.pr, op="wettest")

    Notes
    -----
    According to the ANUCLIM user-guide (:cite:t:`xu_anuclim_2010`, ch. 6), input values should be at a weekly
    (or monthly) frequency. However, the xclim.indices implementation here will calculate the result with input data
    with daily frequency as well. As such weekly or monthly input values, if desired, should be calculated prior to
    calling the function.

    References
    ----------
    :cite:cts:`xu_anuclim_2010`
    """
    # returns mm values
    pr_qrt = _to_quarter(pr=pr)

    if op not in ["wettest", "driest", "dryest"]:
        raise NotImplementedError(
            f'op parameter ({op}) may only be one of "wettest" or "driest"'
        )
    op = _np_ops[op]

    out = select_resample_op(pr_qrt, op, freq)
    out.attrs["units"] = pr_qrt.units
    return out


@declare_units(pr="[precipitation]", tas="[temperature]")
def prcptot_warmcold_quarter(
    pr: xarray.DataArray,
    tas: xarray.DataArray,
    op: str = None,
    freq: str = "YS",
) -> xarray.DataArray:
    r"""Total precipitation of warmest/coldest quarter.

    The warmest (or coldest) quarter of the year is determined, and the total precipitation of this period is
    calculated. If the input data frequency is daily ("D) or weekly ("W"), quarters are defined as 13-week periods,
    otherwise are 3 months.

    Parameters
    ----------
    pr : xarray.DataArray
        Total precipitation rate at daily, weekly, or monthly frequency.
    tas : xarray.DataArray
        Mean temperature at daily, weekly, or monthly frequency.
    op : {'warmest', 'coldest'}
        Operation to perform: 'warmest' calculate for the warmest quarter ; 'coldest' calculate for the coldest quarter.
    freq : str
        Resampling frequency.

    Returns
    -------
    xarray.DataArray, [mm]
       Precipitation of {op} quarter

    Notes
    -----
    According to the ANUCLIM user-guide (:cite:t:`xu_anuclim_2010`, ch. 6), input values should be at a weekly
    (or monthly) frequency. However, the xclim.indices implementation here will calculate the result with input data
    with daily frequency as well. As such weekly or monthly input values, if desired, should be calculated prior to
    calling the function.

    References
    ----------
    :cite:cts:`xu_anuclim_2010`
    """
    # determine input data frequency
    tas_qrt = _to_quarter(tas=tas)
    # returns mm values
    pr_qrt = _to_quarter(pr=pr)

    if op not in ["warmest", "coldest"]:
        raise NotImplementedError(
            f'op parameter ({op}) may only be one of "warmest", "coldest"'
        )
    xr_op = _xr_argops[op]

    out = _from_other_arg(criteria=tas_qrt, output=pr_qrt, op=xr_op, freq=freq)
    out.attrs = pr_qrt.attrs
    return out


@declare_units(pr="[precipitation]", thresh="[precipitation]")
def prcptot(
    pr: xarray.DataArray, thresh: Quantified = "0 mm/d", freq: str = "YS"
) -> xarray.DataArray:
    r"""Accumulated total precipitation.

    The total accumulated precipitation from days where precipitation exceeds a given amount. A threshold is provided in
    order to allow the option of reducing the impact of days with trace precipitation amounts on period totals.

    Parameters
    ----------
    pr : xarray.DataArray
        Total precipitation flux [mm d-1], [mm week-1], [mm month-1] or similar.
    thresh : str
        Threshold over which precipitation starts being cumulated.
    freq : str
        Resampling frequency.

    Returns
    -------
    xarray.DataArray, [length]
       Total {freq} precipitation.
    """
    thresh = convert_units_to(thresh, pr, context="hydro")
    pram = rate2amount(pr.where(pr >= thresh, 0))
    return pram.resample(time=freq).sum().assign_attrs(units=pram.units)


@declare_units(pr="[precipitation]")
def prcptot_wetdry_period(
    pr: xarray.DataArray, *, op: str, freq: str = "YS"
) -> xarray.DataArray:
    r"""Precipitation of the wettest/driest day, week, or month, depending on the time step.

    The wettest (or driest) period is determined, and the total precipitation of this period is calculated.

    Parameters
    ----------
    pr : xarray.DataArray
        Total precipitation flux [mm d-1], [mm week-1], [mm month-1] or similar.
    op : {'wettest', 'driest'}
        Operation to perform :  'wettest' calculate the wettest period ; 'driest' calculate the driest period.
    freq : str
        Resampling frequency.

    Returns
    -------
    xarray.DataArray, [length]
        Precipitation of {op} period

    Notes
    -----
    According to the ANUCLIM user-guide (:cite:t:`xu_anuclim_2010`, ch. 6), input values should be at a weekly
    (or monthly) frequency. However, the xclim.indices implementation here will calculate the result with input data
    with daily frequency as well. As such weekly or monthly input values, if desired, should be calculated prior to
    calling the function.

    References
    ----------
    :cite:cts:`xu_anuclim_2010`
    """
    pram = rate2amount(pr)

    if op not in ["wettest", "driest", "dryest"]:
        raise NotImplementedError(
            f'op parameter ({op}) may only be one of "wettest" or "driest"'
        )
    op = _np_ops[op]

    return getattr(pram.resample(time=freq), op)(dim="time").assign_attrs(
        units=pram.units
    )


def _anuclim_coeff_var(arr: xarray.DataArray, freq: str = "YS") -> xarray.DataArray:
    """Calculate the annual coefficient of variation for ANUCLIM indices."""
    std = arr.resample(time=freq).std(dim="time")
    mu = arr.resample(time=freq).mean(dim="time")
    return std / mu


def _from_other_arg(
    criteria: xarray.DataArray, output: xarray.DataArray, op: Callable, freq: str
) -> xarray.DataArray:
    """Pick values from output based on operation returning an index from criteria.

    Parameters
    ----------
    criteria : xarray.DataArray
        Series on which operation returning index is applied.
    output : xarray.DataArray
        Series to be indexed.
    op : Callable
        Function returning an index, for example: `np.argmin`, `np.argmax`, `np.nanargmin`, `np.nanargmax`.
    freq : str
        Temporal grouping.

    Returns
    -------
    xarray.DataArray
        Output values where criteria are met at the given frequency.
    """
    ds = xarray.Dataset(data_vars={"criteria": criteria, "output": output})
    dim = "time"

    def get_other_op(dataset):
        all_nans = dataset.criteria.isnull().all(dim=dim)
        index = op(dataset.criteria.where(~all_nans, 0), dim=dim)
        return lazy_indexing(dataset.output, index=index, dim=dim).where(~all_nans)

    return ds.resample(time=freq).map(get_other_op)


def _to_quarter(
    pr: xarray.DataArray | None = None,
    tas: xarray.DataArray | None = None,
) -> xarray.DataArray:
    """Convert daily, weekly or monthly time series to quarterly time series according to ANUCLIM specifications."""
    if tas is not None and pr is not None:
        raise ValueError("Supply only one variable, 'tas' (exclusive) or 'pr'.")

    freq = xarray.infer_freq((tas if tas is not None else pr).time)
    if freq is None:
        raise ValueError("Can't infer sampling frequency of the input data.")

    if freq.upper().startswith("D"):
        if tas is not None:
            tas = tg_mean(tas, freq="7D")

        if pr is not None:
            # Accumulate on a week
            # Ensure units are back to a "rate" for rate2amount below
            pr = convert_units_to(
                precip_accumulation(pr, freq="7D"), "mm", context="hydro"
            )
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
        out = tas.rolling(time=window, center=False).mean(skipna=False)
        out.attrs = tas.attrs
    elif pr is not None:
        pr = ensure_chunk_size(pr, time=np.ceil(window / 2))
        pram = rate2amount(pr)
        out = pram.rolling(time=window, center=False).sum()
        out.attrs = pr.attrs
        out.attrs["units"] = pram.units
    else:
        raise ValueError("No variables supplied.")

    out = ensure_chunk_size(out, time=-1)
    return out

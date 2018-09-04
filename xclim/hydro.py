# -*- coding: utf-8 -*-
"""Hydrological indices

http://pandas.pydata.org/pandas-docs/version/0.12/timeseries.html#offset-aliases
"""

from .checks import valid_daily_mean_discharge, valid_daily_mean_temperature
import xarray as xr
import numpy as np

K2C = 273.15


@valid_daily_mean_discharge
def base_flow_index(q, freq=None):
    """Return the base flow index, defined as the minimum 7-day average flow divided by the mean flow."""
    m7 = q.rolling(time=7, center=True).mean(dim='time')

    if freq:
        m7 = m7.resample(time=freq)
        q = q.resample(time=freq)

    m7m = m7.min(dim='time')
    return m7m/q.mean(dim='time')


@valid_daily_mean_temperature
def freshet_start(tas, thresh=0, window=5, freq='YS'):
    """Return first day of year when a temperature threshold is exceeded
    over a given number of days.
    """
    i = xr.DataArray(np.arange(tas.time.size), dims='time')
    ind = xr.broadcast(i, tas)[0]

    over = ((tas > K2C + thresh) * 1).rolling(time=window).sum(dim='time')
    i = ind.where(over == window)
    return i.resample(time=freq).min(dim='time')

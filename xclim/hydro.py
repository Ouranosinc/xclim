# -*- coding: utf-8 -*-
"""Hydrological indices

http://pandas.pydata.org/pandas-docs/version/0.12/timeseries.html#offset-aliases
"""

from .checks import *
import xarray as xr
import pandas as pd
import numpy as np

@valid_daily_mean_discharge
def BFI(q, freq=None):
    """Return the base flow index, defined as the minimum 7-day average flow divided by the mean flow."""
    m7 = q.rolling(time=7, center=True).mean(dim='time')

    if freq:
        m7 = m7.resample(time=freq)
        q = q.resample(time=freq)

    m7m = m7.min(dim='time')
    return m7m/q.mean(dim='time')

@valid_daily_mean_temperature
def freshet_start(tas, thresh=0, window=5):
    """Return first day of year when a temperature threshold is exceeded
    over a given number of days.
    """
    i = xr.DataArray(np.arange(tasmin.time.size), dims='time')
    ind = xr.broadcast(i, tasmin)[0]

    over = ((tas > K2C + thresh) * 1).rolling(time=window).sum(dim='time')
    i = ind.where(over==window)
    return i.resample(time=freq).min(dim='time')


#da = xr.DataArray(np.linspace(0, 1000, num=1001), coords=[pd.date_range('15/12/1999',  periods=1001, freq=pd.DateOffset(days=1))], dims='time')

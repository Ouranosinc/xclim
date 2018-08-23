# -*- coding: utf-8 -*-
"""Hydrological indices

http://pandas.pydata.org/pandas-docs/version/0.12/timeseries.html#offset-aliases
"""

from xclim.checks import *
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


#da = xr.DataArray(np.linspace(0, 1000, num=1001), coords=[pd.date_range('15/12/1999',  periods=1001, freq=pd.DateOffset(days=1))], dims='time')

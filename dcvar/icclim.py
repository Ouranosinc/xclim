# -*- coding: utf-8 -*-
"""ICCLIM indices.



"""
from checks import *

import xarray as xr
# Frequencies : YS: year start, QS-DEC: seasons starting in december, MS: month start

def resample1(comp):
    """Decorator to resample univariate input data according to a given frequency."""
    def func(*args, **kwds):
        var = args[0]
        if 'freq' in kwds.keys():
            return comp(var.resample(time=kwds['freq']))
        else:
            return comp(var)
    return func


@valid_daily_mean_temperature
@resample1
def TGtest(tas):
    """Mean of daily mean temperature."""
    return tas.mean(dim='time')


@valid_daily_mean_temperature
def TG(tas, freq='YS'):
    """Mean of daily mean temperature."""
    return tas.resample(time=freq).mean(dim='time')


@valid_daily_min_temperature
def TN(tasmin, freq='YS'):
    """Mean of daily minimum temperature."""
    return tasmin.resample(time=freq).mean(dim='time')


@valid_daily_max_temperature
def TX(tasmax, freq='YS'):
    """Mean of daily maximum temperature."""
    return tasmax.resample(time=freq).mean(dim='time')


@valid_daily_max_temperature
def TXx(tasmax, freq='YS'):
    """Maximum of daily maximum temperature."""
    return tasmax.resample(time=freq).max(dim='time')


@valid_daily_min_temperature
def TNx(tasmin, freq='YS'):
    """Maximum of daily minimum temperature."""
    return tasmin.resample(time=freq).max(dim='time')


@valid_daily_max_temperature
def TXn(tasmax, freq='YS'):
    """Minimum of daily maximum temperature."""
    return tasmax.resample(time=freq).min(dim='time')


@valid_daily_min_temperature
def TNn(tasmin, freq='YS'):
    """Minimum of daily minimum temperature."""
    return tasmin.resample(time=freq).min(dim='time')


@valid_daily_max_min_temperature
def DTR(tasmax, tasmin, freq='YS'):
    """Mean of daily temperature range."""
    dtr = tasmax-tasmin
    return dtr.resample(time=freq).mean(dim='time')




def check():
    # Let's try it

    fn = '~/src/flyingpigeon/flyingpigeon/tests/testdata/cmip3/tas.sresb1.giss_model_e_r.run1.atm.da.nc'
    D = xr.open_dataset(fn, chunks={'lat': 1})
    return TG(D.tas)

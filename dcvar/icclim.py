# -*- coding: utf-8 -*-
"""ICCLIM indices.



"""
from checks import *
import numpy as np
import xarray as xr
import dask
# Frequencies : YS: year start, QS-DEC: seasons starting in december, MS: month start
K2C=273.15

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

@valid_daily_mean_temperature
def GD4(tas, freq='YS'):
    """Growing degree days over 4℃.

    The sum of degree-days over 4℃.
    """
    thresh = 4 + K2C
    dt = tas.clip(min=thresh) - thresh
    return dt.resample(time=freq).sum(dim='time')

# Work in progress...
@valid_daily_mean_temperature
def GSL(tas):
    """Growing season length.

    The number of days between the first occurrence of at least
    six consecutive days with mean daily temperature over 5C and
    the first occurrence of at least six consecutive days with
    mean daily temperature below 5C after July 1st in the northern
    hemisphere and January 1st in the southern hemisphere.
    """
    freq= 'YS'
    i = xr.DataArray(np.arange(tas.time.size), dims='time')
    ind = xr.broadcast(i, tas)[0]

    c = ((tas > 5 + K2C)*1).rolling(time=6).sum(dim='time')
    i1 = ind.where(c==6).resample(time=freq).min(dim='time')

    # Resample sets the time to T00:00.
    i11 = i1.reindex_like(c, method='ffill')

    # TODO: Adjust for southern hemisphere
    i2 = ind.where(c==0).where(tas.time.dt.month >= 7)
    d = i2 - i11

    return d.resample(time=freq).max(dim='time')

@valid_daily_min_temperature
def CFD(tasmin, freq='AS-JUL'):
    """Maximum number of consecutive frost days (TN < 0℃)."""

    # TODO: Deal with start and end boundaries
    # TODO: Handle missing values ?
    # TODO: Check that input array has no missing dates (holes)

    # Create an monotonously increasing index [0,1,2,...] along the time dimension.
    i = xr.DataArray(np.arange(tasmin.time.size), dims='time')
    ind = xr.broadcast(i, tasmin)[0]

    # Mask index  values where tasmin > K2C
    d = ind.where(tasmin > K2C)

    # Fill NaNs with the following valid value
    b = d.bfill(dim='time')

    # Find the difference between start and end indices
    d = b.diff(dim='time') - 1

    return d.resample(time=freq).max(dim='time')

@valid_daily_min_temperature
def FD(tasmin, freq='YS'):
    """Number of frost days (TN < 0℃)."""
    f = (tasmin < K2C)*1
    return f.resample(time=freq).sum(dim='time')

@valid_daily_mean_temperature
def HD(tas, freq='YS', thresh=17):
    """Heating degree days.

    Sum of positive values of threshold - tas.
    """
    hd = (thresh + K2C - tas).clip(0)
    return hd.resample(time=freq).sum(dim='time')

@valid_daily_max_temperature
def ID(tasmax, freq='YS'):
    """Number of days where the daily maximum temperature is below 0℃."""
    f = (tasmax < K2C)*1
    return f.resample(time=freq).sum(dim='time')

@valid_daily_min_temperature
def CSDI(tasmin, freq='YS'):
    """Cold spell duration index."""
    raise NotImplementedError




xr.set_options(enable_cftimeindex=False)
def check():
    # Let's try it

    fn = '~/src/flyingpigeon/flyingpigeon/tests/testdata/cmip3/tas.sresb1.giss_model_e_r.run1.atm.da.nc'
    D = xr.open_dataset(fn, chunks={'lat': 1})
    return CFD(D.tas)
    #return GSL(D.tas)

o = check()

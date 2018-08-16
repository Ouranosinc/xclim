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

    thresh = 5 + K2C
    over = (tas > thresh)*1
    c = over.rolling(time=6).sum(dim='time')
    nh = c.where(tas.lat >= 0)
    sh = c.where(tas.lat < 0)

    def gsl1(ar):

        started = False
        i1 = 0
        i2 = -1
        for i, x in enumerate(ar):
            if x == 6:
                i1 = i
                started = True
            if started and x == 0 and i > 181:
                i2 = i
                break

        return i2 - i1

    def gsl(arr, axis=-1):
        return np.apply_along_axis(gsl1, axis, arr)

        #return np.clip(i2-i1, 0, np.infty)
        #i1 = ar.argmax(dim='time').where(ar.max(dim='time') == 6)


    def start(ar):
        # Could probably use ar.time.dt.dayofyear to handle bisextile years
        return ar.argmax(dim='time').where(ar.max(dim='time') == 6)

    def end(ar):
        # Need to pass the start date to make sure end > start
        return ar.argmin(dim='time').where(ar.min(dim='time') == 0)

    #out = dask.array.apply_along_axis(lambda x: x.resample(time='AS-JAN').apply(gsl), tas.get_axis_num('time'), c)
    #c.resample(time='AS-JAN').apply(lambda x: dask.array.apply_along_axis(gsl, tas.get_axis_num('time'), x))

    #return c.resample(time='AS-JAN').apply(gsl)
    r = c.resample(time='AS-JAN')
    return xr.apply_ufunc(gsl, r, input_core_dims=[['time']], kwargs={'axis': -1}, dask='allowed')

    # The code assumes that the series starts in January to align both series
    nh_i1 = c.resample(time='AS-JAN').apply(start, shortcut=False)

    nh_i2 = c.resample(time='AS-JUL').apply(end, shortcut=True)[1:] + 181
    nh_i2.coords.update(nh_i1.coords)

    sh_i2 = c.resample(time='AS-JAN').apply(end, shortcut=True)
    sh_i1 = c.resample(time='AS-JUL').apply(start, shortcut=True)[:-1] + 181
    sh_i1.coords.update(sh_i2.coords)


    nh_gsl = nh_i2 - nh_i1
    sh_gsl = sh_i2 - sh_i1 + 365
    return nh_i2, nh_i1, nh_gsl
    gsl = nh_gsl.where(tas.lat >= 0, sh_gsl)
    return nh_gsl, sh_gsl

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

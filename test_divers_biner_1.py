import xarray as xr
import xclim.indices as xci
import numpy as np
import pandas as pd
import time

def time_series( values):
    coords = pd.date_range('1/1/2000', periods=len(values), freq=pd.DateOffset(days=1))
    return xr.DataArray(values, coords=[coords, ], dims='time')
#
# testing warm_days_frequency
#
f = 'tests/testdata/NRCANdaily/nrcan_canada_daily_tasmax_1990.nc'
tasmax = xr.open_dataset(f).tasmax
tasmax -= 273.16
wdf = xci.warm_day_frequency(tasmax, freq='MS')

# testing trivial 1D values
a = np.zeros(35)
a[10:15] = 31
a[30:35] = 31
tasmax = time_series(a)
wdf = xci.warm_day_frequency(tasmax, freq='MS')
np.testing.assert_allclose(wdf.values, [ 6,  4])
wdf = xci.warm_day_frequency(tasmax, thresh=32, freq='YS')
np.testing.assert_allclose(wdf.values, [0])
wdf = xci.warm_day_frequency(tasmax, thresh=-1, freq='YS')
np.testing.assert_allclose(wdf.values, [35])

#
# testing warm_night_frequency
#
f = 'tests/testdata/NRCANdaily/nrcan_canada_daily_tasmin_1990.nc'
tasmin = xr.open_dataset(f).tasmin
tasmin  -= 273.16
wdf = xci.warm_day_frequency(tasmin , freq='MS')

# testing trivial 1D values
a = np.zeros(35)
a[:] = 21
a[10:15] = 23
a[30:35] = 23
tasmin  = time_series(a)
wdf = xci.warm_night_frequency(tasmin , freq='MS')
np.testing.assert_allclose(wdf.values, [ 6,  4])
wdf = xci.warm_night_frequency(tasmin , thresh=32, freq='YS')
np.testing.assert_allclose(wdf.values, [0])
wdf = xci.warm_night_frequency(tasmin , thresh=20, freq='YS')
np.testing.assert_allclose(wdf.values, [35])
#
# testing heat_wave_frequency
#
fn = 'tests/testdata/NRCANdaily/nrcan_canada_daily_tasmin_1990.nc'
fx = 'tests/testdata/NRCANdaily/nrcan_canada_daily_tasmax_1990.nc'
tasmin = xr.open_dataset(fn).tasmin
tasmin -= 273.16
tasmax = xr.open_dataset(fx).tasmax
tasmax -= 273.16
#hwf = xci.heat_wave_frequency(tasmin, tasmax, method = 2)
hwf = xci.heat_wave_frequency(tasmin, tasmax, method = 3)


t0 = time.time()
hwi = xci.heat_wave_index(tasmax)
print('timing hwi:{:}'.format(time.time()-t0))



print('done')


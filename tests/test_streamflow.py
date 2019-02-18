from xclim import streamflow
import xarray as xr
import numpy as np
import pandas as pd

class TestStreamflow:

    def setup(self):
        self.nx, self.ny, self.nt = 2, 3, 5000
        x = np.arange(0, self.nx)
        y = np.arange(0, self.ny)

        cx = xr.IndexVariable('x', x)
        cy = xr.IndexVariable('y', y)
        dates = pd.date_range('1900-01-01', periods=self.nt, freq=pd.DateOffset(days=1))

        time = xr.IndexVariable('time', dates)
                                #attrs={'units': 'days since 1900-01-01', 'calendar': 'standard'})

        self.da = xr.DataArray(np.random.lognormal(10, 1, (self.nt, self.nx, self.ny)),
                               dims=('time', 'x', 'y'),
                               coords={'time': time, 'x': cx, 'y': cy},
                               attrs={'units': 'm^3 s-1',
                                      'standard_name': 'streamflow'}
                               )

    def test_q1max2sp(self):

        out = streamflow.q1max2sp(self.da)


    def test_q1max2sp2(self):

        out = streamflow.q1max2sp_new(self.da)

from xclim import streamflow
import xarray as xr
import numpy as np

class TestStreamflow:

    def setup(self):
        self.nx, self.ny = 2, 3
        x = np.arange(0, self.nx)
        y = np.arange(0, self.ny)

        cx = xr.IndexVariable('x', x)
        cy = xr.IndexVariable('y', y)
        time = xr.IndexVariable('time', np.arange(500),
                                attrs={'units': 'days since 1900-01-01', 'calendar': 'standard'})

        self.da = xr.DataArray(np.random.lognormal(10, 1, (len(time), self.nx, self.ny)),
                               dims=('time', 'x', 'y'),
                               coords={'time': time, 'x': cx, 'y': cy}
                               )

    def test_simple(self):
        self.da

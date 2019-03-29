from xclim import generic
import numpy as np
import xarray as xr
from scipy.stats import lognorm


class TestStats(object):

    def setup(self):
        self.nx, self.ny = 2, 3
        x = np.arange(0, self.nx)
        y = np.arange(0, self.ny)

        cx = xr.IndexVariable('x', x)
        cy = xr.IndexVariable('y', y)
        time = xr.IndexVariable('time', np.arange(50))

        self.da = xr.DataArray(np.random.lognormal(10, 1, (len(time), self.nx, self.ny)),
                               dims=('time', 'x', 'y'),
                               coords={'time': time, 'x': cx, 'y': cy}
                               )

    def test_fit(self):
        p = generic.fit(self.da, 'lognorm')

        assert p.dims[0] == 'dparams'
        assert p.get_axis_num('dparams') == 0
        p0 = lognorm.fit(self.da.values[:, 0, 0])
        np.testing.assert_array_equal(p[:, 0, 0], p0)

        # Check that we can reuse the parameters with scipy distributions
        cdf = lognorm.cdf(.99, *p.values)
        assert cdf.shape == (self.nx, self.ny)

    def test_fa(self):
        T = 10
        q = generic.fa(self.da, T, 'lognorm')

        p0 = lognorm.fit(self.da.values[:, 0, 0])
        q0 = lognorm.ppf(1-1./T, *p0)
        np.testing.assert_array_equal(q[0, 0, 0], q0)

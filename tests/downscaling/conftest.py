import collections

import numpy as np
import pandas as pd
import pytest
import xarray as xr


@pytest.fixture
def mon_triangular():
    return np.array(list(range(0, 6)) + list(range(6, 0, -1)))


@pytest.fixture
def mon_tas(tas_series, mon_triangular):
    def _mon_tas(values):
        """Random time series whose mean varies over a monthly cycle."""
        tas = tas_series(values)
        m = mon_triangular
        factor = tas_series(m[tas.time.dt.month - 1])
        with xr.set_options(keep_attrs=True):
            return tas + factor

    return _mon_tas


@pytest.fixture
def tas_series():
    def _tas_series(values, start="2000-01-01"):
        coords = collections.OrderedDict()
        for dim, n in zip(("time", "lon", "lat"), values.shape):
            if dim == "time":
                coords[dim] = pd.date_range(
                    start, periods=n, freq=pd.DateOffset(days=1)
                )
            else:
                coords[dim] = xr.IndexVariable(dim, np.arange(n))

        return xr.DataArray(
            values,
            coords=coords,
            dims=list(coords.keys()),
            name="tas",
            attrs={
                "standard_name": "air_temperature",
                "cell_methods": "time: mean within days",
                "units": "K",
            },
        )

    return _tas_series


@pytest.fixture
def qds_month():
    dims = ("quantile", "month")
    source = xr.Variable(dims=dims, data=np.zeros((5, 12)))
    target = xr.Variable(dims=dims, data=np.ones((5, 12)) * 2)

    return xr.Dataset(
        data_vars={"source": source, "target": target},
        coords={"quantile": [0, 0.3, 5.0, 7, 1], "month": range(1, 13)},
        attrs={"group": "time.month", "window": 1},
    )

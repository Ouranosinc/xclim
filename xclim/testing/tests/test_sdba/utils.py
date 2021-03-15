import collections

import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import gamma

from xclim.sdba.utils import equally_spaced_nodes

__all__ = ["series", "cannon_2015_rvs", "cannon_2015_dist"]


def series(values, name, start="2000-01-01"):
    coords = collections.OrderedDict()
    for dim, n in zip(("time", "lon", "lat"), values.shape):
        if dim == "time":
            coords[dim] = pd.date_range(start, periods=n, freq=pd.DateOffset(days=1))
        else:
            coords[dim] = xr.IndexVariable(dim, np.arange(n))

    if name == "tas":
        attrs = {
            "standard_name": "air_temperature",
            "cell_methods": "time: mean within days",
            "units": "K",
            "kind": "+",
        }
    elif name == "pr":
        attrs = {
            "standard_name": "precipitation_flux",
            "cell_methods": "time: sum over day",
            "units": "kg m-2 s-1",
            "kind": "*",
        }

    return xr.DataArray(
        values,
        coords=coords,
        dims=list(coords.keys()),
        name=name,
        attrs=attrs,
    )


def cannon_2015_dist():
    # ref ~ gamma(k=4, theta=7.5)  mu: 30, sigma: 15
    ref = gamma(4, scale=7.5)

    # hist ~ gamma(k=8.15, theta=3.68) mu: 30, sigma: 10.5
    hist = gamma(8.15, scale=3.68)

    # sim ~ gamma(k=16, theta=2.63) mu: 42, sigma: 10.5
    sim = gamma(16, scale=2.63)

    return ref, hist, sim


def cannon_2015_rvs(n, random=True):
    # Frozen distributions
    fd = cannon_2015_dist()

    if random:
        r = [d.rvs(n) for d in fd]
    else:
        u = equally_spaced_nodes(n, None)
        r = [d.ppf(u) for d in fd]

    return map(lambda x: series(x, "pr"), r)


def nancov(X):
    """Numpy's cov but dropping observations with NaNs."""
    X_na = np.isnan(X).any(axis=0)
    return np.cov(X[:, ~X_na])

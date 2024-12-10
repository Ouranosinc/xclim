"""
SDBA Testing Utilities Module
=============================
"""

from __future__ import annotations

import collections

import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import gamma

from xclim.sdba.utils import equally_spaced_nodes

__all__ = ["cannon_2015_dist", "cannon_2015_rvs", "series"]


def series(values: np.ndarray, name: str, start: str = "2000-01-01"):
    """
    Create a DataArray with time, lon and lat dimensions.

    Parameters
    ----------
    values : np.ndarray
        The values of the DataArray.
    name : str
        The name of the DataArray.
    start : str
        The start date of the time dimension.

    Returns
    -------
    xr.DataArray
        A DataArray with time, lon and lat dimensions.
    """
    coords = collections.OrderedDict()
    for dim, n in zip(("time", "lon", "lat"), values.shape, strict=False):
        if dim == "time":
            coords[dim] = pd.date_range(start, periods=n, freq="D")
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
    else:
        raise ValueError(f"Name `{name}` not supported.")

    return xr.DataArray(
        values,
        coords=coords,
        dims=list(coords.keys()),
        name=name,
        attrs=attrs,
    )


def cannon_2015_dist() -> (gamma, gamma, gamma):  # noqa: D103
    """
    Generate the distributions used in Cannon et al. 2015.

    Returns
    -------
    tuple[gamma, gamma, gamma]
        The reference, historical and simulated distributions.
    """
    # ref ~ gamma(k=4, theta=7.5)  mu: 30, sigma: 15
    ref = gamma(4, scale=7.5)

    # hist ~ gamma(k=8.15, theta=3.68) mu: 30, sigma: 10.5
    hist = gamma(8.15, scale=3.68)

    # sim ~ gamma(k=16, theta=2.63) mu: 42, sigma: 10.5
    sim = gamma(16, scale=2.63)

    return ref, hist, sim


def cannon_2015_rvs(n: int, random: bool = True) -> list[xr.DataArray]:  # noqa: D103
    """
    Generate the Random Variables used in Cannon et al. 2015.

    Parameters
    ----------
    n : int
        The number of random variables to generate.
    random : bool
        If True, generate random variables. Otherwise, generate evenly spaced nodes.

    Returns
    -------
    list[xr.DataArray]
        A list of DataArrays with time, lon and lat dimensions.
    """
    # Frozen distributions
    fd = cannon_2015_dist()

    if random:
        r = [d.rvs(n) for d in fd]
    else:
        u = equally_spaced_nodes(n, None)
        r = [d.ppf(u) for d in fd]

    return list(map(lambda x: series(x, "pr"), r))

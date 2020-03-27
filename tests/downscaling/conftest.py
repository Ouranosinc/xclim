import collections

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from xclim.downscaling.utils import apply_correction
from xclim.downscaling.utils import equally_spaced_nodes
from xclim.downscaling.utils import parse_group


@pytest.fixture
def mon_triangular():
    return np.array(list(range(1, 7)) + list(range(7, 1, -1))) / 7


@pytest.fixture
def mon_series(series, mon_triangular):
    def _mon_series(values, name):
        """Random time series whose mean varies over a monthly cycle."""
        x = series(values, name)
        m = mon_triangular
        factor = series(m[x.time.dt.month - 1], name)

        with xr.set_options(keep_attrs=True):
            return apply_correction(x, factor, x.kind)

    return _mon_series


@pytest.fixture
def series():
    def _series(values, name, start="2000-01-01"):
        coords = collections.OrderedDict()
        for dim, n in zip(("time", "lon", "lat"), values.shape):
            if dim == "time":
                coords[dim] = pd.date_range(
                    start, periods=n, freq=pd.DateOffset(days=1)
                )
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
            values, coords=coords, dims=list(coords.keys()), name=name, attrs=attrs,
        )

    return _series


@pytest.fixture
def qm_month():
    return xr.DataArray(
        np.arange(5 * 12).reshape(5, 12),
        dims=("quantiles", "month"),
        coords={"quantiles": [0, 0.3, 0.5, 0.7, 1], "month": range(1, 13)},
        attrs={"group": "time.month", "window": 1},
    )


@pytest.fixture
def qm_small():
    return xr.DataArray(
        np.arange(2 * 3).reshape(2, 3),
        dims=("quantiles", "month"),
        coords={"quantiles": [0.3, 0.7], "month": range(1, 4)},
        attrs={"group": "time.month", "window": 1},
    )


@pytest.fixture
def make_qm():
    def _make_qm(a, group="time.month"):
        dim, prop = parse_group(group)
        a = np.atleast_2d(a)
        n, m = a.shape
        mo = range(1, m + 1)

        if prop:
            q = equally_spaced_nodes(n, None)
            dims = ("quantiles", prop)
            coords = {"quantiles": q, "month": mo}
        else:
            q = equally_spaced_nodes(m, None)
            dims = ("quantiles",)
            coords = {"quantiles": q}
            a = a[0]

        return xr.DataArray(
            a, dims=dims, coords=coords, attrs={"group": group, "window": 1},
        )

    return _make_qm


@pytest.fixture
def qds_month():
    dims = ("quantiles", "month")
    source = xr.Variable(dims=dims, data=np.zeros((5, 12)))
    target = xr.Variable(dims=dims, data=np.ones((5, 12)) * 2)

    return xr.Dataset(
        data_vars={"source": source, "target": target},
        coords={"quantiles": [0, 0.3, 5.0, 7, 1], "month": range(1, 13)},
        attrs={"group": "time.month", "window": 1},
    )


@pytest.fixture
def obs_sim_fut_tuto():
    """Return obs, sim, fut time series of air temperature."""

    def _obs_sim_fut_tuto(fut_offset=3, delta=0.1, smth_win=3, trend=True):
        ds = xr.tutorial.open_dataset("air_temperature")
        obs = ds.air.resample(time="D").mean()
        sim = obs.rolling(time=smth_win, min_periods=1).mean() + delta
        fut_time = sim.time + np.timedelta64(730 + fut_offset * 365, "D").astype(
            "<m8[ns]"
        )
        fut = sim + (
            0
            if not trend
            else xr.DataArray(
                np.linspace(0, 2, num=sim.time.size),
                dims=("time",),
                coords={"time": sim.time},
            )
        )
        fut["time"] = fut_time
        return obs, sim, fut

    return _obs_sim_fut_tuto

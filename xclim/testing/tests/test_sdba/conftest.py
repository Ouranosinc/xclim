import numpy as np
import pytest
import xarray as xr

from xclim.sdba.base import parse_group
from xclim.sdba.utils import apply_correction, equally_spaced_nodes

from . import utils as tu

# Some test fixtures are useful to have around, so they are implemented as normal python functions and objects in
# utils.py, and converted into fixtures here.
cannon_2015_dist = pytest.fixture(tu.cannon_2015_dist)


@pytest.fixture
def cannon_2015_rvs():
    return tu.cannon_2015_rvs


@pytest.fixture
def series():
    return tu.series


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
    @parse_group
    def _make_qm(a, *, group="time.month"):
        a = np.atleast_2d(a)
        n, m = a.shape
        mo = range(1, m + 1)

        if group.prop:
            q = equally_spaced_nodes(n, None)
            dims = ("quantiles", group.prop)
            coords = {"quantiles": q, "month": mo}
        else:
            q = equally_spaced_nodes(m, None)
            dims = ("quantiles",)
            coords = {"quantiles": q}
            a = a[0]

        return xr.DataArray(
            a,
            dims=dims,
            coords=coords,
            attrs={"group": group, "window": 1},
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
def ref_hist_sim_tuto():
    """Return ref, hist, sim time series of air temperature."""

    def _ref_hist_sim_tuto(sim_offset=3, delta=0.1, smth_win=3, trend=True):
        ds = xr.tutorial.open_dataset("air_temperature")
        ref = ds.air.resample(time="D").mean(keep_attrs=True)
        hist = ref.rolling(time=smth_win, min_periods=1).mean(keep_attrs=True) + delta
        hist.attrs["units"] = ref.attrs["units"]
        sim_time = hist.time + np.timedelta64(730 + sim_offset * 365, "D").astype(
            "<m8[ns]"
        )
        sim = hist + (
            0
            if not trend
            else xr.DataArray(
                np.linspace(0, 2, num=hist.time.size),
                dims=("time",),
                coords={"time": hist.time},
                attrs={"units": hist.attrs["units"]},
            )
        )
        sim["time"] = sim_time
        return ref, hist, sim

    return _ref_hist_sim_tuto

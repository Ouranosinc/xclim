import numpy as np
import pytest
import xarray as xr

sdba = pytest.importorskip("xclim.sdba")  # noqa

from xclim.sdba.base import Grouper
from xclim.sdba.base import Parametrizable
from xclim.sdba.processing import normalize


def test_param_class():
    gr = Grouper(group="time.month")
    in_params = dict(
        anint=4, abool=True, astring="a string", adict={"key": "val"}, group=gr
    )
    obj = Parametrizable(**in_params)

    assert obj.parameters == in_params

    repr(obj).startswith(
        "ParametrizableClass(anint=4, abool=True, astring='a string', adict={'key': 'val'}, "
        "group=Grouper(dim='time',"
    )


@pytest.mark.parametrize(
    "group,window,nvals",
    [("time", 1, 366), ("time.month", 1, 31), ("time.dayofyear", 5, 1)],
)
def test_grouper_group(tas_series, group, window, nvals):
    tas = tas_series(np.ones(366), start="2000-01-01")

    grouper = Grouper(group, window=window)
    grpd = grouper.group(tas)

    if window > 1:
        assert "window" in grpd.dims

    assert grpd.count().max() == nvals


@pytest.mark.parametrize(
    "group,interp,val90",
    [("time", False, True), ("time.month", False, 3), ("time.month", True, 3.5)],
)
def test_grouper_get_index(tas_series, group, interp, val90):
    tas = tas_series(np.ones(366), start="2000-01-01")
    grouper = Grouper(group, interp=interp)
    indx = grouper.get_index(tas)
    # 90 is March 31st
    assert indx[90] == val90


def test_grouper_apply(tas_series):
    tas1 = tas_series(np.arange(366), start="2000-01-01")
    tas0 = tas_series(np.zeros(366), start="2000-01-01")
    tas = xr.concat((tas1, tas0), dim="lat")

    grouper = Grouper("time.month")
    out = grouper.apply("mean", tas)
    assert out.isel(month=0, lat=0) == 15.0
    out = normalize(tas, group=grouper)

    grouper = Grouper("time.month", add_dims=["lat"])
    out = grouper.apply("mean", tas)
    assert out.ndim == 1
    assert out.isel(month=0,) == 7.5
    assert out.attrs["group"] == "time.month"
    assert out.attrs["group_compute_dims"] == ["time", "lat"]
    assert out.attrs["group_window"] == 1

    grouper = Grouper("time.month", window=5)
    out = grouper.apply("mean", tas)
    np.testing.assert_almost_equal(out.isel(month=0, lat=0), 15.32236842)

    tas = tas.chunk({"lat": 1})
    out = grouper.apply("mean", tas)
    assert out.chunks == ((1, 1), (12,))

    out = normalize(tas, group=grouper)
    assert out.chunks == ((1, 1), (366,))

    def mixed_reduce(grdds, dim=None):
        tas1 = grdds.tas1.mean(dim=dim)
        tas0 = grdds.tas0 / grdds.tas0.mean(dim=dim)
        tas1.attrs["_group_apply_reshape"] = True
        return xr.Dataset(data_vars={"tas1_mean": tas1, "norm_tas0": tas0})

    tas1 = tas1.chunk({"time": -1})
    out = grouper.apply(mixed_reduce, {"tas1": tas1, "tas0": tas0})
    assert "month" not in out.norm_tas0.dims
    assert "month" in out.tas1_mean.dims

    assert out.tas1_mean.chunks == ((12,),)
    assert out.norm_tas0.chunks == ((366,),)

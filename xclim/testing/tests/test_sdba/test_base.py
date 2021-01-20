import numpy as np
import pytest
import xarray as xr

from xclim.sdba.base import Grouper, Parametrizable


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


@pytest.mark.slow
@pytest.mark.parametrize(
    "group,n",
    [("time", 1), ("time.month", 12), ("time.week", 52)],
)
@pytest.mark.parametrize("use_dask", [True, False])
def test_grouper_apply(tas_series, use_dask, group, n):
    tas1 = tas_series(np.arange(366), start="2000-01-01")
    tas0 = tas_series(np.zeros(366), start="2000-01-01")
    tas = xr.concat((tas1, tas0), dim="lat")

    grouper = Grouper(group)
    if not group.startswith("time"):
        tas = tas.rename(time=grouper.dim)
        tas1 = tas1.rename(time=grouper.dim)
        tas0 = tas0.rename(time=grouper.dim)

    if use_dask:
        tas = tas.chunk({"lat": 1, grouper.dim: -1})
        tas0 = tas1.chunk({grouper.dim: -1})
        tas1 = tas0.chunk({grouper.dim: -1})

    # Normal monthly mean
    out_mean = grouper.apply("mean", tas)
    if grouper.prop:
        exp = tas.groupby(group).mean()
    else:
        exp = tas.mean(dim=grouper.dim)
    np.testing.assert_array_equal(out_mean, exp)

    # With additionnal dimension included
    grouper = Grouper(group, add_dims=["lat"])
    out = grouper.apply("mean", tas)
    assert out.ndim == int(grouper.prop is not None)
    np.testing.assert_array_equal(out, exp.mean("lat"))
    assert out.attrs["group"] == group
    assert out.attrs["group_compute_dims"] == [grouper.dim, "lat"]
    assert out.attrs["group_window"] == 1

    # Additionnal but main_only
    out = grouper.apply("mean", tas, main_only=True)
    np.testing.assert_array_equal(out, out_mean)

    # With window
    win_grouper = Grouper(group, window=5)
    out = win_grouper.apply("mean", tas)
    rolld = tas.rolling({win_grouper.dim: 5}, center=True).construct(
        window_dim="window"
    )
    if grouper.prop:
        exp = rolld.groupby(group).mean(dim=[win_grouper.dim, "window"])
    else:
        exp = rolld.mean(dim=[grouper.dim, "window"])
    np.testing.assert_array_equal(out, exp)

    # With function + nongrouping-grouped
    grouper = Grouper(group)

    def normalize(grp, dim):
        return grp / grp.mean(dim=dim)

    normed = grouper.apply(normalize, tas)
    assert normed.shape == tas.shape
    if use_dask:
        assert normed.chunks == ((1, 1), (366,))

    # With window + nongrouping-grouped
    out = win_grouper.apply(normalize, tas)
    assert out.shape == tas.shape

    # Mixed output
    def mixed_reduce(grdds, dim=None):
        tas1 = grdds.tas1.mean(dim=dim)
        tas0 = grdds.tas0 / grdds.tas0.mean(dim=dim)
        tas1.attrs["_group_apply_reshape"] = True
        return xr.Dataset(data_vars={"tas1_mean": tas1, "norm_tas0": tas0})

    out = grouper.apply(mixed_reduce, {"tas1": tas1, "tas0": tas0})
    if grouper.prop:
        assert grouper.prop not in out.norm_tas0.dims
        assert grouper.prop in out.tas1_mean.dims

    if use_dask:
        assert out.tas1_mean.chunks == (((n,),) if grouper.prop else tuple())
        assert out.norm_tas0.chunks == ((366,),)

    # Mixed input
    if grouper.prop:

        def normalize_from_precomputed(grpds, dim=None):
            return (grpds.tas / grpds.tas1_mean).mean(dim=dim)

        out = grouper.apply(
            normalize_from_precomputed, {"tas": tas, "tas1_mean": out.tas1_mean}
        ).isel(lat=0)
        exp = normed.groupby(group).mean().isel(lat=0)
        assert grouper.prop in out.dims
        np.testing.assert_array_equal(out, exp)

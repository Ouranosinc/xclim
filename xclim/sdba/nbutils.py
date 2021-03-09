"""Numba-accelerated utils."""
import numpy as np
import xarray as xr
from numba import float32, float64, guvectorize, jit  # , int32, int64


@guvectorize(
    [(float32[:], float32, float32[:]), (float64[:], float64, float64[:])],
    "(n),()->()",
    nopython=True,
)
def _vecquantiles(arr, rnk, res):
    res[0] = np.nanquantile(arr, rnk)


def vecquantiles(da, rnk, dims):
    tem = xr.core.utils.get_temp_dimname(da.dims, "temporal")
    da = da.stack({tem: dims})
    da = da.transpose(*rnk.dims, tem)

    res = xr.DataArray(
        _vecquantiles(da.values, rnk.values),
        dims=rnk.dims,
        coords=rnk.coords,
        attrs=da.attrs,
    )
    return res


@guvectorize(
    [(float32[:], float32[:]), (float64[:], float64[:])], "(n)->(n)", nopython=True
)
def _rank(arr, out):
    out[:] = np.argsort(np.argsort(arr))


def rank(da):
    return da.copy(data=_rank(da.values))


@jit(
    [
        float32[:, :](float32[:, :], float32[:]),
        float64[:, :](float64[:, :], float32[:]),
    ],
    nopython=True,
)
def _quantile(arr, q):
    out = np.empty((arr.shape[0], q.size), dtype=arr.dtype)
    for index in range(out.shape[0]):
        out[index] = np.nanquantile(arr[index], q)
    return out


def quantile(da, q, dims):
    spa = xr.core.utils.get_temp_dimname(da.dims, "spatial")
    tem = xr.core.utils.get_temp_dimname(da.dims, "temporal")
    da = da.stack({spa: set(da.dims) - set(dims), tem: dims})
    da = da.transpose(spa, tem)

    res = xr.DataArray(
        _quantile(da.values, q),
        dims=(spa, "quantiles"),
        coords={spa: da[spa], "quantiles": q},
        attrs=da.attrs,
    )
    return res.unstack(spa)

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


def vecquantiles(da, rnk, dim):
    """For when the quantile (rnk) is different for each point.

    da and rnk must share all dimensions but dim.
    """
    tem = xr.core.utils.get_temp_dimname(da.dims, "temporal")
    dims = [dim] if isinstance(dim, str) else dim
    da = da.stack({tem: dims})
    da = da.transpose(*rnk.dims, tem)

    res = xr.DataArray(
        _vecquantiles(da.values, rnk.values),
        dims=rnk.dims,
        coords=rnk.coords,
        attrs=da.attrs,
    )
    return res


@jit(
    [
        float32[:, :](float32[:, :], float32[:]),
        float64[:, :](float64[:, :], float64[:]),
        float32[:](float32[:], float32[:]),
        float64[:](float64[:], float64[:]),
    ],
    nopython=True,
)
def _quantile(arr, q):
    if arr.ndim == 1:
        out = np.empty((q.size,), dtype=arr.dtype)
        out[:] = np.nanquantile(arr, q)
    else:
        out = np.empty((arr.shape[0], q.size), dtype=arr.dtype)
        for index in range(out.shape[0]):
            out[index] = np.nanquantile(arr[index], q)
    return out


def quantile(da, q, dim):
    """Compute the quantiles from a fixed list "q" """
    # We have two cases :
    # - When all dims are processed : we stack them and use _quantile1d
    # - When the quantiles are vectorized over some dims, these are also stacked and then _quantile2D is used.
    # All this stacking is so we can cover all ND+1D cases with one numba function.

    # Stack the dims and send to the last position
    # This is in case there are more than one
    dims = [dim] if isinstance(dim, str) else dim
    tem = xr.core.utils.get_temp_dimname(da.dims, "temporal")
    da = da.stack({tem: dims})

    # So we cut in half the definitions to declare in numba
    if not hasattr(q, "dtype") or q.dtype != da.dtype:
        q = np.array(q, dtype=da.dtype)

    if len(da.dims) > 1:
        # There are some extra dims
        extra = xr.core.utils.get_temp_dimname(da.dims, "extra")
        da = da.stack({extra: set(da.dims) - {tem}})
        da = da.transpose(..., tem)
        res = xr.DataArray(
            _quantile(da.values, q),
            dims=(extra, "quantiles"),
            coords={extra: da[extra], "quantiles": q},
            attrs=da.attrs,
        ).unstack(extra)

    else:
        # All dims are processed
        res = xr.DataArray(
            _quantile(da.values, q),
            dims=("quantiles"),
            coords={"quantiles": q},
            attrs=da.attrs,
        )

    return res

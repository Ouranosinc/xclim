"""SDBA utilities module."""
from typing import Callable, List, Mapping, Optional, Union
from warnings import warn

import bottleneck as bn
import numpy as np
import xarray as xr
from boltons.funcutils import wraps
from scipy.interpolate import griddata, interp1d

from xclim.core.calendar import _interpolate_doy_calendar
from xclim.core.utils import ensure_chunk_size

from .base import Grouper, parse_group

MULTIPLICATIVE = "*"
ADDITIVE = "+"
loffsets = {"MS": "14d", "M": "15d", "YS": "181d", "Y": "182d", "QS": "45d", "Q": "46d"}


@parse_group
def map_cdf(
    x: xr.DataArray,
    y: xr.DataArray,
    y_value: xr.DataArray,
    *,
    group: Union[str, Grouper] = "time",
    skipna: bool = False,
):
    """Return the value in `x` with the same CDF as `y_value` in `y`.

    Parameters
    ----------
    x : xr.DataArray
      Values from which to pick
    y : xr.DataArray
      Reference values giving the ranking
    y_value : float, array
      Value within the support of `y`.
    dim : str
      Dimension along which to compute quantile.
    group: Union[str, Grouper]
    skipna: bool

    Returns
    -------
    array
      Quantile of `x` with the same CDF as `y_value` in `y`.
    """

    def _map_cdf_1d(x, y, y_value, skipna=False):
        q = _ecdf_1d(y, y_value)
        _func = np.nanquantile if skipna else np.quantile
        return _func(x, q=q)

    def _map_cdf_group(gr, y_value, dim=["time"], skipna=False):
        return xr.apply_ufunc(
            _map_cdf_1d,
            gr.x,
            gr.y,
            input_core_dims=[dim] * 2,
            output_core_dims=[["x"]],
            vectorize=True,
            keep_attrs=True,
            kwargs={"y_value": y_value, "skipna": skipna},
            dask="parallelized",
            output_dtypes=[gr.x.dtype],
        )

    return group.apply(
        _map_cdf_group,
        {"x": x, "y": y},
        y_value=np.atleast_1d(y_value),
        skipna=skipna,
    )


def _ecdf_1d(x, value):
    sx = np.r_[-np.inf, np.sort(x)]
    return np.searchsorted(sx, value, side="right") / np.sum(~np.isnan(sx))


def ecdf(x: xr.DataArray, value: float, dim: str = "time"):
    """Return the empirical CDF of a sample at a given value.

    Parameters
    ----------
    x : array
      Sample.
    value : float
      The value within the support of `x` for which to compute the CDF value.

    Returns
    -------
    array
      Empirical CDF.
    """
    return (x <= value).sum(dim) / x.notnull().sum(dim)


def ensure_longest_doy(func: Callable) -> Callable:
    """Ensure that selected day is the longest day of year for x and y dims."""

    @wraps(func)
    def _ensure_longest_doy(x, y, *args, **kwargs):
        if (
            hasattr(x, "dims")
            and hasattr(y, "dims")
            and "dayofyear" in x.dims
            and "dayofyear" in y.dims
            and x.dayofyear.max() != y.dayofyear.max()
        ):
            warn(
                (
                    "get_correction received inputs defined on different dayofyear ranges. "
                    "Interpolating to the longest range. Results could be strange."
                ),
                stacklevel=4,
            )
            if x.dayofyear.max() < y.dayofyear.max():
                x = _interpolate_doy_calendar(x, int(y.dayofyear.max()))
            else:
                y = _interpolate_doy_calendar(y, int(x.dayofyear.max()))
        return func(x, y, *args, **kwargs)

    return _ensure_longest_doy


@ensure_longest_doy
def get_correction(x: xr.DataArray, y: xr.DataArray, kind: str):
    """Return the additive or multiplicative correction/adjustment factors."""
    with xr.set_options(keep_attrs=True):
        if kind == ADDITIVE:
            out = y - x
        elif kind == MULTIPLICATIVE:
            out = y / x
        else:
            raise ValueError("kind must be + or *.")

    if isinstance(out, xr.DataArray):
        out.attrs["kind"] = kind
    return out


@ensure_longest_doy
def apply_correction(x: xr.DataArray, factor: xr.DataArray, kind: Optional[str] = None):
    """Apply the additive or multiplicative correction/adjustment factors.

    If kind is not given, default to the one stored in the "kind" attribute of factor.
    """
    kind = kind or factor.get("kind", None)
    with xr.set_options(keep_attrs=True):
        if kind == ADDITIVE:
            out = x + factor
        elif kind == MULTIPLICATIVE:
            out = x * factor
        else:
            raise ValueError
    return out


def invert(x: xr.DataArray, kind: Optional[str] = None):
    """Invert a DataArray either additively (-x) or multiplicatively (1/x).

    If kind is not given, default to the one stored in the "kind" attribute of x.
    """
    kind = kind or x.get("kind", None)
    with xr.set_options(keep_attrs=True):
        if kind == ADDITIVE:
            return -x
        if kind == MULTIPLICATIVE:
            return 1 / x
        raise ValueError


@parse_group
def broadcast(
    grouped: xr.DataArray,
    x: xr.DataArray,
    *,
    group: Union[str, Grouper] = "time",
    interp: str = "nearest",
    sel: Optional[Mapping[str, xr.DataArray]] = None,
):
    """Broadcast a grouped array back to the same shape as a given array.

    Parameters
    ----------
    grouped : xr.DataArray
      The grouped array to broadcast like `x`.
    x : xr.DataArray
      The array to broadcast grouped to.
    group : Union[str, Grouper]
      Grouping information. See :py:class:`xclim.sdba.base.Grouper` for details.
    interp : {'nearest', 'linear', 'cubic'}
      The interpolation method to use,
    sel : Mapping[str, xr.DataArray]
      Mapping of grouped coordinates to x coordinates (other than the grouping one).
    """
    if sel is None:
        sel = {}

    if group.prop is not None and group.prop not in sel:
        sel.update({group.prop: group.get_index(x, interp=interp)})

    if sel:
        # Extract the correct mean factor for each time step.
        if interp == "nearest":  # Interpolate both the time group and the quantile.
            grouped = grouped.sel(sel, method="nearest")
        else:  # Find quantile for nearest time group and quantile.
            if group.prop is not None:
                grouped = add_cyclic_bounds(grouped, group.prop, cyclic_coords=False)

            if interp == "cubic" and len(sel.keys()) > 1:
                interp = "linear"
                warn(
                    "Broadcasting operations in multiple dimensions can only be done with linear and nearest-neighbor interpolation, not cubic. Using linear."
                )

            grouped = grouped.interp(sel, method=interp)

        for var in sel.keys():
            if var in grouped.coords and var not in grouped.dims:
                grouped = grouped.drop_vars(var)

    return grouped


def equally_spaced_nodes(n: int, eps: Union[float, None] = 1e-4):
    """Return nodes with `n` equally spaced points within [0, 1] plus two end-points.

    Parameters
    ----------
    n : int
      Number of equally spaced nodes.
    eps : float, None
      Distance from 0 and 1 of end nodes. If None, do not add endpoints.

    Returns
    -------
    array
      Nodes between 0 and 1.

    Notes
    -----
    For n=4, eps=0 :  0---x------x------x------x---1
    """
    dq = 1 / n / 2
    q = np.linspace(dq, 1 - dq, n)
    if eps is None:
        return q
    return sorted(np.append([eps, 1 - eps], q))


def add_cyclic_bounds(da: xr.DataArray, att: str, cyclic_coords: bool = True):
    """Reindex an array to include the last slice at the beginning and the first at the end.

    This is done to allow interpolation near the end-points.

    Parameters
    ----------
    da : Union[xr.DataArray, xr.Dataset]
        An array
    att : str
        The name of the coordinate to make cyclic
    cyclic_coords : bool
        If True, the coordinates are made cyclic as well,
        if False, the new values are guessed using the same step as their neighbour.

    Returns
    -------
    Union[xr.DataArray, xr.Dataset]
        da but with the last element along att prepended and the last one appended.
    """
    qmf = da.pad({att: (1, 1)}, mode="wrap")

    if not cyclic_coords:
        vals = qmf.coords[att].values
        diff = da.coords[att].diff(att)
        vals[0] = vals[1] - diff[0]
        vals[-1] = vals[-2] + diff[-1]
        qmf = qmf.assign_coords({att: vals})
        qmf[att].attrs.update(da.coords[att].attrs)
    return ensure_chunk_size(qmf, **{att: -1})


def extrapolate_qm(qf: xr.DataArray, xq: xr.DataArray, method: str = "constant"):
    """Extrapolate quantile adjustment factors beyond the computed quantiles.

    Parameters
    ----------
    qf : xr.DataArray
      Adjustment factors over `quantile` coordinates.
    xq : xr.DataArray
      Values at each `quantile`.
    method : {"constant"}
      Extrapolation method. See notes below.

    Returns
    -------
    xr.Dataset
        Extrapolated adjustment factors and x-values.

    Notes
    -----
    nan
      Estimating values above or below the computed values will return a NaN.
    constant
      The adjustment factor above and below the computed values are equal to the last and first values
      respectively.
    """
    # constant_iqr
    #   Same as `constant`, but values are set to NaN if farther than one interquartile range from the min and max.
    if method == "nan":
        return qf, xq

    if method == "constant":
        q_l, q_r = [0], [1]
        x_l, x_r = [-np.inf], [np.inf]
        qf_l, qf_r = qf.isel(quantiles=0), qf.isel(quantiles=-1)

    elif (
        method == "constant_iqr"
    ):  # This won't work because add_endpoints does not support mixed y (float and DA)
        raise NotImplementedError
        # iqr = np.diff(xq.interp(quantile=[0.25, 0.75]))[0]
        # ql, qr = [0, 0], [1, 1]
        # xl, xr = [-np.inf, xq.isel(quantile=0) - iqr], [xq.isel(quantile=-1) + iqr, np.inf]
        # qml, qmr = [np.nan, qm.isel(quantile=0)], [qm.isel(quantile=-1), np.nan]
    else:
        raise ValueError

    qf = add_endpoints(qf, left=[q_l, qf_l], right=[q_r, qf_r])
    xq = add_endpoints(xq, left=[q_l, x_l], right=[q_r, x_r])
    return qf, xq


def add_endpoints(
    da: xr.DataArray,
    left: List[Union[int, float, xr.DataArray, List[int], List[float]]],
    right: List[Union[int, float, xr.DataArray, List[int], List[float]]],
    dim: str = "quantiles",
):
    """Add left and right endpoints to a DataArray.

    Parameters
    ----------
    da : DataArray
      Source array.
    left : [x, y]
      Values to prepend
    right : [x, y]
      Values to append.
    dim : str
      Dimension along which to add endpoints.
    """
    elems = []
    for (x, y) in (left, right):
        if isinstance(y, xr.DataArray):
            if "quantiles" not in y.dims:
                y = y.expand_dims("quantiles")
            y = y.assign_coords(quantiles=x)
        else:
            y = xr.DataArray(y, coords={dim: x}, dims=(dim,))
        elems.append(y)
    l, r = elems  # pylint: disable=unbalanced-tuple-unpacking
    out = xr.concat((l, da, r), dim=dim)
    return ensure_chunk_size(out, **{dim: -1})


@parse_group
def interp_on_quantiles(
    newx: xr.DataArray,
    xq: xr.DataArray,
    yq: xr.DataArray,
    *,
    group: Union[str, Grouper] = "time",
    method: str = "linear",
):
    """Interpolate values of yq on new values of x.

    Interpolate in 2D if grouping is used, in 1D otherwise.

    Parameters
    ----------
    newx : xr.DataArray
        The values at wich to evalute `yq`. If `group` has group information,
        `new` should have a coordinate with the same name as the group name
         In that case, 2D interpolation is used.
    xq, yq : xr.DataArray
        coordinates and values on which to interpolate. The interpolation is done
        along the "quantiles" dimension if `group` has no group information.
        If it does, interpolation is done in 2D on "quantiles" and on the group dimension.
    group : Union[str, Grouper]
        The dimension and grouping information. (ex: "time" or "time.month").
        Defaults to the "group" attribute of xq, or "time" if there is none.
    method : {'nearest', 'linear', 'cubic'}
        The interpolation method.
    """
    dim = group.dim
    prop = group.prop

    if prop is None:
        fill_value = "extrapolate" if method == "nearest" else np.nan

        def _interp_quantiles_1D(newx, oldx, oldy):
            return interp1d(
                oldx, oldy, bounds_error=False, kind=method, fill_value=fill_value
            )(newx)

        return xr.apply_ufunc(
            _interp_quantiles_1D,
            newx,
            xq,
            yq,
            input_core_dims=[[dim], ["quantiles"], ["quantiles"]],
            output_core_dims=[[dim]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[np.float],
        )
    # else:

    def _interp_quantiles_2D(newx, newg, oldx, oldy, oldg):
        if method != "nearest":
            oldx = np.clip(oldx, newx.min() - 1, newx.max() + 1)
        if np.all(np.isnan(newx)):
            warn(
                "All-NaN slice encountered in interp_on_quantiles",
                category=RuntimeWarning,
            )
            return newx
        return griddata(
            (oldx.flatten(), oldg.flatten()),
            oldy.flatten(),
            (newx, newg),
            method=method,
        )

    xq = add_cyclic_bounds(xq, prop, cyclic_coords=False)
    yq = add_cyclic_bounds(yq, prop, cyclic_coords=False)
    newg = group.get_index(newx)
    oldg = xq[prop].expand_dims(quantiles=xq.coords["quantiles"])

    return xr.apply_ufunc(
        _interp_quantiles_2D,
        newx,
        newg,
        xq,
        yq,
        oldg,
        input_core_dims=[
            [dim],
            [dim],
            [prop, "quantiles"],
            [prop, "quantiles"],
            [prop, "quantiles"],
        ],
        output_core_dims=[[dim]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[yq.dtype],
    )


def rank(da, dim="time", pct=False):
    """Ranks data.

    Replicates `xr.DataArray.rank` but with support for dask-stored data. Xarray's docstring is below:

    Equal values are assigned a rank that is the average of the ranks that
    would have been otherwise assigned to all of the values within that
    set.  Ranks begin at 1, not 0. If pct, computes percentage ranks.

    NaNs in the input array are returned as NaNs.

    The `bottleneck` library is required.

    Parameters
    ----------
    dim : hashable
        Dimension over which to compute rank.
    pct : bool, optional
        If True, compute percentage ranks, otherwise compute integer ranks.

    Returns
    -------
    ranked : DataArray
        DataArray with the same coordinates and dtype 'float64'.
    """

    def _nanrank(data):
        func = bn.nanrankdata if data.dtype.kind == "f" else bn.rankdata
        ranked = func(data, axis=-1)
        if pct:
            count = np.sum(~np.isnan(data), axis=-1, keepdims=True)
            ranked /= count
        return ranked

    return xr.apply_ufunc(
        _nanrank,
        da,
        input_core_dims=[[dim]],
        output_core_dims=[[dim]],
        dask="parallelized",
        output_dtypes=[da.dtype],
    )

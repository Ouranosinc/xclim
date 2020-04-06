from warnings import warn

import numpy as np
import xarray as xr
from scipy.interpolate import griddata
from scipy.interpolate import interp1d

from .base import Grouper


MULTIPLICATIVE = "*"
ADDITIVE = "+"
loffsets = {"MS": "14d", "M": "15d", "YS": "181d", "Y": "182d", "QS": "45d", "Q": "46d"}


def map_cdf(x, y, y_value, group, skipna=False):
    """Return the value in `x` with the same CDF as `y_value` in `y`.

    Parameters
    ----------
    x : xr.DataArray
      Training target.
    y : xr.DataArray
      Training data.
    y_value : float, array
      Value within the support of `y`.
    dim : str
      Dimension along which to compute quantile.

    Returns
    -------
    array
      Quantile of `x` with the same CDF as `y_value` in `y`.
    """

    def _map_cdf_1d(x, y, y_value, skipna=False):
        q = _ecdf_1d(y, y_value)
        _func = np.nanquantile if skipna else np.quantile
        return _func(x, q=q)

    def _map_cdf_group(gr, y_value, dim="time", skipna=False):
        return xr.apply_ufunc(
            _map_cdf_1d,
            gr.x,
            gr.y,
            input_core_dims=[[dim]] * 2,
            output_core_dims=[["x"]],
            vectorize=True,
            keep_attrs=True,
            kwargs={"y_value": y_value, "skipna": skipna},
            dask="parallelized",
            output_dtypes=[gr.x.dtype],
        )

    return group_apply(
        _map_cdf_group,
        xr.Dataset(data_vars={"x": x, "y": y}),
        group,
        y_value=np.atleast_1d(y_value),
        skipna=skipna,
    )


def _ecdf_1d(x, value):
    sx = np.r_[-np.inf, np.sort(x)]
    return np.searchsorted(sx, value, side="right") / np.sum(~np.isnan(sx))


def ecdf(x, value, dim="time"):
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


def parse_group(group):
    """Return dimension and property."""
    if "." in group:
        return group.split(".")
    else:
        return group, None


# TODO: When ready this should be a method of a Grouping object
def group_apply(func, x, group, window=1, grouped_args=None, **kwargs):
    """Group values by time, then compute function.

    Parameters
    ----------
    func : str
      DataArray method applied to each group.
    x : DataArray, tuple of DataArray for functions with multiple arguments.
      Data.
    group : {'time.season', 'time.month', 'time.dayofyear', 'time'}
      Grouping criterion. If only coordinate is given (e.g. 'time') no grouping will be done.
    window : int
      Length of the rolling window centered around the time of interest used to estimate the quantiles. This is mostly
      useful for time.dayofyear grouping.
    grouped_args : Sequence of DataArray
      Args passed here are results from a previous groupby that contain the "prop" dim, but not "dim" (ex: "month", but not "time")
      Before func is called on a group, the corresponding slice of each grouped_args will be extracted and passed as args to func.
      Useful for using precomputed results.
    **kwargs : dict
      Arguments passed to function.

    Returns
    -------
    Union[xr.DataArray, xr.Dataset]
        Which ever is returned by func. With added attributes `group` and `group_window`.
    """
    dim, prop = parse_group(group)

    if isinstance(x, (tuple, list)):
        x = xr.Dataset({f"v{i}": da for i, da in enumerate(x)})

    dims = dim
    if "." in group:
        if window > 1:
            # Construct rolling window
            x = x.rolling(center=True, **{dim: window}).construct(window_dim="window")
            dims = ("window", dim)

        sub = x.groupby(group)

    else:
        sub = x

    def wrap_func_with_grouped_args(func):
        def call_func_with_grouped_element(dsgr, *grouped, **kwargs):
            # For each element in grouped, we extract the correspong slice for the current group
            # TODO: Is there any better way to get the label of the current group??
            if prop is not None:
                label = getattr(dsgr[dim][0].dt, prop)
            else:
                label = dsgr[group][0]
            elements = [arg.sel({prop or group: label}) for arg in grouped]
            return func(dsgr, *elements, **kwargs)

        return call_func_with_grouped_element

    if isinstance(func, str):
        out = getattr(sub, func)(dim=dims, **kwargs)
    else:
        if grouped_args is not None:
            func = wrap_func_with_grouped_args(func)
        if isinstance(sub, xr.core.groupby.GroupBy):
            out = sub.map(func, args=grouped_args or [], dim=dims, **kwargs)
        else:
            out = func(sub, dim=dims, **kwargs)

    # Case where the function wants to return more than one variables
    # and that some have grouped dims and other have the same dimensions as the input.
    # In that specific case, groupby broadcasts everything back to the input's dim, copying the grouped data.
    if isinstance(out, xr.Dataset):
        for name, da in out.data_vars.items():
            if "_group_apply_reshape" in da.attrs:
                if da.attrs["_group_apply_reshape"] and prop is not None:
                    out[name] = da.groupby(group).first(skipna=False, keep_attrs=True)
                del out[name].attrs["_group_apply_reshape"]

    # Save input parameters as attributes of output DataArray.
    out.attrs["group"] = group
    out.attrs["group_window"] = window

    # If the grouped operation did not reduce the array, the result is sometimes unsorted along dim
    if dim in out.dims:
        out = out.sortby(dim)
    return out


def get_correction(x, y, kind):
    """Return the additive or multiplicative correction factor."""
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


def broadcast(grouped, x, group=None, interp="nearest", sel=None):
    if not hasattr(group, "dim"):
        group = Grouper(group or grouped.group)

    if sel is None:
        sel = {}

    if group.prop is not None and group.prop not in sel:
        sel.update({group.prop: group.get_index(x)})

    if sel:
        # Extract the correct mean factor for each time step.
        if interp == "nearest":  # Interpolate both the time group and the quantile.
            grouped = grouped.sel(sel, method="nearest")
        else:  # Find quantile for nearest time group and quantile.
            if group.prop is not None:
                grouped = add_cyclic_bounds(grouped, group.prop, cyclic_coords=False)

            if interp == "cubic" and len(sel.keys) > 1:
                interp = "linear"
                warn(
                    "Broadcasting operations in multiple dimensions can only be done with linear and nearest-neighbor interpolation, not cubic. Using linear."
                )
            grouped = grouped.interp(sel, method=interp)

        for var in sel.keys():
            if var in grouped.coords and var not in grouped.dims:
                grouped = grouped.drop_vars(var)

    return grouped


def apply_correction(x, factor, kind=None):
    kind = kind or factor.get("kind", None)
    with xr.set_options(keep_attrs=True):
        if kind == ADDITIVE:
            out = x + factor
        elif kind == MULTIPLICATIVE:
            out = x * factor
        else:
            raise ValueError
    return out


def invert(x, kind):
    kind = kind or x.get("kind", None)
    with xr.set_options(keep_attrs=True):
        if kind == ADDITIVE:
            return -x
        elif kind == MULTIPLICATIVE:
            return 1 / x
        else:
            raise ValueError


def equally_spaced_nodes(n, eps=1e-4):
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


def add_cyclic_bounds(da, att, cyclic_coords=True):
    """Reindex an array to include the last slice at the beginning
    and the first at the end.

    This is done to allow interpolation near the end-points.

    Parameters
    ----------
    da : Union[xr.DataArray, xr.Dataset]
        An array or a dataset
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
    return qmf


# TODO: use xr.pad once it's implemented.
# Rename to extrapolate_q ?
# TODO: improve consistency with extrapolate_qm
def add_q_bounds(qmf, method="constant"):
    """Reindex the scaling factors to set the quantile at 0 and 1 to the first and last quantile respectively.

    This is a naive approach that won't work well for extremes.
    """
    att = "quantile"
    q = qmf.coords[att]
    i = np.concatenate(([0], range(len(q)), [-1]))
    qmf = qmf.reindex({att: q[i]})
    if method == "constant":
        qmf.coords[att] = np.concatenate(([0], q, [1]))
    else:
        raise ValueError
    return qmf


def get_index(da, dim, prop, interp):
    # Compute the `dim` value for indexing along grouping dimension.
    # TODO: Adjust for different calendars if necessary.

    if prop == "season" and interp:
        raise NotImplementedError

    ind = da.indexes[dim]
    i = getattr(ind, prop)

    if interp != "nearest":
        if dim == "time":
            if prop == "month":
                i = ind.month - 0.5 + ind.day / ind.daysinmonth
            elif prop == "dayofyear":
                i = ind.dayofyear
            else:
                raise NotImplementedError

    xi = xr.DataArray(
        i, dims=dim, coords={dim: da.coords[dim]}, name=dim + " group index"
    )

    # Expand dimensions of index to match the dimensions of xq
    # We want vectorized indexing with no broadcasting
    return xi.expand_dims(**{k: v for (k, v) in da.coords.items() if k != dim})


def extrapolate_qm(qf, xq, method="constant"):
    """Extrapolate quantile correction factors beyond the computed quantiles.

    Parameters
    ----------
    qf : xr.DataArray
      Correction factors over `quantile` coordinates.
    xq : xr.DataArray
      Values at each `quantile`.
    method : {"constant"}
      Extrapolation method. See notes below.

    Returns
    -------
    xr.Dataset
        Extrapolated correction factors and x-values.

    Notes
    -----
    nan
      Estimating values above or below the computed values will return a NaN.
    constant
      The correction factor above and below the computed values are equal to the last and first values
      respectively.
    constant_iqr
      Same as `constant`, but values are set to NaN if farther than one interquartile range from the min and max.
    """
    if method == "nan":
        return qf, xq

    elif method == "constant":
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


def add_endpoints(da, left, right, dim="quantiles"):
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
    l, r = elems
    return xr.concat((l, da, r), dim=dim)


def interp_on_quantiles(newx, xq, yq, group=None, method="linear"):
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
    }
    """
    if not hasattr(group, "dim"):
        group = Grouper(group or xq.attrs.get("group", "time"))
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
        output_dtypes=[np.float],
    )

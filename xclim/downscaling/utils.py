import numpy as np
import xarray as xr
from scipy.interpolate import griddata
from scipy.stats import gamma


MULTIPLICATIVE = "*"
ADDITIVE = "+"


def _adjust_freq_1d(sm, ob, thresh=0):
    """Adjust freq on 1D timeseries"""
    # Frequency of values below threshold in obs
    o = ob[~np.isnan(ob)]
    s = sm[~np.isnan(sm)]

    below = o < thresh
    ndO = below.sum().astype(int)  # Number of dry days in obs

    # Target number of values below threshold in sim to match obs frequency
    ndSo = np.ceil(s.size * ndO / o.size).astype(int)

    # Sort sim values
    ind_sort = np.argsort(s)
    Ss = s[ind_sort]

    # Last precip value in sim that should be 0
    # if ndSo > 0:
    #     Sth = Ss[min(ndSo, s.size) - 1]
    # else:
    #     Sth = np.nan

    # Where precip values are under thresh but shouldn't (Indices Wrong)
    iw = np.where(Ss[ndSo:] < thresh)[0] + ndSo

    if iw.size > 0:
        Os = np.sort(o)
        # Linear mapping between Os and Ss if size don't match
        iwO = np.ceil(o.size * iw.max() / s.size).astype(int)
        # Values in obs corresponding to small precips
        auxo = Os[ndO : iwO + 1]
        if np.unique(auxo).size > 6:
            gamA, gamLoc, gamShape = gamma.fit(auxo)
            Ss[ndSo : iw.max() + 1] = gamma.rvs(
                gamA, loc=gamLoc, scale=gamShape, size=iw.size
            )
        else:
            Ss[ndSo : iw.max() + 1] = auxo.mean()
        Ss = np.sort(Ss)

    if ndO > 0:
        Ss[:ndSo] = 0

    ind_unsort = np.empty_like(ind_sort)
    ind_unsort[ind_sort] = np.arange(ind_sort.size)

    out = np.full_like(sm, np.nan)
    out[~np.isnan(sm)] = Ss[ind_unsort]
    return out


def _adjust_freq_group(gr, thresh=0, dim="time"):
    """Adjust freq on group"""
    return xr.apply_ufunc(
        _adjust_freq_1d,
        gr.sim,
        gr.obs,
        input_core_dims=[[dim]] * 2,
        output_core_dims=[[dim]],
        vectorize=True,
        keep_attrs=True,
        kwargs={"thresh": thresh},
        dask="parallelized",
        output_dtypes=[gr.sim.dtype],
    )


def adjust_freq(obs, sim, thresh, group):
    """
    Adjust frequency of values under thresh of sim, based on obs

    Parameters
    ----------
    obs : xr.DataArray
      Observed data.
    sim : xr.DataArray
      Simulated data.
    thresh : float
      Threshold below which values are considered null.

    Returns
    -------
    adjusted_sim : xr.DataArray
        Simulated data with the same frequency of values under threshold than obs.
        Adjustement is made group-wise.

    References
    ----------
    Themeßl et al. (2012)
    """
    return group_apply(
        _adjust_freq_group,
        xr.Dataset(data_vars={"sim": sim, "obs": obs}),
        group,
        thresh=thresh,
    )


def parse_group(group):
    """Return dimension and property."""
    if "." in group:
        return group.split(".")
    else:
        return group, None


def group_apply(func, x, group, window=1, grouped_args=None, **kwargs):
    """Group values by time, then compute function.

    Parameters
    ----------
    func : str
      DataArray method applied to each group.
    x : DataArray
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

    """
    dim, prop = parse_group(group)

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
        out = sub.map(func, args=grouped_args or [], dim=dims, **kwargs)

    # Save input parameters as attributes of output DataArray.
    out.attrs["group"] = group
    out.attrs["window"] = window

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

    out.attrs["kind"] = kind
    return out


def broadcast(grouped, x, interp=False, sel=None):
    dim, prop = parse_group(grouped.group)

    if sel is None:
        sel = {}

    if prop is not None and prop not in sel:
        sel.update({prop: get_index(x, dim, prop, interp)})

    if sel:
        # Extract the correct mean factor for each time step.
        if interp:  # Interpolate both the time group and the quantile.
            return grouped.interp(sel)
        else:  # Find quantile for nearest time group and quantile.
            return grouped.sel(sel, method="nearest")
    else:
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

    out.attrs["bias_corrected"] = True
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


def nodes(n, eps=1e-4):
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
    For nq=4, eps=0 :  0---x------x------x------x---1
    """
    dq = 1 / n / 2
    q = np.linspace(dq, 1 - dq, n)
    if eps is None:
        return q
    return sorted(np.append([eps, 1 - eps], q))


# TODO: use xr.pad once it's implemented.
def add_cyclic(da, att):
    """Reindex the scaling factors to include the last time grouping
    at the beginning and the first at the end.

    This is done to allow interpolation near the end-points.
    """
    gc = da.coords[att]
    i = np.concatenate(([-1], range(len(gc)), [0]))
    qmf = da.reindex({att: gc[i]})
    qmf.coords[att] = range(len(i))
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

    if interp:
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


def reindex(qm, xq, extrapolation="constant"):
    """Create a mapping between x values and y values based on their respective quantiles.

    Parameters
    ----------
    qm : xr.DataArray
      Quantile correction factors.
    xq : xr.DataArray
      Quantiles for source array (historical simulation).
    extrapolation : {"constant"}
      Method to extrapolate outside the estimated quantiles.

    Returns
    -------
    xr.DataArray
      Quantile correction factors whose quantile coordinates have been replaced by corresponding x values.


    Notes
    -----
    The original qm object has `quantile` coordinates and some grouping coordinate (e.g. month). This function
    reindexes the array based on the values of x, instead of the quantiles. Since the x values are different from
    group to group, the index can get fairly large.
    """
    dim, prop = parse_group(xq.group)
    if prop is None:
        q, x = extrapolate_qm(qm, xq, extrapolation)
        out = q.rename({"quantile": "x"}).assign_coords(x=x.values)

    else:
        # Interpolation from quantile to values.
        def func(d):
            q, x = extrapolate_qm(d.qm, d.xq, extrapolation)
            return xr.DataArray(
                dims="x",
                data=np.interp(newx, x, q, left=np.nan, right=np.nan),
                coords={"x": newx},
            )

        ds = xr.Dataset({"xq": xq, "qm": qm})
        gr = ds.groupby(prop)

        # X coordinates common to all groupings
        xs = list(map(lambda g: extrapolate_qm(g[1].qm, g[1].xq, extrapolation)[1], gr))
        newx = np.unique(np.concatenate(xs))
        out = gr.map(func, shortcut=True)

    out.attrs = qm.attrs
    out.attrs["quantile"] = qm.coords["quantile"].values
    return out


def extrapolate_qm(qm, xq, method="constant"):
    """Extrapolate quantile correction factors beyond the computed quantiles.

    Parameters
    ----------
    qm : xr.DataArray
      Correction factors over `quantile` coordinates.
    xq : xr.DataArray
      Values at each `quantile`.
    method : {"constant"}
      Extrapolation method. See notes below.

    Returns
    -------
    DataArray, DataArray
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
        return qm, xq

    elif method == "constant":
        q_l, q_r = [0,], [1,]
        x_l, x_r = [-np.inf,], [np.inf,]
        qm_l, qm_r = qm.isel(quantile=0), qm.isel(quantile=-1)

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

    qm = add_endpoints(qm, left=[x_l, qm_l], right=[x_r, qm_r])
    xq = add_endpoints(xq, left=[q_l, x_l], right=[q_r, x_r])

    return qm, xq


def add_endpoints(da, left, right, dim="quantile"):
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
            if "quantile" not in y.dims:
                y = y.expand_dims("quantile")
            y.assign_coords(quantile=x)
        else:
            y = xr.DataArray(y, coords={dim: x}, dims=(dim,))
        elems.append(y)
    l, r = elems
    return xr.concat((l, da, r), dim=dim)


# TODO: Would need to set right and left values.
def interp_quantiles(x, g, xq, yq, dim="time", group=None, method="linear"):
    def _interp_quantiles_2D(newx, newg, oldx, oldg, oldy):
        if newx.ndim > 1:
            out = np.empty_like(newx)
            for idx in np.ndindex(*newx.shape[:-1]):
                out[idx] = _interp_quantiles_2D(
                    newx[idx], newg, oldx[idx], oldg, oldy[idx]
                )
            return out

        return griddata(
            (oldx.flatten(), oldg.flatten()),
            oldy.flatten(),
            (newx, newg),
            method=method,
        )

    return xr.apply_ufunc(
        _interp_quantiles_2D,
        x,
        g,
        xq,
        xq[group].expand_dims(quantile=xq.coords["quantile"]),
        yq,
        input_core_dims=[
            [dim],
            [dim],
            [group, "quantile"],
            [group, "quantile"],
            [group, "quantile"],
        ],
        output_core_dims=[[dim]],
        dask="parallelized",
        output_dtypes=[np.float],
    )


def jitter_under_thresh(x, thresh):
    """Add a small noise to values smaller than threshold."""
    epsilon = np.finfo(x.dtype).eps
    jitter = np.random.uniform(low=epsilon, high=thresh, size=x.shape)
    return x.where(~(x < thresh & x.notnull()), jitter)

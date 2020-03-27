import numpy as np
import xarray as xr
from scipy.interpolate import griddata
from scipy.interpolate import LinearNDInterpolator
from scipy.stats import gamma


MULTIPLICATIVE = "*"
ADDITIVE = "+"


# TODO: This function should also return sth
def _adjust_freq_1d(sm, ob, thresh=0):
    """Adjust frequency of null values, where values are considered null if below a threshold.

    Assuming we want to map sm to ob, but that there are proportionally more null values in sm than in obs. Null
    values in sm cannot be converted into non-null values by multiplying by a factor.

    Parameters
    ----------
    sm : np.array
    """
    o = ob[~np.isnan(ob)]
    s = sm[~np.isnan(sm)]

    # Frequency of values below threshold in obs
    n_dry_o = (o < thresh).sum()  # Number of dry days in obs

    # Target number of values below threshold in sim to match obs frequency
    n_dry_sp = np.ceil(s.size * n_dry_o / o.size).astype(int)

    # Sort sim values, storing the index to reorder the values later.
    ind_sort = np.argsort(s)
    ss = s[ind_sort]  # sorted s

    # Last precip value in sim that should be 0
    if n_dry_sp > 0:
        sth = ss[min(n_dry_sp, s.size) - 1]  # noqa
    else:
        sth = np.nan  # noqa

    # Where precip values are under thresh but shouldn't. iw: indices wrong
    iw = np.where(ss[n_dry_sp:] < thresh)[0] + n_dry_sp

    # More zeros in sim than in obs: need to create non-zero sims
    if iw.size > 0:
        so = np.sort(o)  # sorted o

        # Linear mapping between sorted o and sorted s if size don't match
        iw_max = np.ceil(o.size * iw.max() / s.size).astype(int)

        # Values in obs corresponding to small precips
        auxo = so[n_dry_o : iw_max + 1]

        # Generate new values matching those small obs
        if np.unique(auxo).size > 6:
            params = gamma.fit(auxo)
            ss[n_dry_sp : iw.max() + 1] = gamma.rvs(*params, size=iw.size)
            # Sometimes we generate values lower than sth... problematic ?
            # if np.any(ss[n_dry_sp : iw.max() + 1] < sth):
            #    raise ValueError
        else:
            ss[n_dry_sp : iw.max() + 1] = auxo.mean()

        # TODO: This additional sort wrecks the original sorting order.
        ss = np.sort(ss)

    # Less zeros in sim than obs: simply set sim values to 0
    if n_dry_o > 0:
        ss[:n_dry_sp] = 0

    # Reorder sim
    out = np.full_like(sm, np.nan)
    ind_unsort = np.empty_like(ind_sort)
    ind_unsort[ind_sort] = np.arange(ind_sort.size)
    out[~np.isnan(sm)] = ss[ind_unsort]
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
    Adjust frequency of values under thresh of sim, based on obs.


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
    Themeßl et al. (2012), Empirical-statistical downscaling and error correction of regional climate models and its
    impact on the climate change signal, Climatic Change, DOI 10.1007/s10584-011-0224-4.
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


# TODO: When ready this should be a method of a Grouping object
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
        if hasattr(sub, "map"):
            out = sub.map(func, args=grouped_args or [], dim=dims, **kwargs)
        else:
            out = func(sub, dim=dims, **kwargs)

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


def broadcast(grouped, x, interp=False, sel=None):
    dim, prop = parse_group(grouped.group)

    if sel is None:
        sel = {}

    if prop is not None and prop not in sel:
        sel.update({prop: get_index(x, dim, prop, interp)})

    if sel:
        # Extract the correct mean factor for each time step.
        if interp:  # Interpolate both the time group and the quantile.
            grouped = grouped.interp(sel)
        else:  # Find quantile for nearest time group and quantile.
            grouped = grouped.sel(sel, method="nearest")

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
        diff = da.coods[att].diff(att)
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


# def reindex(qm, xq, extrapolation="constant"):
#     """Create a mapping between x values and y values based on their respective quantiles.

#     Parameters
#     ----------
#     qm : xr.DataArray
#       Quantile correction factors.
#     xq : xr.DataArray
#       Quantiles for source array (historical simulation).
#     extrapolation : {"constant"}
#       Method to extrapolate outside the estimated quantiles.

#     Returns
#     -------
#     xr.DataArray
#       Quantile correction factors whose quantile coordinates have been replaced by corresponding x values.

#     Notes
#     -----
#     The original qm object has `quantile` coordinates and some grouping coordinate (e.g. month). This function
#     reindexes the array based on the values of x, instead of the quantiles. Since the x values are different from
#     group to group, the index can get fairly large.
#     """
#     dim, prop = parse_group(xq.group)
#     if prop is None:
#         q, x = extrapolate_qm(qm, xq, extrapolation)
#         out = q.rename({"quantile": "x"}).assign_coords(x=x.values)

#     else:
#         # Interpolation from quantile to values.
#         def func(d):
#             q, x = extrapolate_qm(d.qm, d.xq, extrapolation)
#             return xr.DataArray(
#                 dims="x",
#                 data=np.interp(newx, x, q, left=np.nan, right=np.nan),
#                 coords={"x": newx},
#             )

#         ds = xr.Dataset({"xq": xq, "qm": qm})
#         gr = ds.groupby(prop)

#         # X coordinates common to all groupings
#         xs = list(map(lambda g: extrapolate_qm(g[1].qm, g[1].xq, extrapolation)[1], gr))
#         newx = np.unique(np.concatenate(xs))
#         out = gr.map(func, shortcut=True)

#     out.attrs = qm.attrs
#     out.attrs["quantiles"] = qm.coords["quantile"].values
#     return out


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
    group : str
        The dimension and grouping information. (ex: "time" or "time.month").
        Defaults to the "group" attribute of xq, or "time" if there is none.
    method : str
        The interpolation method.
    """
    dim, prop = parse_group(group or xq.attrs.get("group", "time"))
    if prop is None:
        return xr.apply_ufunc(
            np.interp,
            newx,
            xq,
            yq,
            input_core_dims=[[dim], ["quantiles"], ["quantiles"]],
            output_core_dims=[[dim]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[np.float],
        )

    def _interp_quantiles_2D(newx, newg, oldx, oldy, oldg):
        return griddata(
            (oldx.flatten(), oldg.flatten()),
            oldy.flatten(),
            (newx, newg),
            method=method,
        )

    newg = newx[prop]
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


def jitter_under_thresh(x, thresh):
    """Add a small noise to values smaller than threshold."""
    epsilon = np.finfo(x.dtype).eps
    jitter = np.random.uniform(low=epsilon, high=thresh, size=x.shape)
    return x.where(~(x < thresh & x.notnull()), jitter)

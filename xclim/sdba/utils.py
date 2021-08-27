"""SDBA utilities module."""
from typing import Callable, List, Mapping, Optional, Tuple, Union
from warnings import warn

import numpy as np
import xarray as xr
from boltons.funcutils import wraps
from dask import array as dsk
from scipy.interpolate import griddata, interp1d

from xclim.core.calendar import _interpolate_doy_calendar  # noqa
from xclim.core.utils import ensure_chunk_size

from .base import Grouper, parse_group

MULTIPLICATIVE = "*"
ADDITIVE = "+"
loffsets = {"MS": "14d", "M": "15d", "YS": "181d", "Y": "182d", "QS": "45d", "Q": "46d"}


def _ecdf_1d(x, value):
    sx = np.r_[-np.inf, np.sort(x, axis=None)]
    return np.searchsorted(sx, value, side="right") / np.sum(~np.isnan(sx))


def map_cdf_1d(x, y, y_value):
    """Return the value in `x` with the same CDF as `y_value` in `y`."""
    q = _ecdf_1d(y, y_value)
    _func = np.nanquantile
    return _func(x, q=q)


def map_cdf(
    ds: xr.Dataset,
    *,
    y_value: xr.DataArray,
    dim,
):
    """Return the value in `x` with the same CDF as `y_value` in `y`.

    This function is meant to be wrapped in a `Grouper.apply`.

    Parameters
    ----------
    ds : xr.Dataset
      Variables: x, Values from which to pick,
      y, Reference values giving the ranking
    y_value : float, array
      Value within the support of `y`.
    dim : str
      Dimension along which to compute quantile.

    Returns
    -------
    array
      Quantile of `x` with the same CDF as `y_value` in `y`.
    """
    return xr.apply_ufunc(
        map_cdf_1d,
        ds.x,
        ds.y,
        input_core_dims=[dim] * 2,
        output_core_dims=[["x"]],
        vectorize=True,
        keep_attrs=True,
        kwargs={"y_value": np.atleast_1d(y_value)},
        output_dtypes=[ds.x.dtype],
    )


def ecdf(x: xr.DataArray, value: float, dim: str = "time") -> xr.DataArray:
    """Return the empirical CDF of a sample at a given value.

    Parameters
    ----------
    x : array
      Sample.
    value : float
      The value within the support of `x` for which to compute the CDF value.
    dim : str
      Dimension name.

    Returns
    -------
    xr.DataArray
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
def get_correction(x: xr.DataArray, y: xr.DataArray, kind: str) -> xr.DataArray:
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
def apply_correction(
    x: xr.DataArray, factor: xr.DataArray, kind: Optional[str] = None
) -> xr.DataArray:
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


def invert(x: xr.DataArray, kind: Optional[str] = None) -> xr.DataArray:
    """Invert a DataArray either additively (-x) or multiplicatively (1/x).

    If kind is not given, default to the one stored in the "kind" attribute of x.
    """
    kind = kind or x.get("kind", None)
    with xr.set_options(keep_attrs=True):
        if kind == ADDITIVE:
            return -x
        if kind == MULTIPLICATIVE:
            return 1 / x  # type: ignore
        raise ValueError


@parse_group
def broadcast(
    grouped: xr.DataArray,
    x: xr.DataArray,
    *,
    group: Union[str, Grouper] = "time",
    interp: str = "nearest",
    sel: Optional[Mapping[str, xr.DataArray]] = None,
) -> xr.DataArray:
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

    Returns
    -------
    xr.DataArray
    """
    if sel is None:
        sel = {}

    if group.prop != "group" and group.prop not in sel:
        sel.update({group.prop: group.get_index(x, interp=interp)})

    if sel:
        # Extract the correct mean factor for each time step.
        if interp == "nearest":  # Interpolate both the time group and the quantile.
            grouped = grouped.sel(sel, method="nearest")
        else:  # Find quantile for nearest time group and quantile.
            # For `.interp` we need to explicitly pass the shared dims
            # (see pydata/xarray#4463 and Ouranosinc/xclim#449,567)
            sel.update(
                {dim: x[dim] for dim in set(grouped.dims).intersection(set(x.dims))}
            )
            if group.prop != "group":
                grouped = add_cyclic_bounds(grouped, group.prop, cyclic_coords=False)

            if interp == "cubic" and len(sel.keys()) > 1:
                interp = "linear"
                warn(
                    "Broadcasting operations in multiple dimensions can only be done with linear and nearest-neighbor"
                    " interpolation, not cubic. Using linear."
                )

            grouped = grouped.interp(sel, method=interp).astype(grouped.dtype)

        for var in sel.keys():
            if var in grouped.coords and var not in grouped.dims:
                grouped = grouped.drop_vars(var)

    if group.prop == "group" and "group" in grouped.dims:
        grouped = grouped.squeeze("group", drop=True)
    return grouped


def equally_spaced_nodes(n: int, eps: Union[float, None] = 1e-4) -> np.array:
    """Return nodes with `n` equally spaced points within [0, 1] plus two end-points.

    Parameters
    ----------
    n : int
      Number of equally spaced nodes.
    eps : float, None
      Distance from 0 and 1 of end nodes. If None, do not add endpoints.

    Returns
    -------
    np.array
      Nodes between 0 and 1.

    Notes
    -----
    For n=4, eps=0 :  0---x------x------x------x---1
    """
    dq = 1 / n / 2
    q = np.linspace(dq, 1 - dq, n)
    if eps is None:
        return q
    return np.insert(np.append(q, 1 - eps), 0, eps)


def add_cyclic_bounds(
    da: xr.DataArray, att: str, cyclic_coords: bool = True
) -> Union[xr.DataArray, xr.Dataset]:
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


def extrapolate_qm(
    qf: xr.DataArray,
    xq: xr.DataArray,
    method: str = "constant",
    abs_bounds: Optional[tuple] = (-np.inf, np.inf),
) -> Tuple[xr.DataArray, xr.DataArray]:
    """Extrapolate quantile adjustment factors beyond the computed quantiles.

    Parameters
    ----------
    qf : xr.DataArray
      Adjustment factors over `quantile` coordinates.
    xq : xr.DataArray
      Values at each `quantile`.
    method : {"constant"}
      Extrapolation method. See notes below.
    abs_bounds : 2-tuple
      The absolute bounds for the "constant*" methods. Defaults to (-inf, inf).

    Returns
    -------
    xr.Dataset or xr.DataArray
        Extrapolated adjustment factors.
    xr.Dataset or xr.DataArray
        Extrapolated x-values.

    Notes
    -----
    qf: xr.DataArray
      Estimating values above or below the computed values will return a NaN.
    xq: xr.DataArray
      The adjustment factor above and below the computed values are equal to the last and first values
      respectively.
    """
    # constant_iqr
    #   Same as `constant`, but values are set to NaN if farther than one interquartile range from the min and max.
    if method == "nan":
        return qf, xq

    if method == "constant":
        q_l, q_r = [0], [1]
        x_l, x_r = [abs_bounds[0]], [abs_bounds[1]]
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
) -> xr.DataArray:
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
        The values at which to evaluate `yq`. If `group` has group information,
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

    if prop == "group":
        fill_value = "extrapolate" if method == "nearest" else np.nan

        def _interp_quantiles_1D(newx, oldx, oldy):
            return interp1d(
                oldx, oldy, bounds_error=False, kind=method, fill_value=fill_value
            )(newx)

        return xr.apply_ufunc(
            _interp_quantiles_1D,
            newx,
            xq.squeeze("group", drop=True),
            yq.squeeze("group", drop=True),
            input_core_dims=[[dim], ["quantiles"], ["quantiles"]],
            output_core_dims=[[dim]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[float],
        )
    # else:

    def _interp_quantiles_2D(_newx, _newg, _oldx, _oldy, _oldg):  # noqa
        if method != "nearest":
            _oldx = np.clip(_oldx, _newx.min() - 1, _newx.max() + 1)
        if np.all(np.isnan(_newx)):
            warn(
                "All-NaN slice encountered in interp_on_quantiles",
                category=RuntimeWarning,
            )
            return _newx
        return griddata(
            (_oldx.flatten(), _oldg.flatten()),
            _oldy.flatten(),
            (_newx, _newg),
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


# TODO is this useless?
def rank(da: xr.DataArray, dim: str = "time", pct: bool = False) -> xr.DataArray:
    """Ranks data along a dimension.

    Replicates `xr.DataArray.rank` but as a function usable in a Grouper.apply().
    Xarray's docstring is below:

    Equal values are assigned a rank that is the average of the ranks that
    would have been otherwise assigned to all of the values within that
    set.  Ranks begin at 1, not 0. If pct, computes percentage ranks.

    NaNs in the input array are returned as NaNs.

    The `bottleneck` library is required.

    Parameters
    ----------
    da: xr.DataArray
    dim : str, hashable
        Dimension over which to compute rank.
    pct : bool, optional
        If True, compute percentage ranks, otherwise compute integer ranks.

    Returns
    -------
    DataArray
        DataArray with the same coordinates and dtype 'float64'.
    """
    return da.rank(dim, pct=pct)


def pc_matrix(arr: Union[np.ndarray, dsk.Array]) -> Union[np.ndarray, dsk.Array]:
    """Construct a Principal Component matrix.

    This matrix can be used to transform points in arr to principal components
    coordinates. Note that this function does not manage NaNs; if a single observation is null, all elements
    of the transformation matrix involving that variable will be NaN.

    Parameters
    ----------
    arr : numpy.ndarray or dask.array.Array
      2D array (M, N) of the M coordinates of N points.

    Returns
    -------
    numpy.ndarray or dask.array.Array
      MxM Array of the same type as arr.
    """
    # Get appropriate math module
    mod = dsk if isinstance(arr, dsk.Array) else np

    # Covariance matrix
    cov = mod.cov(arr)

    # Get eigenvalues and eigenvectors
    # There are no such method yet in dask, but we are lucky:
    # the SVD decomposition of a symmetric matrix gives the eigen stuff.
    # And covariance matrices are by definition symmetric!
    # Numpy has a hermitian=True option to accelerate, but not dask...
    kwargs = {} if mod is dsk else {"hermitian": True}
    eig_vec, eig_vals, _ = mod.linalg.svd(cov, **kwargs)

    # The PC matrix is the eigen vectors matrix scaled by the square root of the eigen values
    return eig_vec * mod.sqrt(eig_vals)


def best_pc_orientation(
    A: np.ndarray, Binv: np.ndarray, val: float = 1000
) -> np.ndarray:
    """Return best orientation vector for A.

    Eigenvectors returned by `pc_matrix` do not have a defined orientation.
    Given an inverse transform Binv and a transform A, this returns the orientation
    minimizing the projected distance for a test point far from the origin.

    This trick is explained in [hnilica2017]_. See documentation of
    `sdba.adjustment.PrincipalComponentAdjustment`.

    Parameters
    ----------
    A : np.ndarray
      MxM Matrix defining the final transformation.
    Binv : np.ndarray
      MxM Matrix defining the (inverse) first transformation.
    val : float
      The coordinate of the test point (same for all axes). It should be much
      greater than the largest furthest point in the array used to define B.

    Returns
    -------
    np.ndarray
      Mx1 vector of orientation correction (1 or -1).
    """
    m = A.shape[0]
    orient = np.ones(m)
    P = np.diag(val * np.ones(m))

    # Compute first reference error
    err = np.linalg.norm(P - A @ Binv @ P)
    for i in range(m):
        # Switch the ith axis orientation
        orient[i] = -1
        # Compute new error
        new_err = np.linalg.norm(P - (A * orient) @ Binv @ P)
        if new_err > err:
            # Previous error was lower, switch back
            orient[i] = 1
        else:
            # New orientation is better, keep and remember error.
            err = new_err
    return orient


def get_clusters_1d(
    data: np.ndarray, u1: float, u2: float
) -> Tuple[np.array, np.array, np.array, np.array]:
    """Get clusters of a 1D array.

    A cluster is defined as a sequence of values larger than u2 with at least one value larger than u1.

    Parameters
    ----------
    data: 1D ndarray
      Values to get clusters from.
    u1 : float
      Extreme value threshold, at least one value in the cluster must exceed this.
    u2 : float
      Cluster threshold, values above this can be part of a cluster.

    Returns
    -------
    (np.array, np.array, np.array, np.array)

    References
    ----------
    `getcluster` of Extremes.jl (read on 2021-04-20) https://github.com/jojal5/Extremes.jl
    """
    # Boolean array, True where data is over u2
    # We pad with values under u2, so that clusters never start or end at boundaries.
    exce = np.concatenate(([u2 - 1], data, [u2 - 1])) > u2

    # 1 just before the start of the cluster
    # -1 on the last element of the cluster
    bounds = np.diff(exce.astype(np.int32))
    # We add 1 to get the first element and sub 1 to get the same index as in data
    starts = np.where(bounds == 1)[0]
    # We sub 1 to get the same index as in data and add 1 to get the element after (for python slicing)
    ends = np.where(bounds == -1)[0]

    cl_maxpos = []
    cl_maxval = []
    cl_start = []
    cl_end = []
    for start, end in zip(starts, ends):
        cluster_max = data[start:end].max()
        if cluster_max > u1:
            cl_maxval.append(cluster_max)
            cl_maxpos.append(start + np.argmax(data[start:end]))
            cl_start.append(start)
            cl_end.append(end - 1)

    return (
        np.array(cl_start),
        np.array(cl_end),
        np.array(cl_maxpos),
        np.array(cl_maxval),
    )


def get_clusters(data: xr.DataArray, u1, u2, dim: str = "time") -> xr.Dataset:
    """Get cluster count, maximum and position along a given dim.

    See `get_clusters_1d`. Used by `adjustment.ExtremeValues`.

    Parameters
    ----------
    data: 1D ndarray
      Values to get clusters from.
    u1 : float
      Extreme value threshold, at least one value in the cluster must exceed this.
    u2 : float
      Cluster threshold, values above this can be part of a cluster.
    dim : str
      Dimension name.

    Returns
    -------
    xr.Dataset
      With variables,
        - `nclusters` : Number of clusters for each point (with `dim` reduced), int
        - `start` : First index in the cluster (`dim` reduced, new `cluster`), int
        - `end` : Last index in the cluster, inclusive (`dim` reduced, new `cluster`), int
        - `maxpos` : Index of the maximal value within the cluster (`dim` reduced, new `cluster`), int
        - `maximum` : Maximal value within the cluster (`dim` reduced, new `cluster`), same dtype as data.

      For `start`, `end` and `maxpos`, -1 means NaN and should always correspond to a `NaN` in `maximum`.
      The length along `cluster` is half the size of "dim", the maximal theoritical number of clusters.
    """

    def _get_clusters(arr, u1, u2, N):
        st, ed, mp, mv = get_clusters_1d(arr, u1, u2)
        count = len(st)
        pad = [-1] * (N - count)
        return (
            np.append(st, pad),
            np.append(ed, pad),
            np.append(mp, pad),
            np.append(mv, [np.NaN] * (N - count)),
            count,
        )

    # The largest possible number of clusters. Ex: odd positions are < u2, even positions are > u1.
    N = data[dim].size // 2

    starts, ends, maxpos, maxval, nclusters = xr.apply_ufunc(
        _get_clusters,
        data,
        u1,
        u2,
        input_core_dims=[[dim], [], []],
        output_core_dims=[["cluster"], ["cluster"], ["cluster"], ["cluster"], []],
        kwargs={"N": N},
        dask="parallelized",
        vectorize=True,
        dask_gufunc_kwargs={
            "meta": (
                np.array((), dtype=int),
                np.array((), dtype=int),
                np.array((), dtype=int),
                np.array((), dtype=data.dtype),
                np.array((), dtype=int),
            ),
            "output_sizes": {"cluster": N},
        },
    )

    ds = xr.Dataset(
        {
            "start": starts,
            "end": ends,
            "maxpos": maxpos,
            "maximum": maxval,
            "nclusters": nclusters,
        }
    )

    return ds


def rand_rot_matrix(
    crd: xr.DataArray, num: int = 1, new_dim: Optional[str] = None
) -> xr.DataArray:
    r"""Generate random rotation matrices.

    Rotation matrices are members of the SO(n) group, where n is the matrix size (`crd.size`).
    They can be characterized as orthogonal matrices with determinant 1. A square matrix :math:`R`
    is a rotation matrix if and only if :math:`R^t = R^{−1}` and :math:`\mathrm{det} R = 1`.

    Parameters
    ----------
    crd: xr.DataArray
      1D coordinate DataArray along which the rotation occurs.
      The output will be square with the same coordinate replicated,
      the second renamed to `new_dim`.
    num : int
      If larger than 1 (default), the number of matrices to generate, stacked along a "matrices" dimension.
    new_dim : str
      Name of the new "prime" dimension, defaults to the same name as `crd` + "_prime".

    Returns
    -------
    xr.DataArray
      float, NxN if num = 1, numxNxN otherwise, where N is the length of crd.

    References
    ----------
    .. [Mezzadri] Mezzadri, F. (2006). How to generate random matrices from the classical compact groups. arXiv preprint math-ph/0609050.
    """
    if num > 1:
        return xr.concat([rand_rot_matrix(crd, num=1) for i in range(num)], "matrices")

    N = crd.size
    dim = crd.dims[0]
    # Rename and rebuild second coordinate : "prime" axis.
    if new_dim is None:
        new_dim = dim + "_prime"
    crd2 = xr.DataArray(crd.values, dims=new_dim, name=new_dim, attrs=crd.attrs)

    # Random floats from the standardized normal distribution
    Z = np.random.standard_normal((N, N))

    # QR decomposition and manipulation from Mezzadri 2006
    Q, R = np.linalg.qr(Z)
    num = np.diag(R)
    denum = np.abs(num)
    lam = np.diag(num / denum)  # "lambda"
    return xr.DataArray(
        Q @ lam, dims=(dim, new_dim), coords={dim: crd, new_dim: crd2}
    ).astype("float32")

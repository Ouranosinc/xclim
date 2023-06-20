"""
Statistical Downscaling and Bias Adjustment Utilities
=====================================================
"""
from __future__ import annotations

import itertools
from typing import Callable
from warnings import warn

import numpy as np
import xarray as xr
from boltons.funcutils import wraps
from dask import array as dsk
from scipy.interpolate import griddata, interp1d
from scipy.stats import spearmanr

from xclim.core.calendar import _interpolate_doy_calendar  # noqa
from xclim.core.utils import ensure_chunk_size

from .base import Grouper, parse_group
from .nbutils import _extrapolate_on_quantiles

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
                x = _interpolate_doy_calendar(
                    x, int(y.dayofyear.max()), int(y.dayofyear.min())
                )
            else:
                y = _interpolate_doy_calendar(
                    y, int(x.dayofyear.max()), int(x.dayofyear.min())
                )
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
    x: xr.DataArray, factor: xr.DataArray, kind: str | None = None
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


def invert(x: xr.DataArray, kind: str | None = None) -> xr.DataArray:
    """Invert a DataArray either by addition (-x) or by multiplication (1/x).

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
    group: str | Grouper = "time",
    interp: str = "nearest",
    sel: dict[str, xr.DataArray] | None = None,
) -> xr.DataArray:
    """Broadcast a grouped array back to the same shape as a given array.

    Parameters
    ----------
    grouped : xr.DataArray
      The grouped array to broadcast like `x`.
    x : xr.DataArray
      The array to broadcast grouped to.
    group : str or Grouper
      Grouping information. See :py:class:`xclim.sdba.base.Grouper` for details.
    interp : {'nearest', 'linear', 'cubic'}
      The interpolation method to use,
    sel : dict[str, xr.DataArray]
      Mapping of grouped coordinates to x coordinates (other than the grouping one).

    Returns
    -------
    xr.DataArray
    """
    if sel is None:
        sel = {}

    if group.prop != "group" and group.prop not in sel:
        sel.update({group.prop: group.get_index(x, interp=interp != "nearest")})

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


def equally_spaced_nodes(n: int, eps: float | None = None) -> np.array:
    """Return nodes with `n` equally spaced points within [0, 1], optionally adding two end-points.

    Parameters
    ----------
    n : int
      Number of equally spaced nodes.
    eps : float, optional
      Distance from 0 and 1 of added end nodes. If None (default), do not add endpoints.

    Returns
    -------
    np.array
      Nodes between 0 and 1. Nodes can be seen as the middle points of `n` equal bins.

    Warnings
    --------
    Passing a small `eps` will effectively clip the scenario to the bounds of the reference
    on the historical period in most cases. With normal quantile mapping algorithms, this can
    give strange result when the reference does not show as many extremes as the simulation does.

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
) -> xr.DataArray | xr.Dataset:
    """Reindex an array to include the last slice at the beginning and the first at the end.

    This is done to allow interpolation near the end-points.

    Parameters
    ----------
    da : xr.DataArray or xr.Dataset
        An array
    att : str
        The name of the coordinate to make cyclic
    cyclic_coords : bool
        If True, the coordinates are made cyclic as well,
        if False, the new values are guessed using the same step as their neighbour.

    Returns
    -------
    xr.DataArray or xr.Dataset
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


def _interp_on_quantiles_1D(newx, oldx, oldy, method, extrap):
    mask_new = np.isnan(newx)
    mask_old = np.isnan(oldy) | np.isnan(oldx)
    out = np.full_like(newx, np.NaN, dtype=f"float{oldy.dtype.itemsize * 8}")
    if np.all(mask_new) or np.all(mask_old):
        warn(
            "All-NaN slice encountered in interp_on_quantiles",
            category=RuntimeWarning,
        )
        return out

    if extrap == "constant":
        fill_value = (
            oldy[~np.isnan(oldy)][0],
            oldy[~np.isnan(oldy)][-1],
        )
    else:  # extrap == 'nan'
        fill_value = np.NaN

    out[~mask_new] = interp1d(
        oldx[~mask_old],
        oldy[~mask_old],
        kind=method,
        bounds_error=False,
        fill_value=fill_value,
    )(newx[~mask_new])
    return out


def _interp_on_quantiles_2D(newx, newg, oldx, oldy, oldg, method, extrap):  # noqa
    mask_new = np.isnan(newx) | np.isnan(newg)
    mask_old = np.isnan(oldy) | np.isnan(oldx) | np.isnan(oldg)
    out = np.full_like(newx, np.NaN, dtype=f"float{oldy.dtype.itemsize * 8}")
    if np.all(mask_new) or np.all(mask_old):
        warn(
            "All-NaN slice encountered in interp_on_quantiles",
            category=RuntimeWarning,
        )
        return out
    out[~mask_new] = griddata(
        (oldx[~mask_old], oldg[~mask_old]),
        oldy[~mask_old],
        (newx[~mask_new], newg[~mask_new]),
        method=method,
    )
    if method == "nearest" or extrap != "nan":
        # 'nan' extrapolation implicit for cubic and linear interpolation.
        out = _extrapolate_on_quantiles(out, oldx, oldg, oldy, newx, newg, extrap)
    return out


@parse_group
def interp_on_quantiles(
    newx: xr.DataArray,
    xq: xr.DataArray,
    yq: xr.DataArray,
    *,
    group: str | Grouper = "time",
    method: str = "linear",
    extrapolation: str = "constant",
):
    """Interpolate values of yq on new values of x.

    Interpolate in 2D with :py:func:`scipy.interpolate.griddata` if grouping is used, in 1D otherwise, with
    :py:class:`scipy.interpolate.interp1d`.
    Any NaNs in `xq` or `yq` are removed from the input map.
    Similarly, NaNs in newx are left NaNs.

    Parameters
    ----------
    newx : xr.DataArray
      The values at which to evaluate `yq`. If `group` has group information,
      `new` should have a coordinate with the same name as the group name
      In that case, 2D interpolation is used.
    xq, yq : xr.DataArray
      Coordinates and values on which to interpolate. The interpolation is done
      along the "quantiles" dimension if `group` has no group information.
      If it does, interpolation is done in 2D on "quantiles" and on the group dimension.
    group : str or Grouper
      The dimension and grouping information. (ex: "time" or "time.month").
      Defaults to "time".
    method : {'nearest', 'linear', 'cubic'}
      The interpolation method.
    extrapolation : {'constant', 'nan'}
      The extrapolation method used for values of `newx` outside the range of `xq`.
      See notes.

    Notes
    -----
    Extrapolation methods:

    - 'nan' : Any value of `newx` outside the range of `xq` is set to NaN.
    - 'constant' : Values of `newx` smaller than the minimum of `xq` are set to the first
      value of `yq` and those larger than the maximum, set to the last one (first and
      last non-nan values along the "quantiles" dimension). When the grouping is "time.month",
      these limits are linearly interpolated along the month dimension.
    """
    dim = group.dim
    prop = group.prop

    if prop == "group":
        if "group" in xq.dims:
            xq = xq.squeeze("group", drop=True)
        if "group" in yq.dims:
            yq = yq.squeeze("group", drop=True)

        out = xr.apply_ufunc(
            _interp_on_quantiles_1D,
            newx,
            xq,
            yq,
            kwargs={"method": method, "extrap": extrapolation},
            input_core_dims=[[dim], ["quantiles"], ["quantiles"]],
            output_core_dims=[[dim]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[yq.dtype],
        )
        return out

    # else:
    if prop not in xq.dims:
        xq = xq.expand_dims({prop: group.get_coordinate()})
    if prop not in yq.dims:
        yq = yq.expand_dims({prop: group.get_coordinate()})

    xq = add_cyclic_bounds(xq, prop, cyclic_coords=False)
    yq = add_cyclic_bounds(yq, prop, cyclic_coords=False)
    newg = group.get_index(newx, interp=method != "nearest")
    oldg = xq[prop].expand_dims(quantiles=xq.coords["quantiles"])

    return xr.apply_ufunc(
        _interp_on_quantiles_2D,
        newx,
        newg,
        xq,
        yq,
        oldg,
        kwargs={"method": method, "extrap": extrapolation},
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


def rank(da: xr.DataArray, dim: str = "time", pct: bool = False) -> xr.DataArray:
    """Ranks data along a dimension.

    Replicates `xr.DataArray.rank` but as a function usable in a Grouper.apply(). Xarray's docstring is below:

    Equal values are assigned a rank that is the average of the ranks that would have been otherwise assigned to all the
    values within that set. Ranks begin at 1, not 0. If pct, computes percentage ranks, ranging from 0 to 1.

    Parameters
    ----------
    da: xr.DataArray
      Source array.
    dim : str, hashable
      Dimension over which to compute rank.
    pct : bool, optional
      If True, compute percentage ranks, otherwise compute integer ranks.
      Percentage ranks range from 0 to 1, in opposition to xarray's implementation,
      where they range from 1/N to 1.

    Returns
    -------
    DataArray
        DataArray with the same coordinates and dtype 'float64'.

    Notes
    -----
    The `bottleneck` library is required. NaNs in the input array are returned as NaNs.

    See Also
    --------
    xarray.DataArray.rank
    """
    rnk = da.rank(dim, pct=pct)
    if pct:
        mn = rnk.min(dim)
        mx = rnk.max(dim)
        return mx * (rnk - mn) / (mx - mn)
    return rnk


def pc_matrix(arr: np.ndarray | dsk.Array) -> np.ndarray | dsk.Array:
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


def best_pc_orientation_simple(
    R: np.ndarray, Hinv: np.ndarray, val: float = 1000
) -> np.ndarray:
    """Return best orientation vector according to a simple test.

    Eigenvectors returned by `pc_matrix` do not have a defined orientation.
    Given an inverse transform `Hinv` and a transform `R`, this returns the orientation minimizing the projected
    distance for a test point far from the origin.

    This trick is inspired by the one exposed in :cite:t:`sdba-hnilica_multisite_2017`. For each possible orientation vector,
    the test point is reprojected and the distance from the original point is computed. The orientation
    minimizing that distance is chosen.

    Parameters
    ----------
    R : np.ndarray
      MxM Matrix defining the final transformation.
    Hinv : np.ndarray
      MxM Matrix defining the (inverse) first transformation.
    val : float
      The coordinate of the test point (same for all axes). It should be much
      greater than the largest furthest point in the array used to define B.

    Returns
    -------
    np.ndarray
      Mx1 vector of orientation correction (1 or -1).

    See Also
    --------
    sdba.adjustment.PrincipalComponentAdjustment

    References
    ----------
    :cite:cts:`sdba-hnilica_multisite_2017`
    """
    m = R.shape[0]
    P = np.diag(val * np.ones(m))
    signes = dict(itertools.zip_longest(itertools.product(*[[1, -1]] * m), [None]))
    for orient in list(signes.keys()):
        # Compute new error
        signes[orient] = np.linalg.norm(P - ((orient * R) @ Hinv) @ P)
    return np.array(min(signes, key=lambda o: signes[o]))


def best_pc_orientation_full(
    R: np.ndarray,
    Hinv: np.ndarray,
    Rmean: np.ndarray,
    Hmean: np.ndarray,
    hist: np.ndarray,
) -> np.ndarray:
    """Return best orientation vector for `A` according to the method of :cite:t:`sdba-alavoine_distinct_2022`.

    Eigenvectors returned by `pc_matrix` do not have a defined orientation.
    Given an inverse transform Hinv, a transform R, the actual and target origins `Hmean` and `Rmean` and the matrix of
    training observations hist, this computes a scenario for all possible orientations and return the orientation that
    maximizes the Spearman correlation coefficient of all variables. The correlation is computed for each variable
    individually, then averaged.

    This trick is explained in :cite:t:`sdba-alavoine_distinct_2022`.
    See docstring of :py:func:`sdba.adjustment.PrincipalComponentAdjustment`.

    Parameters
    ----------
    R : np.ndarray
      MxM Matrix defining the final transformation.
    Hinv : np.ndarray
      MxM Matrix defining the (inverse) first transformation.
    Rmean : np.ndarray
      M vector defining the target distribution center point.
    Hmean : np.ndarray
      M vector defining the original distribution center point.
    hist : np.ndarray
      MxN matrix of all training observations of the M variables/sites.

    Returns
    -------
    np.ndarray
      M vector of orientation correction (1 or -1).

    References
    ----------
    :cite:cts:`sdba-alavoine_distinct_2022`

    """
    # All possible orientation vectors
    m = R.shape[0]
    signes = dict(itertools.zip_longest(itertools.product(*[[1, -1]] * m), [None]))
    for orient in list(signes.keys()):
        # Calculate scen for hist
        scen = np.atleast_2d(Rmean).T + ((orient * R) @ Hinv) @ (
            hist - np.atleast_2d(Hmean).T
        )
        # Correlation for each variable
        corr = [spearmanr(hist[i, :], scen[i, :])[0] for i in range(hist.shape[0])]
        # Store mean correlation
        signes[orient] = np.mean(corr)
    # Return orientation that maximizes the correlation
    return np.array(max(signes, key=lambda o: signes[o]))


def get_clusters_1d(
    data: np.ndarray, u1: float, u2: float
) -> tuple[np.array, np.array, np.array, np.array]:
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
    `getcluster` of Extremes.jl (:cite:cts:`sdba-jalbert_extreme_2022`).
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
      The length along `cluster` is half the size of "dim", the maximal theoretical number of clusters.
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
    crd: xr.DataArray, num: int = 1, new_dim: str | None = None
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
    :cite:cts:`sdba-mezzadri_how_2007`

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


def copy_all_attrs(ds: xr.Dataset | xr.DataArray, ref: xr.Dataset | xr.DataArray):
    """Copy all attributes of ds to ref, including attributes of shared coordinates, and variables in the case of Datasets."""
    ds.attrs.update(ref.attrs)
    extras = ds.variables if isinstance(ds, xr.Dataset) else ds.coords
    others = ref.variables if isinstance(ref, xr.Dataset) else ref.coords
    for name, var in extras.items():
        if name in others:
            var.attrs.update(ref[name].attrs)


def _pairwise_spearman(da, dims):
    """Area-averaged pairwise temporal correlation.

    With skipna-shortcuts for cases where all times or all points are NaN.
    """
    da = da - da.mean(dims)
    da = (
        da.stack(_spatial=dims)
        .reset_index("_spatial")
        .drop_vars(["_spatial"], errors=["ignore"])
    )

    def _skipna_correlation(data):
        nv, nt = data.shape
        # Mask of which variable are all NaN
        mask_omit = np.isnan(data).all(axis=1)
        # Remove useless variables
        data_noallnan = data[~mask_omit, :]
        # Mask of which times are nan on all variables
        mask_skip = np.isnan(data_noallnan).all(axis=0)
        # Remove those times (they'll be omitted anyway)
        data_nonan = data_noallnan[:, ~mask_skip]

        # We still have a possibility that a NaN was unique to a variable and time.
        # If this is the case, it will be a lot longer, but what can we do.
        coef = spearmanr(data_nonan, axis=1, nan_policy="omit").correlation

        # The output
        out = np.empty((nv, nv), dtype=coef.dtype)
        # A 2D mask of removed variables
        M = (mask_omit)[:, np.newaxis] | (mask_omit)[np.newaxis, :]
        out[~M] = coef.flatten()
        out[M] = np.nan
        return out

    return xr.apply_ufunc(
        _skipna_correlation,
        da,
        input_core_dims=[["_spatial", "time"]],
        output_core_dims=[["_spatial", "_spatial2"]],
        vectorize=True,
        output_dtypes=[float],
        dask="parallelized",
        dask_gufunc_kwargs={
            "output_sizes": {
                "_spatial": da._spatial.size,
                "_spatial2": da._spatial.size,
            },
            "allow_rechunk": True,
        },
    ).rename("correlation")

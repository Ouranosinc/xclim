"""
LOESS Smoothing Submodule
=========================
"""
from __future__ import annotations

from typing import Callable
from warnings import warn

import numba
import numpy as np
import xarray as xr


@numba.njit
def _gaussian_weighting(x):  # pragma: no cover
    """
    Kernel function for loess with a gaussian shape.

    The span f covers 95% of the gaussian.
    """
    w = np.exp(-(x**2) / (2 * (1 / 1.96) ** 2))
    w[x >= 1] = 0
    return w


@numba.njit
def _tricube_weighting(x):  # pragma: no cover
    """Kernel function for loess with a tricubic shape."""
    w = (1 - x**3) ** 3
    w[x >= 1] = 0
    return w


@numba.njit
def _constant_regression(xi, x, y, w):  # pragma: no cover
    return (w * y).sum() / w.sum()


@numba.njit
def _linear_regression(xi, x, y, w):  # pragma: no cover
    b = np.array([np.sum(w * y), np.sum(w * y * x)])
    A = np.array([[np.sum(w), np.sum(w * x)], [np.sum(w * x), np.sum(w * x * x)]])
    beta = np.linalg.solve(A, b)
    return beta[0] + beta[1] * xi


@numba.njit
def _loess_nb(
    x,
    y,
    f=0.5,
    niter=2,
    weight_func=_tricube_weighting,
    reg_func=_linear_regression,
    dx=0,
    skipna=True,
):  # pragma: no cover
    """1D Locally weighted regression: fits a nonparametric regression curve to a scatter plot.

    The arrays x and y contain an equal number of elements; each pair (x[i], y[i]) defines
    a data point in the scatter plot. The function returns the estimated (smooth) values of y.
    Originally proposed in :cite:t:`sdba-cleveland_robust_1979`.

    Users should call `utils.loess_smoothing`. See that function for the main documentation.

    Parameters
    ----------
    x : np.ndarray
      X-coordinates of the points.
    y : np.ndarray
      Y-coordinates of the points.
    f : float
      Parameter controlling the shape of the weight curve. Behavior depends on the weighting function.
    niter : int
      Number of robustness iterations to execute.
    weight_func : numba func
      Numba function giving the weights when passed abs(x - xi) / hi
    dx : float
      The spacing of the x coordinates. If above 0, this enables the optimization for equally spaced x coordinates.
      Must be 0 if spacing is unequal (default).
    skipna : bool
      If True (default), remove NaN values before computing the loess. The output has the
      same missing values as the input.

    References
    ----------
    :cite:cts:`sdba-cleveland_robust_1979`

    Code adapted from: :cite:cts:`sdba-gramfort_lowess_2015`
    """
    if skipna:
        nan = np.isnan(y)
        out = np.full(x.size, np.NaN)
        y = y[~nan]
        x = x[~nan]

    n = x.size
    yest = np.zeros(n)
    delta = np.ones(n)

    # Number of points included in the weights calculation
    if dx == 0:
        # No opt. directly the nearest int
        r = int(np.round(f * n))
        # With unequal spacing, the rth closest point could be up to r points on either size.
        HW = min(r + 2, n)
        R = min(2 * HW, n)
    else:
        # Equal spacing, the nearest odd number equal or above f * n
        r = int(2 * (f * n // 2) + 1)
        # half width of the weights
        hw = int((r - 1) / 2)
        # Number of values sent to the weight func. Just a bit larger than the window.
        R = min(r + 4, n)
        HW = hw + 2

    for iteration in range(niter):
        for i in range(n):
            # We can pass only a subset of the arrays as we already know where the rth closest point will be.
            if i < HW:
                xi = x[:R]
                yi = y[:R]
                di = delta[:R]
            elif i >= n - HW - 1:
                di = delta[n - R :]
                xi = x[n - R :]
                yi = y[n - R :]
            else:
                di = delta[i - HW : i + HW + 1]
                xi = x[i - HW : i + HW + 1]
                yi = y[i - HW : i + HW + 1]

            if dx > 0:
                # When x is equally spaced, we don't need to recompute the weights each time.
                # We can also skip the sorting part.
                # However, contrary to a moving mean, the weights change shape near the edges
                if i <= HW or i >= n - HW:
                    # Near the edges and on the first iteration away from them,
                    # compute the weights.
                    diffs = np.abs(xi - x[i])

                    if i < hw:
                        h = (r - i) * dx
                    elif i >= n - hw:
                        h = (i - (n - r) + 1) * dx
                    else:
                        h = (hw + 1) * dx
                    wi = weight_func(diffs / h)
                w = di * wi
            else:
                # The weights computation is repeated niter times
                # The distance of points from the current centre point.
                diffs = np.abs(xi - x[i])
                # h is the distance of the rth closest point.
                h = np.sort(diffs)[r]
                # The weights will be 0 everywhere diffs > h.
                w = di * weight_func(diffs / h)
            yest[i] = reg_func(x[i], xi, yi, w)

        if iteration < niter - 1:
            residuals = y - yest
            s = np.median(np.abs(residuals))
            xres = residuals / (6.0 * s)
            delta = (1 - xres**2) ** 2
            delta[np.abs(xres) >= 1] = 0

    if skipna:
        out[~nan] = yest
        return out
    return yest


def loess_smoothing(
    da: xr.DataArray,
    dim: str = "time",
    d: int = 1,
    f: float = 0.5,
    niter: int = 2,
    weights: str | Callable = "tricube",
    equal_spacing: bool | None = None,
    skipna: bool = True,
):
    r"""Locally weighted regression in 1D: fits a nonparametric regression curve to a scatter plot.

    Returns a smoothed curve along given dimension. The regression is computed for each point using a subset of
    neighbouring points as given from evaluating the weighting function locally.
    Follows the procedure of :cite:t:`sdba-cleveland_robust_1979`.

    Parameters
    ----------
    da: xr.DataArray
      The data to smooth using the loess approach.
    dim : str
      Name of the dimension along which to perform the loess.
    d : [0, 1]
      Degree of the local regression.
    f : float
      Parameter controlling the shape of the weight curve. Behavior depends on the weighting function,
      but it usually represents the span of the weighting function in reference to x-coordinates
      normalized from 0 to 1.
    niter : int
      Number of robustness iterations to execute.
    weights : ["tricube", "gaussian"] or callable
      Shape of the weighting function, see notes. The user can provide a function or a string:
      "tricube" : a smooth top-hat like curve.
      "gaussian" : a gaussian curve, f gives the span for 95% of the values.
    equal_spacing : bool, optional
      Whether to use the equal spacing optimization. If `None` (the default), it is activated only if the
      x-axis is equally-spaced. When activated, `dx = x[1] - x[0]`.
    skipna : bool
        If True (default), skip missing values (as marked by NaN). The output will have the
        same missing values as the input.

    Notes
    -----
    As stated in :cite:t:`sdba-cleveland_robust_1979`, the weighting function :math:`W(x)` should respect the following
    conditions:

    - :math:`W(x) > 0` for :math:`|x| < 1`
    - :math:`W(-x) = W(x)`
    - :math:`W(x)` is non-increasing for :math:`x \ge 0`
    - :math:`W(x) = 0` for :math:`|x| \ge 0`

    If a Callable is provided, it should only accept the 1D `np.ndarray` :math:`x` which is an absolute value
    function going from 1 to 0 to 1 around :math:`x_i`, for all values where :math:`x - x_i < h_i` with
    :math:`h_i` the distance of the rth nearest neighbor of  :math:`x_i`, :math:`r = f * size(x)`.

    References
    ----------
    :cite:cts:`sdba-cleveland_robust_1979`

    Code adapted from: :cite:cts:`sdba-gramfort_lowess_2015`
    """
    x = da[dim]
    x = ((x - x[0]) / (x[-1] - x[0])).astype(float)

    weight_func = {"tricube": _tricube_weighting, "gaussian": _gaussian_weighting}.get(
        weights, weights
    )

    reg_func = {0: _constant_regression, 1: _linear_regression}[d]

    diffx = np.diff(da[dim])
    if np.all(diffx == diffx[0]):
        if equal_spacing is None:
            equal_spacing = True
    elif equal_spacing:
        warn(
            "The equal spacing optimization was requested, but the x axis is not equally spaced. Strange results might occur."
        )
    if equal_spacing:
        dx = float(x[1] - x[0])
    else:
        dx = 0

    return xr.apply_ufunc(
        _loess_nb,
        x,
        da,
        input_core_dims=[[dim], [dim]],
        output_core_dims=[[dim]],
        vectorize=True,
        kwargs={
            "f": f,
            "weight_func": weight_func,
            "niter": niter,
            "reg_func": reg_func,
            "dx": dx,
            "skipna": skipna,
        },
        dask="parallelized",
        output_dtypes=[float],
    )

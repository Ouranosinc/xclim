"""LOESS smoothing module."""
from typing import Callable, Union

import numba
import numpy as np
import xarray as xr


@numba.njit
def _gaussian_weighting(x):  # pragma: no cover
    """
    Kernel function for loess with a gaussian shape.

    The span f covers 95% of the gaussian.
    """
    w = np.exp(-(x ** 2) / (2 * (1 / 1.96) ** 2))
    w[x >= 1] = 0
    return w


@numba.njit
def _tricube_weighting(x):  # pragma: no cover
    """Kernel function for loess with a tricubic shape."""
    w = (1 - x ** 3) ** 3
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
    x, y, f=0.5, niter=2, weight_func=_tricube_weighting, reg_func=_linear_regression
):  # pragma: no cover
    """1D Locally weighted regression: fits a nonparametric regression curve to a scatterplot.

    The arrays x and y contain an equal number of elements; each pair (x[i], y[i]) defines
    a data point in the scatterplot. The function returns the estimated (smooth) values of y.

    Users should call `utils.loess_smoothing`. See that function for the main documentation.

    Parameters
    ----------
    x : np.ndarray
      X-coordinates of the points.
    y : np.ndarray
      Y-coordinates of the points.
    f : float
      Parameter controling the shape of the weight curve. Behavior depends on the weighting function.
    niter : int
      Number of robustness iterations to execute.
    weight_func : numba func
      Numba function giving the weights when passed abs(x - xi) / hi

    References
    ----------
    Code adapted from https://gist.github.com/agramfort/850437
    Cleveland, W. S., 1979. Robust Locally Weighted Regression and Smoothing Scatterplot, Journal of the American Statistical Association 74, 829–836.
    """
    n = x.size
    r = int(np.round(f * n))
    yest = np.zeros(n)
    delta = np.ones(n)

    for iteration in range(niter):
        for i in range(n):
            # The weights computation is repeater niter times
            # The loss in speed is a clear gain in memory
            diffs = np.abs(x - x[i])
            h = np.sort(diffs)[r]
            w = delta * weight_func(diffs / h)
            yest[i] = reg_func(x[i], x, y, w)

        if iteration < niter - 1:
            residuals = y - yest
            s = np.median(np.abs(residuals))
            delta = residuals / (6.0 * s)
            delta = (1 - delta ** 2) ** 2
            delta[np.abs(delta) >= 1] = 0

    return yest


def loess_smoothing(
    da: xr.DataArray,
    dim: str = "time",
    d: int = 1,
    f: float = 0.5,
    niter: int = 2,
    weights: Union[str, Callable] = "tricube",
):
    r"""Locally weighted regression in 1D: fits a nonparametric regression curve to a scatterplot.

    Returns a smoothed curve along given dimension. The regression is computed for each point using
    a subset of neigboring points as given from evaluating the weighting function locally.
    Follows the procedure of [Cleveland1979]_.

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

    Notes
    -----
    As stated in [Cleveland1979]_, the weighting function :math:`W(x)` should respect the following conditions:

        - :math:`W(x) > 0` for :math:`|x| < 1`
        - :math:`W(-x) = W(x)`
        - :math:`W(x)` is nonincreasing for :math:`x \ge 0`
        - :math:`W(x) = 0` for :math:`|x| \ge 0`

    If a callable is provided, it should only accept the 1D `np.ndarray` :math:`x` which is an absolute value
    function going from 1 to 0 to 1 around :math:`x_i`, for all values where :math:`x - x_i < h_i` with
    :math:`h_i` the distance of the rth nearest neighbor of  :math:`x_i`, :math:`r = f * size(x)`.

    References
    ----------
    Code adapted from https://gist.github.com/agramfort/850437
    [Cleveland1979] Cleveland, W. S., 1979. Robust Locally Weighted Regression and Smoothing Scatterplot, Journal of the American Statistical Association 74, 829–836.
    """
    x = da[dim]
    x = (x - x[0]) / (x[-1] - x[0])

    weight_func = {"tricube": _tricube_weighting, "gaussian": _gaussian_weighting}.get(
        weights, weights
    )

    reg_func = {0: _constant_regression, 1: _linear_regression}[d]
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
        },
        dask="parallelized",
        output_dtypes=[float],
    )

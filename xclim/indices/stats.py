"""Statistic-related functions. See the `frequency_analysis` notebook for examples."""

from typing import Dict, Optional, Sequence, Tuple, Union

import dask.array
import numpy as np
import xarray as xr

from xclim.core.formatting import (
    merge_attributes,
    prefix_attrs,
    unprefix_attrs,
    update_history,
)

from . import generic

__all__ = [
    "fit",
    "parametric_quantile",
    "fa",
    "frequency_analysis",
    "get_dist",
    "get_lm3_dist",
    "_fit_start",
    "_lm3_dist_map",
]


# Map the scipy distribution name to the lmoments3 name. Distributions with mismatched parameters are excluded.
_lm3_dist_map = {
    "expon": "exp",
    "gamma": "gam",
    "genextreme": "gev",
    # "genlogistic": "glo",
    # "gennorm": "gno",
    "genpareto": "gpa",
    "gumbel_r": "gum",
    # "kappa4": "kap",
    "norm": "nor",
    "pearson3": "pe3",
    "weibull_min": "wei",
}


# Fit the parameters.
# This would also be the place to impose constraints on the series minimum length if needed.
def _fitfunc_1d(arr, *, dist, nparams, method, **fitkwargs):
    """Fit distribution parameters."""
    x = np.ma.masked_invalid(arr).compressed()

    # Return NaNs if array is empty.
    if len(x) <= 1:
        return [np.nan] * nparams

    # Estimate parameters
    if method == "ML":
        args, kwargs = _fit_start(x, dist.name, **fitkwargs)
        params = dist.fit(x, *args, **kwargs, **fitkwargs)
    elif method == "PWM":
        params = list(dist.lmom_fit(x).values())

    # Fill with NaNs if one of the parameters is NaN
    if np.isnan(params).any():
        params[:] = np.nan

    return params


def fit(
    da: xr.DataArray,
    dist: str = "norm",
    method: str = "ML",
    dim: str = "time",
    **fitkwargs,
) -> xr.DataArray:
    """Fit an array to a univariate distribution along the time dimension.

    Parameters
    ----------
    da : xr.DataArray
      Time series to be fitted along the time dimension.
    dist : str
      Name of the univariate distribution, such as beta, expon, genextreme, gamma, gumbel_r, lognorm, norm
      (see scipy.stats for full list). If the PWM method is used, only the following distributions are
      currently supported: 'expon', 'gamma', 'genextreme', 'genpareto', 'gumbel_r', 'pearson3', 'weibull_min'.
    method : {"ML", "PWM"}
      Fitting method, either maximum likelihood (ML) or probability weighted moments (PWM), also called L-Moments.
      The PWM method is usually more robust to outliers.
    dim : str
      The dimension upon which to perform the indexing (default: "time").
    **fitkwargs
      Other arguments passed directly to :py:func:`_fitstart` and to the distribution's `fit`.

    Returns
    -------
    xr.DataArray
      An array of fitted distribution parameters.

    Notes
    -----
    Coordinates for which all values are NaNs will be dropped before fitting the distribution. If the array
    still contains NaNs, the distribution parameters will be returned as NaNs.
    """
    method_name = {"ML": "maximum likelihood", "PWM": "probability weighted moments"}

    # Get the distribution
    dc = get_dist(dist)
    if method == "PWM":
        lm3dc = get_lm3_dist(dist)

    shape_params = [] if dc.shapes is None else dc.shapes.split(",")
    dist_params = shape_params + ["loc", "scale"]

    # xarray.apply_ufunc does not yet support multiple outputs with dask parallelism.
    duck = dask.array if isinstance(da.data, dask.array.Array) else np
    data = duck.apply_along_axis(
        _fitfunc_1d,
        da.get_axis_num(dim),
        da,
        dist=dc if method == "ML" else lm3dc,
        nparams=len(dist_params),
        method=method,
        **fitkwargs,
    )

    # Coordinates for the distribution parameters
    coords = dict(da.coords.items())
    if dim in coords:
        coords.pop(dim)
    coords["dparams"] = dist_params

    # Dimensions for the distribution parameters
    dims = [d if d != dim else "dparams" for d in da.dims]

    out = xr.DataArray(data=data, coords=coords, dims=dims)
    out.attrs = prefix_attrs(
        da.attrs, ["standard_name", "long_name", "units", "description"], "original_"
    )
    attrs = dict(
        long_name=f"{dist} parameters",
        description=f"Parameters of the {dist} distribution",
        method=method,
        estimator=method_name[method].capitalize(),
        scipy_dist=dist,
        units="",
        history=update_history(
            f"Estimate distribution parameters by {method_name[method]} method along dimension {dim}.",
            new_name="fit",
            data=da,
        ),
    )
    out.attrs.update(attrs)
    return out


def parametric_quantile(p: xr.DataArray, q: Union[int, Sequence]) -> xr.DataArray:
    """Return the value corresponding to the given distribution parameters and quantile.

    Parameters
    ----------
    p : xr.DataArray
      Distribution parameters returned by the `fit` function.
      The array should have dimension `dparams` storing the distribution parameters,
      and attribute `scipy_dist`, storing the name of the distribution.
    q : Union[float, Sequence]
      Quantile to compute, which must be between `0` and `1`, inclusive.

    Returns
    -------
    xarray.DataArray
      An array of parametric quantiles estimated from the distribution parameters.

    Notes
    -----
    When all quantiles are above 0.5, the `isf` method is used instead of `ppf` because accuracy is sometimes better.
    """
    q = np.atleast_1d(q)

    # Get the distribution
    dist = p.attrs["scipy_dist"]
    dc = get_dist(dist)

    # Create a lambda function to facilitate passing arguments to dask. There is probably a better way to do this.
    if np.all(q > 0.5):

        def func(x):
            return dc.isf(1 - q, *x)

    else:

        def func(x):
            return dc.ppf(q, *x)

    duck = dask.array if isinstance(p.data, dask.array.Array) else np
    data = duck.apply_along_axis(func, p.get_axis_num("dparams"), p)

    # Create coordinate for the return periods
    coords = dict(p.coords.items())
    coords.pop("dparams")
    coords["quantile"] = q
    # Create dimensions
    dims = [d if d != "dparams" else "quantile" for d in p.dims]

    out = xr.DataArray(data=data, coords=coords, dims=dims)
    out.attrs = unprefix_attrs(p.attrs, ["units", "standard_name"], "original_")

    attrs = dict(
        long_name=f"{dist} quantiles",
        description=f"Quantiles estimated by the {dist} distribution",
        cell_methods=merge_attributes("dparams: ppf", out, new_line=" "),
        history=update_history(
            "Compute parametric quantiles from distribution parameters",
            new_name="parametric_quantile",
            parameters=p,
        ),
    )
    out.attrs.update(attrs)
    return out


def fa(
    da: xr.DataArray, t: Union[int, Sequence], dist: str = "norm", mode: str = "max"
) -> xr.DataArray:
    """Return the value corresponding to the given return period.

    Parameters
    ----------
    da : xr.DataArray
      Maximized/minimized input data with a `time` dimension.
    t : Union[int, Sequence]
      Return period. The period depends on the resolution of the input data. If the input array's resolution is
      yearly, then the return period is in years.
    dist : str
      Name of the univariate distribution, such as beta, expon, genextreme, gamma, gumbel_r, lognorm, norm
      (see scipy.stats).
    mode : {'min', 'max}
      Whether we are looking for a probability of exceedance (max) or a probability of non-exceedance (min).

    Returns
    -------
    xarray.DataArray
      An array of values with a 1/t probability of exceedance (if mode=='max').
    """
    # Fit the parameters of the distribution
    p = fit(da, dist)
    t = np.atleast_1d(t)

    if mode in ["max", "high"]:
        q = 1 - 1.0 / t

    elif mode in ["min", "low"]:
        q = 1.0 / t

    else:
        raise ValueError(f"Mode `{mode}` should be either 'max' or 'min'.")

    # Compute the quantiles
    out = (
        parametric_quantile(p, q)
        .rename({"quantile": "return_period"})
        .assign_coords(return_period=t)
    )
    out.attrs["mode"] = mode
    return out


def frequency_analysis(
    da: xr.DataArray,
    mode: str,
    t: Union[int, Sequence[int]],
    dist: str,
    window: int = 1,
    freq: Optional[str] = None,
    **indexer,
) -> xr.DataArray:
    """Return the value corresponding to a return period.

    Parameters
    ----------
    da : xarray.DataArray
      Input data.
    mode : {'min', 'max'}
      Whether we are looking for a probability of exceedance (high) or a probability of non-exceedance (low).
    t : int or sequence
      Return period. The period depends on the resolution of the input data. If the input array's resolution is
      yearly, then the return period is in years.
    dist : str
      Name of the univariate distribution, such as beta, expon, genextreme, gamma, gumbel_r, lognorm, norm
      (see scipy.stats).
    window : int
      Averaging window length (days).
    freq : str
      Resampling frequency. If None, the frequency is assumed to be 'YS' unless the indexer is season='DJF',
      in which case `freq` would be set to `AS-DEC`.
    **indexer : {dim: indexer, }, optional
      Time attribute and values over which to subset the array. For example, use season='DJF' to select winter values,
      month=1 to select January, or month=[6,7,8] to select summer months. If not indexer is given, all values are
      considered.

    Returns
    -------
    xarray.DataArray
      An array of values with a 1/t probability of exceedance or non-exceedance when mode is high or low respectively.

    """
    # Apply rolling average
    attrs = da.attrs.copy()
    if window > 1:
        da = da.rolling(time=window).mean(skipna=False)
        da.attrs.update(attrs)

    # Assign default resampling frequency if not provided
    freq = freq or generic.default_freq(**indexer)

    # Extract the time series of min or max over the period
    sel = generic.select_resample_op(da, op=mode, freq=freq, **indexer)

    # Frequency analysis
    return fa(sel, t, dist, mode)


def get_dist(dist):
    """Return a distribution object from `scipy.stats`."""
    from scipy import stats

    dc = getattr(stats, dist, None)
    if dc is None:
        e = f"Statistical distribution `{dist}` is not found in scipy.stats."
        raise ValueError(e)
    return dc


def get_lm3_dist(dist):
    """Return a distribution object from `lmoments3.distr`."""
    # fmt: off
    import lmoments3.distr  # isort: skip
    # The lmoments3 library has to be installed from the `develop` branch.
    # pip install git+https://github.com/OpenHydrology/lmoments3.git@develop#egg=lmoments3
    # fmt: on
    if dist not in _lm3_dist_map:
        raise ValueError(
            f"The {dist} distribution is not supported by `lmoments3` or `xclim`."
        )

    return getattr(lmoments3.distr, _lm3_dist_map[dist])


def _fit_start(x, dist, **fitkwargs) -> Tuple[Tuple, Dict]:
    """Return initial values for distribution parameters.

    Providing the ML fit method initial values can help the optimizer find the global optimum.

    Parameters
    ----------
    x : array-like
      Input data.
    dist : str
      Name of the univariate distribution, such as beta, expon, genextreme, gamma, gumbel_r, lognorm, norm
      (see scipy.stats). Only `genextreme` and `weibull_exp` distributions are supported.

    Returns
    -------
    tuple, dict

    References
    ----------
    Coles, S., 2001. An Introduction to Statistical Modeling of Extreme Values. Springer-Verlag, London, U.K., 208pp
    Cohen & Whittle, (1988) "Parameter Estimation in Reliability and Life Span Models", p. 25 ff, Marcel Dekker.
    """
    x = np.asarray(x)
    m = x.mean()
    v = x.var()

    if dist == "genextreme":
        s = np.sqrt(6 * v) / np.pi
        return (0.1,), {"loc": m - 0.57722 * s, "scale": s}

    if dist == "genpareto" and "floc" in fitkwargs:
        # Taken from julia' Extremes. Case for when "mu/loc" is known.
        t = fitkwargs["floc"]
        if not np.isclose(t, 0):
            m = (x - t).mean()
            v = (x - t).var()

        c = 0.5 * (1 - m ** 2 / v)
        scale = (1 - c) * m
        return (c,), {"scale": scale}

    if dist in ("weibull_min"):
        s = x.std()
        loc = x.min() - 0.01 * s
        chat = np.pi / np.sqrt(6) / (np.log(x - loc)).std()
        scale = ((x - loc) ** chat).mean() ** (1 / chat)
        return (chat,), {"loc": loc, "scale": scale}

    return (), {}


def _dist_method_1D(
    params: Sequence[float], arg=None, *, dist: str, function: str, **kwargs
) -> xr.DataArray:
    """Statistical function for given argument on given distribution initialized with params.

    See :ref:`scipy:scipy.stats.rv_continuous` for a available functions and their arguments.
    Every method where "*args" are the distribution parameters can be wrapped.

    Parameters
    ----------
    params: 1D sequence of floats
      Distribution parameters, in the same order as given by :py:func:`fit`.
    arg: optional, array_like
      The argument for the requested function.
    dist: str
      The scipy name of the distribution.
    function : str
      The name of the function to call.
    kwargs
      Other parameters to pass to the function call.

    Returns
    -------
    array_like
      Same shape as arg in most cases.
    """
    dist = get_dist(dist)
    args = ([arg] if arg is not None else []) + list(params)
    return getattr(dist, function)(*args, **kwargs)


def dist_method(
    function: str,
    fit_params: xr.DataArray,
    arg: Optional[xr.DataArray] = None,
    **kwargs,
) -> xr.DataArray:
    """Vectorized statistical function for given argument on given distribution initialized with params.

    See :ref:`scipy:scipy.stats.rv_continuous` for a available functions and their arguments.
    Methods where "*args" are the distribution parameters can be wrapped, except those
    that return new dimensions (Ex: 'rvs' with size != 1, 'stats' with more than one moment, 'interval', 'support')

    Parameters
    ----------
    function : str
      The name of the function to call.
    fit_params: xr.DataArray
      Distribution parameters are along `dparams`, in the same order as given by :py:func:`fit`.
      Must have a `scipy_dist` attribute with the name of the distribution fitted.
    arg: optional, array_like
      The argument for the requested function.
    kwargs
      Other parameters to pass to the function call.

    Returns
    -------
    array_like
      Same shape as arg.
    """

    args = [fit_params]
    input_core_dims = [["dparams"]]

    if arg is not None:
        args.append(arg)
        input_core_dims.append([])

    return xr.apply_ufunc(
        _dist_method_1D,
        *args,
        input_core_dims=input_core_dims,
        output_core_dims=[[]],
        kwargs={"dist": fit_params.attrs["scipy_dist"], "function": function, **kwargs},
        vectorize=True,
        output_dtypes=[float],
        dask="parallelized",
    )

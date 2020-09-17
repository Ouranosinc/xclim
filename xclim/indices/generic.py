# -*- coding: utf-8 -*-
# noqa: D205,D400
"""
Generic indices submodule
=========================

Helper functions for common generic actions done in the computation of indices.
"""
# Note: scipy.stats.dist.shapes: comma separated names of shape parameters
# The other parameters, common to all distribution, are loc and scale.
from typing import Optional, Sequence, Union

import dask.array
import numpy as np
import xarray as xr

from xclim.core.formatting import update_history

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


def select_time(da: xr.DataArray, **indexer):
    """Select entries according to a time period.

    Parameters
    ----------
    da : xr.DataArray
      Input data.
    **indexer : {dim: indexer, }, optional
      Time attribute and values over which to subset the array. For example, use season='DJF' to select winter values,
      month=1 to select January, or month=[6,7,8] to select summer months. If not indexer is given, all values are
      considered.

    Returns
    -------
    xr.DataArray
      Selected input values.
    """
    if not indexer:
        selected = da
    else:
        key, val = indexer.popitem()
        time_att = getattr(da.time.dt, key)
        selected = da.sel(time=time_att.isin(val)).dropna(dim="time")

    return selected


def select_resample_op(da: xr.DataArray, op: str, freq: str = "YS", **indexer):
    """Apply operation over each period that is part of the index selection.

    Parameters
    ----------
    da : xr.DataArray
      Input data.
    op : str {'min', 'max', 'mean', 'std', 'var', 'count', 'sum', 'argmax', 'argmin'} or func
      Reduce operation. Can either be a DataArray method or a function that can be applied to a DataArray.
    freq : str
      Resampling frequency defining the periods
      defined in http://pandas.pydata.org/pandas-docs/stable/timeseries.html#resampling.
    **indexer : {dim: indexer, }, optional
      Time attribute and values over which to subset the array. For example, use season='DJF' to select winter values,
      month=1 to select January, or month=[6,7,8] to select summer months. If not indexer is given, all values are
      considered.

    Returns
    -------
    xarray.DataArray
      The maximum value for each period.
    """
    da = select_time(da, **indexer)
    r = da.resample(time=freq, keep_attrs=True)
    if isinstance(op, str):
        return getattr(r, op)(dim="time", keep_attrs=True)

    return r.map(op)


def doymax(da: xr.DataArray):
    """Return the day of year of the maximum value."""
    i = da.argmax(dim="time")
    out = da.time.dt.dayofyear[i]
    out.attrs["units"] = ""
    return out


def doymin(da: xr.DataArray):
    """Return the day of year of the minimum value."""
    i = da.argmin(dim="time")
    out = da.time.dt.dayofyear[i]
    out.attrs["units"] = ""
    return out


def fit(da: xr.DataArray, dist: str = "norm", method="ML"):
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

    # Fit the parameters.
    # This would also be the place to impose constraints on the series minimum length if needed.
    def fitfunc(arr):
        """Fit distribution parameters."""
        x = np.ma.masked_invalid(arr).compressed()

        # Return NaNs if array is empty.
        if len(x) <= 1:
            return [np.nan] * len(dist_params)

        # Estimate parameters
        if method == "ML":
            args, kwargs = _fit_start(x, dist)
            params = dc.fit(x, *args, **kwargs)
        elif method == "PWM":
            params = list(lm3dc.lmom_fit(x).values())

        # Fill with NaNs if one of the parameters is NaN
        if np.isnan(params).any():
            params[:] = np.nan

        return params

    # xarray.apply_ufunc does not yet support multiple outputs with dask parallelism.
    duck = dask.array if isinstance(da.data, dask.array.Array) else np
    data = duck.apply_along_axis(fitfunc, da.get_axis_num("time"), da)

    # Coordinates for the distribution parameters
    coords = dict(da.coords.items())
    coords.pop("time")
    coords["dparams"] = dist_params

    # Dimensions for the distribution parameters
    dims = [d if d != "time" else "dparams" for d in da.dims]

    out = xr.DataArray(data=data, coords=coords, dims=dims)
    out.attrs = da.attrs
    out.attrs["original_name"] = da.attrs.get("standard_name", "")
    out.attrs["original_units"] = da.attrs.get("units", "")
    out.attrs[
        "description"
    ] = f"Parameters of the {dist} distribution fitted over {out.attrs['original_name']}"
    out.attrs["method"] = method
    out.attrs["estimator"] = method_name[method].capitalize()
    out.attrs["scipy_dist"] = dist
    out.attrs["units"] = ""
    out.attrs["xclim_history"] = update_history(
        f"Estimate distribution parameters by {method_name[method]} method.",
        new_name="fit",
        data=da,
    )
    return out


def parametric_quantile(p: xr.DataArray, q: Union[int, Sequence]):
    """Return the value corresponding to the given distribution parameters and quantile.

    Parameters
    ----------
    p : xr.DataArray
      Distribution parameters returned by the `fit` function. The array should have dimension `dparams` storing the
      distribution parameters, and attribute `scipy_dist`, storing the name of the distribution.
    q : Union[float, Sequence]
      Quantile to compute, which must be between 0 and 1 inclusive.

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
    out.attrs = p.attrs
    out.attrs["standard_name"] = f"{dist} quantile"
    out.attrs[
        "long_name"
    ] = f"{dist} return period values for {p.attrs.get('standard_name', '')}"
    out.attrs["cell_methods"] = (
        out.attrs.get("cell_methods", "") + " dparams: ppf"
    ).strip()
    out.attrs["units"] = p.attrs["original_units"]

    out.attrs["xclim_history"] = update_history(
        "Compute parametric quantiles from distribution parameters",
        new_name="parametric_quantile",
        parameters=p,
    )
    return out


def fa(
    da: xr.DataArray, t: Union[int, Sequence], dist: str = "norm", mode: str = "high"
):
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
):
    """Return the value corresponding to a return period.

    Parameters
    ----------
    da : xarray.DataArray
      Input data.
    t : int or sequence
      Return period. The period depends on the resolution of the input data. If the input array's resolution is
      yearly, then the return period is in years.
    dist : str
      Name of the univariate distribution, such as beta, expon, genextreme, gamma, gumbel_r, lognorm, norm
      (see scipy.stats).
    mode : {'min', 'max'}
      Whether we are looking for a probability of exceedance (high) or a probability of non-exceedance (low).
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
        da = da.rolling(time=window).mean(allow_lazy=True, skipna=False)
        da.attrs.update(attrs)

    # Assign default resampling frequency if not provided
    freq = freq or default_freq(**indexer)

    # Extract the time series of min or max over the period
    sel = select_resample_op(da, op=mode, freq=freq, **indexer)

    # Frequency analysis
    return fa(sel, t, dist, mode)


def default_freq(**indexer):
    """Return the default frequency."""
    freq = "AS-JAN"
    if indexer:
        group, value = indexer.popitem()
        if "DJF" in value:
            freq = "AS-DEC"
        if group == "month" and sorted(value) != value:
            raise NotImplementedError

    return freq


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


def _fit_start(x, dist):
    """Return initial values for distribution parameters.

    Providing the ML fit method initial values can help the optimizer find the global optimum.

    Parameters
    ----------
    x : array-like
      Input data.
    dist : str
      Name of the univariate distribution, such as beta, expon, genextreme, gamma, gumbel_r, lognorm, norm
      (see scipy.stats). Only `genextreme` and `weibull_exp` distributions are supported.

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

    if dist in ("weibull_min"):
        s = x.std()
        loc = x.min() - 0.01 * s
        chat = np.pi / np.sqrt(6) / (np.log(x - loc)).std()
        scale = ((x - loc) ** chat).mean() ** (1 / chat)
        return (chat,), {"loc": loc, "scale": scale}

    return (), {}


binary_ops = {">": "gt", "<": "lt", ">=": "ge", "<=": "le"}


def threshold_count(
    da: xr.DataArray, op: str, thresh: float, freq: str
) -> xr.DataArray:
    """Count number of days above or below threshold.

    Parameters
    ----------
    da : xr.DataArray
      Input data.
    op : str
      Logical operator {>, <, >=, <=, gt, lt, ge, le }. e.g. arr > thresh.
    thresh : float
      Threshold value.
    freq : str
      Resampling frequency defining the periods
      defined in http://pandas.pydata.org/pandas-docs/stable/timeseries.html#resampling.

    Returns
    -------
    xr.DataArray
      The number of days meeting the constraints for each period.
    """
    from xarray.core.ops import get_op

    if op in binary_ops:
        op = binary_ops[op]
    elif op in binary_ops.values():
        pass
    else:
        raise ValueError(f"Operation `{op}` not recognized.")

    func = getattr(da, "_binary_op")(get_op(op))
    c = func(da, thresh) * 1
    return c.resample(time=freq).sum(dim="time")


def get_daily_events(da: xr.DataArray, da_value: float, operator: str) -> xr.DataArray:
    r"""Return a 0/1 mask when a condition is True or False.

    the function returns 1 where operator(da, da_value) is True
                         0 where operator(da, da_value) is False
                         nan where da is nan

    Parameters
    ----------
    da : xr.DataArray
    da_value : float
    operator : str


    Returns
    -------
    xr.DataArray

    """
    events = operator(da, da_value) * 1
    events = events.where(~(np.isnan(da)))
    events = events.rename("events")
    return events


def daily_downsampler(da: xr.DataArray, freq: str = "YS") -> xr.DataArray:
    r"""Daily climate data downsampler.

    Parameters
    ----------
    da : xr.DataArray
    freq : str

    Returns
    -------
    xr.DataArray

    Note
    ----

        Usage Example

            grouper = daily_downsampler(da_std, freq='YS')
            x2 = grouper.mean()

            # add time coords to x2 and change dimension tags to time
            time1 = daily_downsampler(da_std.time, freq=freq).first()
            x2.coords['time'] = ('tags', time1.values)
            x2 = x2.swap_dims({'tags': 'time'})
            x2 = x2.sortby('time')
    """
    # generate tags from da.time and freq
    if isinstance(da.time.values[0], np.datetime64):
        years = [f"{y:04d}" for y in da.time.dt.year.values]
        months = [f"{m:02d}" for m in da.time.dt.month.values]
    else:
        # cannot use year, month, season attributes, not available for all calendars ...
        years = [f"{v.year:04d}" for v in da.time.values]
        months = [f"{v.month:02d}" for v in da.time.values]
    seasons = [
        "DJF DJF MAM MAM MAM JJA JJA JJA SON SON SON DJF".split()[int(m) - 1]
        for m in months
    ]

    n_t = da.time.size
    if freq == "YS":
        # year start frequency
        l_tags = years
    elif freq == "MS":
        # month start frequency
        l_tags = [years[i] + months[i] for i in range(n_t)]
    elif freq == "QS-DEC":
        # DJF, MAM, JJA, SON seasons
        # construct tags from list of season+year, increasing year for December
        ys = []
        for i in range(n_t):
            m = months[i]
            s = seasons[i]
            y = years[i]
            if m == "12":
                y = str(int(y) + 1)
            ys.append(y + s)
        l_tags = ys
    else:
        raise RuntimeError(f"Frequency `{freq}` not implemented.")

    # add tags to buffer DataArray
    buffer = da.copy()
    buffer.coords["tags"] = ("time", l_tags)

    # return groupby according to tags
    return buffer.groupby("tags")

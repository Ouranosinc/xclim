# -*- coding: utf-8 -*-
"""
Generic indices submodule
=========================

Helper functions for common generic actions done in the computation of indices.
"""
# Note: scipy.stats.dist.shapes: comma separated names of shape parameters
# The other parameters, common to all distribution, are loc and scale.
from typing import Sequence
from typing import Union

import dask.array
import numpy as np
import xarray as xr


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


def select_resample_op(da: xr.DataArray, op, freq: str = "YS", **indexer):
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


def fit(da: xr.DataArray, dist: str = "norm"):
    """Fit an array to a univariate distribution along the time dimension.

    Parameters
    ----------
    da : xr.DataArray
      Time series to be fitted along the time dimension.
    dist : str
      Name of the univariate distribution, such as beta, expon, genextreme, gamma, gumbel_r, lognorm, norm
      (see scipy.stats).

    Returns
    -------
    xr.DataArray
      An array of distribution parameters fitted using the method of Maximum Likelihood.

    Notes
    -----
    Coordinates for which all values are NaNs will be dropped before fitting the distribution. If the array
    still contains NaNs, the distribution parameters will be returned as NaNs.
    """
    # Get the distribution
    dc = get_dist(dist)
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

        # Fill with NaNs if one of the parameters is NaN
        params = dc.fit(x)
        if np.isnan(params).any():
            params[:] = np.nan

        return params

    # xarray.apply_ufunc does not yet support multiple outputs with dask parallelism.
    data = dask.array.apply_along_axis(fitfunc, da.get_axis_num("time"), da)

    # Count the number of values used for the fit.
    # n = da.notnull().count(dim='time')

    # Coordinates for the distribution parameters
    coords = dict(da.coords.items())
    coords.pop("time")
    coords["dparams"] = dist_params

    # Dimensions for the distribution parameters
    dims = ["dparams"]
    dims.extend(da.dims)
    dims.remove("time")

    out = xr.DataArray(data=data, coords=coords, dims=dims)
    out.attrs = da.attrs
    out.attrs["original_name"] = getattr(da, "standard_name", "")
    out.attrs[
        "description"
    ] = f"Parameters of the {dist} distribution fitted over {getattr(da, 'standard_name', '')}"
    out.attrs["estimator"] = "Maximum likelihood"
    out.attrs["scipy_dist"] = dist
    out.attrs["units"] = ""
    # out.name = 'params'
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
    t = np.atleast_1d(t)

    # Get the distribution
    dc = get_dist(dist)

    # Fit the parameters of the distribution
    p = fit(da, dist)

    # Create a lambda function to facilitate passing arguments to dask. There is probably a better way to do this.
    if mode in ["max", "high"]:

        def func(x):
            return dc.isf(1.0 / t, *x)

    elif mode in ["min", "low"]:

        def func(x):
            return dc.ppf(1.0 / t, *x)

    else:
        raise ValueError(f"Mode `{mode}` should be either 'max' or 'min'.")

    data = dask.array.apply_along_axis(func, p.get_axis_num("dparams"), p)

    # Create coordinate for the return periods
    coords = dict(p.coords.items())
    coords.pop("dparams")
    coords["return_period"] = t

    # Create dimensions
    dims = list(p.dims)
    dims.remove("dparams")
    dims.insert(0, "return_period")

    # TODO: add time and time_bnds coordinates (Low will work on this)
    # time.attrs['climatology'] = 'climatology_bounds'
    # coords['time'] =
    # coords['climatology_bounds'] =

    out = xr.DataArray(data=data, coords=coords, dims=dims)
    out.attrs = p.attrs
    out.attrs["standard_name"] = f"{dist} quantiles"
    out.attrs[
        "long_name"
    ] = f"{dist} return period values for {getattr(da, 'standard_name', '')}"
    out.attrs["cell_methods"] = (
        out.attrs.get("cell_methods", "") + " dparams: ppf"
    ).strip()
    out.attrs["units"] = da.attrs.get("units", "")
    out.attrs["mode"] = mode
    out.attrs["history"] = (
        out.attrs.get("history", "") + "Compute values corresponding to return periods."
    )

    return out


def frequency_analysis(da, mode, t, dist, window=1, freq=None, **indexer):
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
      in which case `freq` would be set to `YS-DEC`.
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
        if "DJF" in indexer.values():
            freq = "AS-DEC"
        if "month" in indexer and sorted(indexer.values()) != indexer.values():
            raise NotImplementedError

    return freq


def get_dist(dist):
    """Return a distribution object from scipy.stats.
    """
    from scipy import stats

    dc = getattr(stats, dist, None)
    if dc is None:
        e = f"Statistical distribution `{dist}` is not found in scipy.stats."
        raise ValueError(e)
    return dc


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
    r"""
    function that returns a 0/1 mask when a condition is True or False

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
    r"""Daily climate data downsampler

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

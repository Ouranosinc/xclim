"""The solar module offers functions for interpolating and accumulating variables to solar noon."""

import datetime
import importlib
import warnings
from typing import Literal

import numpy as np
import pandas as pd
import xarray as xr

import xclim as xc

for lib in ["pvlib", "astral", "ephem"]:
    if importlib.util.find_spec(lib):
        default_method = lib
        if lib != "pvlib":
            warnings.warn(f"pvlib library not found, default solar calculations will be performed with {lib}")
        break


def _solar_noon_astral_calc(t: np.ndarray, lon: np.ndarray):
    """
    Calculate solar noon with astral library

    Parameters
    ----------
    t : np.ndarray[datetime64]
        day datetime to calculate solar noon at.
    lon : np.ndarray[float]
        longitudes to calculate solar noon for.

    Returns
    -------
    np.ndarray[float]
        time of solar noon, as a fraction of hour away from the date.
    """
    import astral.sun

    vec_eot = np.vectorize(astral.sun.eq_of_time)
    if t.shape:
        date = pd.DatetimeIndex(t)
    else:
        date = pd.Timestamp(t)
    jdate = date.to_julian_date()
    jc = astral.sun.julianday_to_juliancentury(jdate)
    eot = vec_eot(jc)
    timeUTC = (720.0 - 4 * lon - eot) / 60.0

    return timeUTC


def _solar_noon_astral(ds):
    """
    Approximate solar noon values, using the NOAA algorithm implemented in `astral`. Faster than other methods, but
    accuracy is only correct on the order of ±10s.

    Requires the astral library (`pip install astral`).

    Parameters
    ----------
    ds : xr.Dataset or xr.DataArray
        Dataset with time and lon variables when to calculate solar noon.

    Returns
    -------
    xr.DataArray
        Times when solar noon is expected to occur
    """
    solar_noon_timedelta = xr.apply_ufunc(
        _solar_noon_astral_calc,
        ds.time,
        ds.lon,
        input_core_dims=[[], [*ds.lon.dims]],
        output_core_dims=[[*ds.lon.dims]],
        output_dtypes=["float"],
        vectorize=True,
        dask="parallelized",
        dask_gufunc_kwargs=dict(),
    )

    # solar_noon_timedelta is in fractions of hour, as a float, for dask compatibility.
    # We convert to int (timedelta64[s]), to not give better impressions on our accuracy.
    return ds.time + (solar_noon_timedelta * 3600).astype("timedelta64[s]")


def _solar_noon_ephem_calc(lat: float, lon: float, day: datetime.date, elev: float = 0.0):
    """
    Calculate solar noon with ephem library

    Parameters
    ----------
    lat : float
        latitudes to calculate solar noon for.

    lon : float
        longitudes to calculate solar noon for.

    day : datetime.date
        date for calculation of solar noon

    sun : ephem.Sun() instance

    elev : float, optional
        elevation of observer

    Returns
    -------
    pd.Timestamp
        time of solar noon
    """
    import ephem

    sun = ephem.Sun()
    o = ephem.Observer()
    # need to project lat/lon to EPSG:6648.
    o.lat = lat * np.pi / 180
    o.lon = lon * np.pi / 180
    o.elevation = elev
    o.date = day
    noon = o.next_transit(sun)
    return pd.Timestamp(noon.datetime())


def _solar_noon_ephem(ds):
    """
    High precision calculation of solar noon using PyEphem. This is not vectorized, so it is quite slow.

    Requires the ephem library (`pip install ephem`).

    Parameters
    ----------
    ds : xr.Dataset or xr.DataArray
        Dataset with time, lat and lon variables when to calculate solar noon.

    Returns
    -------
    xr.DataArray
        Times when solar noon is expected to occur
    """
    return xr.apply_ufunc(
        _solar_noon_ephem_calc,
        ds.lat[0],  # lat is needed for sunset/sunrise only.
        ds.lon,
        ds.time.dt.date,
        input_core_dims=[[], [], []],
        output_core_dims=[[]],
        vectorize=True,
    )


def solar_noon_pvlib(ds):
    """
    High precision calculation of solar noon using pvlib, accurate up to ±1s.

    Requires the pvlib library (`pip install pvlib`).

    Parameters
    ----------
    ds : xr.Dataset or xr.DataArray
        Dataset with time, lat and lon variables when to calculate solar noon.

    Returns
    -------
    xr.DataArray
        Times when solar noon is expected to occur.
    """
    import pvlib

    deltat = xr.DataArray(
        pvlib.spa.calculate_deltat(ds.time.dt.year, ds.time.dt.month), coords={"time": ds.time}, dims=["time"]
    )
    (
        transit,
        _sunrise,
        _sunset,
    ) = xr.apply_ufunc(
        pvlib.spa.transit_sunrise_sunset,
        ds.time.astype("datetime64[s]").astype("int"),  # seconds since epoch
        ds.lat[0],  # lat is needed for sunset/sunrise only.
        ds.lon,
        deltat,
        kwargs={"numthreads": 1},
        input_core_dims=[["time"], [], [], ["time"]],
        output_core_dims=[["time"], ["time"], ["time"]],
        # output_dtypes=['datetime64[s]','datetime64[s]','datetime64[s]'],
        vectorize=True,
    )
    return transit.astype("datetime64[s]")


def solar_noon(ds, method: Literal["pvlib", "astral", "ephem"] = default_method):
    """
    Return the solar noon time for the given dataset, assuming UTC.

    Requires ds.time, ds.lon, and ds.lat.

    Requires one of 3 libraries: pvlib, PyEphem, or astral.

    Parameters
    ----------
    ds : xr.DataArray or xr.Dataset
        Dataset with variables ds.time, ds.lon, and ds.lat.
    method : {"pvlib", "astral", "ephem"}
        Method to use to calculate solar noon, by default
        uses first available library from ['pvlib','astral','ephem'].

    Returns
    -------
    xr.DataArray
        DataArray with solar noon time.
    """
    do_calc = None
    if method == "pvlib":
        do_calc = solar_noon_pvlib
    elif method == "astral":
        do_calc = _solar_noon_astral
    elif method == "ephem":
        do_calc = _solar_noon_ephem
    else:
        errmsg = f"Method does not exist: {method}"
        ValueError(errmsg)
    return do_calc(ds)


def sel_with_nans(da, dim, sel, label="tmp_time", fill=np.nan, lazy=True):
    """
    Select from da on dimension dim, with DataArray from the *sorted* da[dim] index.

    This is similar to xc.core.utils.lazy_indexing, but allows for labelled indexing,
    and fills locations with `fill` if not available.
    It is similar to xr.reindex, but allows for multi-dimensional reindexing.
    It is also similar to xr.sel, but allows for lazy evaluation and filling for unavailable selections.

    Parameters
    ----------
    da : xr.DataArray
        DataArray to select from. Requires `dim` dimension.
    dim : str
        Dimension over which to select.
    sel : xr.DataArray
        DataArray with which to select. Requires `dim` dimension.
    label : str
        Label to rename dim in `da`, by default "tmp_time".
    fill : float
        Fill value if sel does not exist in da[dim], by default np.nan.
    lazy : bool
        Whether to compute immediately, or evaluate lazily with dask, by default True.

    Returns
    -------
    xr.DataArray
        DataArray `da` selected on dimension `dim` with selection `sel`.

    Warnings
    --------
    This function can be quite memory intensive. Optimizing chunking may help.
    """
    # sel = sel.rename({dim:label})
    dimchunks = {d: s[0] for d, s in da.chunksizes.items() if d != dim}
    sel = sel.chunk({dim: -1, **dimchunks})

    da = da.rename({dim: label})
    dim = label
    ind_insert = xr.apply_ufunc(
        lambda n: da.indexes[dim].searchsorted(n, "left"),
        sel,
        dask="parallelized",
    )

    if lazy:
        lazy_index = xc.core.utils.lazy_indexing(da.chunk({dim: -1}), ind_insert, dim)
        lazy_time = xc.core.utils.lazy_indexing(da[dim].chunk({dim: -1}), ind_insert, dim)
    else:
        ind_insert = ind_insert.compute()
        lazy_index = da.isel({dim: ind_insert})
        lazy_time = da[dim].isel({dim: ind_insert})
    index_correct = lazy_time == sel
    out = xr.where(index_correct, lazy_index, fill)

    return out


def get_dt(freq):
    """
    Get the time delta, in seconds for a given pandas frequency.

    Parameters
    ----------
    freq : str
        Pandas time frequency.

    Returns
    -------
    float
        Total seconds between two timestamps with this frequency.
    """
    return pd.date_range(freq=freq, periods=2, start="2000-01-01").diff()[1].total_seconds()


def accumulate_between_times(ds, var, freq, prev_time, curr_time):
    """
    Accumulate (sum) between the given time DataArray (usually solar noon yesterday and solar noon today).

    Parameters
    ----------
    ds : xr.Dataset
        Dataset with variable `var` to accumulate.
    var : str
        Variable to accumulate.
    freq : str
        Pandas frequency for ds.time.
    prev_time : xr.DataArray
        Time occurrence of the previous event (indexed at the current time).
    curr_time : xr.DataArray
        Time occurrence of the current event (indexed at the current time).

    Returns
    -------
    xr.DataArray
        Variable accumulated between prev_time and curr_time.
    """
    da = ds[var]
    dt = get_dt(freq)
    da_cum = da.cumsum("time")

    curr_fl = curr_time.dt.floor(freq)
    curr_ratio = (curr_time - curr_fl).dt.total_seconds() / dt

    prev_fl = prev_time.dt.floor(freq)
    prev_ratio = (prev_time - prev_fl).dt.total_seconds() / dt

    d_tilcurr = sel_with_nans(da_cum, "time", curr_fl - pd.Timedelta(dt, "s"))
    d_curr = sel_with_nans(da, "time", curr_fl)
    d_tilprev = sel_with_nans(da_cum, "time", prev_fl - pd.Timedelta(dt, "s"))
    d_prev = sel_with_nans(da, "time", prev_fl)
    da_accum = (d_tilcurr + curr_ratio * d_curr) - (d_tilprev + prev_ratio * d_prev)

    return da_accum


def interpolate_to_time(ds, var, freq, curr_time):
    """
    Interpolate Dataset to the given time DataArray (such as Solar noon times).

    This is equivalent to ds[var].interp(time=curr_time), but tends to be faster.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset to interpolate, with dimension time and variable `var`.
    var : str
        Variable of ds to interpolate.
    freq : str
        Pandas frequency for ds.time.
    curr_time : xr.DataArray
        Time array to interpolate.

    Returns
    -------
    xr.DataArray
        DataArray of interpolated times.
    """
    da = ds[var]
    dt = get_dt(freq)

    curr_time_fl = curr_time.dt.floor(freq)
    curr_time_cl = curr_time.dt.ceil(freq)

    curr_ratio = (curr_time - curr_time_fl).dt.total_seconds() / dt

    d_curr_fl = sel_with_nans(da, "time", curr_time_fl)
    d_curr_cl = sel_with_nans(da, "time", curr_time_cl)

    da_interp = (1 - curr_ratio) * d_curr_fl + curr_ratio * d_curr_cl
    return da_interp


def interpolate_to_solar_noon(da, method="interpolate", solar_method=default_method):
    """
    Interpolate (or accumulate) da to solar noon.

    If DataArray is precipitation data, and is accumulated, then the output units will be converted to mm/d.

    Parameters
    ----------
    da : xr.Dataset or xr.DataArray
        Data to interpolate.
    method : str, optional
        Either `interpolate` at solar noon, or `accumulate` between solar noons.
        Defaults to 'interpolate'.
    solar_method : str, optional
        Python library to use to perform solar noon calculations.
        Defaults to first available library from ['pvlib', 'astral', 'ephem'].

    Returns
    -------
        xr.Dataset or xr.DataArray:
            Data interpolated/accumulated to solar noon.
    """
    if da.isinstance(xr.DataArray):
        ds = da.to_dataset()
        var = da.name
    elif da.isinstance(xr.Dataset):
        output = []
        for var in da.data_vars:
            output.append(interpolate_to_solar_noon(da[var], method=method, solar_method=solar_method))
        ds_out = xr.merge(output)
        ds_out.attrs = da.attrs.copy()
        ds_out.attrs["history"] = xc.core.formatting.update_history(f"Interpolated to solar noon with {solar_method}")
        return ds_out
    elif ds.isinstance(xr.DataTree):
        NotImplementedError("interpolate_to_solar_noon is not implemented for DataTrees")
    else:
        ValueError("da must be an instance of xr.DataArray or xr.Dataset.")

    if method not in ["accumulate", "interpolate"]:
        raise ValueError("Unknown method passed to interpolate_to_solar_noon, need one of [accumulate, interpolate]")

    if method == "accumulate" and var == "pr":
        ds["pr"] = xc.units.convert_units_to(ds["pr"], "mm/hr")

    days = ds.time.resample(time="1D").first().dt.floor("D")
    c = {d: ds[d] for d in ds[var].coords if d != "time"}
    ds_days = xr.Dataset({}, coords={"time": days, **c})
    ds_days["noon"] = solar_noon(ds_days, method=solar_method)
    freq = xr.infer_freq(ds.time)

    if method == "accumulate":
        # We need yesterday's noon to accumulate between.
        ds_days["noon_yst"] = solar_noon(
            ds_days.assign_coords(time=ds_days.time - pd.Timedelta(1, "day")), method=solar_method
        ).assign_coords(time=ds_days.time)

    kwargs = dict(
        var=var,
        freq=freq,
        curr_time=ds_days.noon,
    )
    do_calc = interpolate_to_time
    if method == "accumulate":
        do_calc = accumulate_between_times
        kwargs["prev_time"] = ds_days.noon_yst
    da_out = do_calc(ds, **kwargs)

    # set metadata
    # if precip, set proper units.
    da_out.attrs = ds[var].attrs.copy()
    if method == "interpolate":
        da_out.attrs["cell_methods"] = "time: point (comment: at solar noon)"
        da_out.attrs["comments"] = f"Interpolated date to solar noons, calculated with {default_method} library."
    elif method == "accumulate":
        if var == "pr":
            da_out.attrs["units"] = "mm/d"
        da_out.attrs["cell_methods"] = "time: sum (interval: 24 hr comment: since solar noon)"
        da_out.attrs["comments"] = f"Accumulated between solar noons, calculated with {default_method} library."
    da_out.name = var
    return da_out

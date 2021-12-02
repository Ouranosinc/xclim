import xarray
""" Diagnotics is the xclim term for 'indices' in the VALUE project"""

res2freq = {'year': 'YS', 'season': 'QS-DEC', 'month': 'MS'}


def mean(da: xarray.DataArray, time_res: str = 'year') -> xarray.DataArray:
    """Mean.

    Mean over a {time_res}, then averaged over all years.

    Parameters
    ----------
    da : xarray.DataArray
      Variable on which to calculate the diagnostic.
    time_res : str
      Time resolution

    Returns
    -------
    xarray.DataArray,
      Mean of the variable over the time resolution.

    Examples
    --------
    >>> pr = xr.open_dataset(path_to_pr_file).pr
    >>> mean(da=pr, time_res='season')
    """
    da_rs = da.resample(time=res2freq[time_res])
    out = da_rs.mean(dim="time", keep_attrs=True)
    out = out.groupby(f'time.{time_res}').mean(dim='time')
    out.attrs.update(da.attrs)
    out.attrs["long_name"] = f"Mean of {da.attrs['standard_name']}"
    out.attrs["units"] = da.attrs["units"]
    return out


def var(da: xarray.DataArray, time_res: str = 'YS') -> xarray.DataArray:
    """Variance.

    Mean over a {time_res}, then averaged over all years.

    Parameters
    ----------
    da : xarray.DataArray
      Variable on which to calculate the diagnostic.
    time_res : str
      Time resolution

    Returns
    -------
    xarray.DataArray,
      Variance of the variable over the time resolution.

    Examples
    --------
    >>> pr = xr.open_dataset(path_to_pr_file).pr
    >>> var(da=pr, time_res='season')
    """
    da_rs = da.resample(time=res2freq[time_res])
    out = da_rs.mean(dim="time", keep_attrs=True)
    out = out.groupby(f'time.{time_res}').mean(dim='time')
    out.attrs.update(da.attrs)
    out.attrs["long_name"] = f"Variance of {da.attrs['standard_name']}"
    out.attrs["units"] = f"({da.attrs['units']})**2"  # est-ce qu'on a le droit????
    return out


def acf(da: xarray.DataArray, lag_max: int = 1, time_res: str = 'YS') -> xarray.DataArray:
    """Lag autocorrelation.

    Mean over a {time_res}, then averaged over all years.

    Parameters
    ----------
    da : xarray.DataArray
      Variable on which to calculate the diagnostic.
    time_res : str
      Time resolution

    Returns
    -------
    xarray.DataArray,
      Variance of the variable over the time resolution.

    Examples
    --------
    >>> pr = xr.open_dataset(path_to_pr_file).pr
    >>> var(da=pr, time_res='season')
    """
    da_rs = da.resample(time=res2freq[time_res])
    out = da_rs.mean(dim="time", keep_attrs=True)
    out = out.groupby(f'time.{time_res}').mean(dim='time')
    out.attrs.update(da.attrs)
    out.attrs["long_name"] = f"Variance of {da.attrs['standard_name']}"
    out.attrs["units"] = f"({da.attrs['units']})**2"  # est-ce qu'on a le droit????
    return out

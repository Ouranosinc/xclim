from __future__ import annotations


from xclim.core.units import declare_units


declare_units(q="[discharge]")


def flow_index(q: xr.DataArray, p: float = 0.95) -> xr.DataArray:
    """
    Calculate the Qp  (pth percentile of daily streamflow) normalized by the mean flow.

    Reference:
    1. Addor, Nans & Nearing, Grey & Prieto, Cristina & Newman, A. & Le Vine, Nataliya & Clark, Martyn. (2018). A Ranking of Hydrological Signatures Based on Their Predictability in Space. Water Resources Research. 10.1029/2018WR022606.
    2. Clausen, B., & Biggs, B. J. F. (2000). Flow variables for ecological studies in temperate streams: Groupings based on covariance. Journal of Hydrology, 237(3–4), 184–197. https://doi.org/10.1016/S0022-1694(00)00306-1

    Parameters
    ----------
    q : xarray.DataArray
        Daily streamflow data.
    p is the percentile for calculating the flow index, specified as a float between 0 and 1, default is 0.95.

    Returns
    -------
    xarray.DataArray
        out = Normalized Qp, which is the p th percentile of daily streamflow normalized by the median flow.

    """
    qp = q.quantile(p, dim="time")
    q_median = q.median(dim="time")
    out = qp / q_median
    out.attrs["units"] = " "
    return out.rename("flow_index")


@declare_units(q="[discharge]")
def high_flow_frequency(
    q: xr.DataArray,
    threshold_factor: int = 9.0,
    freq: str = "A-SEP",
    statistic: str = "mean",
) -> xr.DataArray:
    """
    Calculate the mean number of days in a given period with flows greater than a specified threshold. By default, the period is the water year starting on 1st October and ending on 30th September, as commonly defined in North America.

    Reference:
    1. Addor, Nans & Nearing, Grey & Prieto, Cristina & Newman, A. & Le Vine, Nataliya & Clark, Martyn. (2018). A Ranking of Hydrological Signatures Based on Their Predictability in Space. Water Resources Research. 10.1029/2018WR022606.
    2. Clausen, B., & Biggs, B. J. F. (2000). Flow variables for ecological studies in temperate streams: Groupings based on covariance. Journal of Hydrology, 237(3–4), 184–197. https://doi.org/10.1016/S0022-1694(00)00306-1

    Parameters
    ----------
    q : xarray.DataArray
        Daily streamflow data.
    threshold_factor : float, optional
        Factor by which the median flow is multiplied to set the high flow threshold, default is 9.0.
    freq : str, optional
        Resampling frequency, default is 'A-SEP' for water year ending in September.
    statistic : str, optional
        Type of statistic to return ('mean', 'sum', 'max', median etc.), default is 'mean'.

    Returns
    -------
    xarray.DataArray
        Calculated statistic of high flow days per water year, by default it is set as mean
    """
    median_flow = q.median(dim="time")
    threshold = threshold_factor * median_flow

    # Resample data to the given frequency and count days above threshold
    high_flow_days = (q > threshold).resample(time=freq).sum(dim="time")

    # Dynamically apply the chosen statistic using getattr
    out = getattr(high_flow_days, statistic)(dim="time")

    # Assign units to the result based on the statistic
    out.attrs["units"] = "days/year" if statistic == "mean" else "days"

    # Rename the result for clarity
    return out.rename(f"high flow frequency({statistic})")


@declare_units(q="[discharge]")
def low_flow_frequency(
    q: xr.DataArray,
    threshold_factor: float = 0.2,
    freq: str = "A-SEP",
    statistic: str = "mean",
) -> xr.DataArray:
    """
    Calculate the specified statistic of the number of days in a given period with flows lower than a specified threshold.
    By default, the period is the water year starting on 1st October and ending on 30th September, as commonly defined in North America.

    Reference:
    Olden, J. D., & Poff, N. L. (2003). Redundancy and the choice of hydrologic indices for characterizing streamflow regimes. River Research and Applications, 19(2), 101–121. https://doi.org/10.1002/rra.700

    Parameters
    ----------
    q : xarray.DataArray
        Daily streamflow data.
    threshold_factor : float, optional
        Factor by which the mean flow is multiplied to set the low flow threshold, default is 0.2.
    freq : str, optional
        Resampling frequency, default is 'A-SEP' for water year ending in September.
    statistic : str, optional
        Type of statistic to return ('mean', 'sum', 'max', median etc.), default is 'mean'.

    Returns
    -------
    xarray.DataArray
        Calculated statistic of low flow days per water year, by default it is set as mean
    """
    mean_flow = q.mean(dim="time")
    threshold = threshold_factor * mean_flow

    # Resample data to the given frequency and count days below threshold
    low_flow_days = (q < threshold).resample(time=freq).sum(dim="time")

    # Dynamically apply the chosen statistic using getattr
    out = getattr(low_flow_days, statistic)(dim="time")

    # Assign units to the result based on the statistic
    out.attrs["units"] = "days/year" if statistic == "mean" else "days"

    # Rename the result for clarity
    return out.rename(f"low flow frequency({statistic})")

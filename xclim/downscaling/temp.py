"""Temporary functions to be deleted when xarray implements them."""
import dask.array
import numpy as np
import xarray as xr
from xarray.core.missing import get_clean_interp_index


# TODO: use xr.polyfit once it's implemented.
def polyfit(da, deg=1, dim="time"):
    """
    Least squares polynomial fit.

    Fit a polynomial ``p(x) = p[deg] * x ** deg + ... + p[0]`` of degree `deg`
    Returns a vector of coefficients `p` that minimises the squared error.
    Parameters
    ----------
    da : xarray.DataArray
        The array to fit
    deg : int, optional
        Degree of the fitting polynomial, Default is 1.
    dim : str
        The dimension along which the data will be fitted. Default is `time`.

    Returns
    -------
    output : xarray.DataArray
        Polynomial coefficients with a new dimension to sort the polynomial
        coefficients by degree
    """
    # Remove NaNs
    y = da.dropna(dim=dim, how="any")

    # Compute the x value.
    x = get_clean_interp_index(da, dim)

    # Fit the parameters (lazy computation)
    coefs = dask.array.apply_along_axis(
        np.polyfit, da.get_axis_num(dim), x, y, deg=deg, shape=(deg + 1,), dtype=float
    )

    coords = dict(da.coords.items())
    coords.pop(dim)
    coords["degree"] = range(deg, -1, -1)

    dims = list(da.dims)
    dims.remove(dim)
    dims.insert(0, "degree")

    out = xr.DataArray(data=coefs, coords=coords, dims=dims)
    return out


# TODO: use xr.polyval once it's implemented.
def polyval(coefs, coord):
    """
    Evaluate polynomial function.

    Parameters
    ----------
    coord : xr.Coordinate
      Coordinate (e.g. time) used as the independent variable to compute polynomial.
    coefs : xr.DataArray
      Polynomial coefficients as returned by polyfit.
    """
    x = xr.Variable(data=get_clean_interp_index(coord, coord.name), dims=(coord.name,))

    y = xr.apply_ufunc(
        np.polyval, coefs, x, input_core_dims=[["degree"], []], dask="allowed"
    )

    return y

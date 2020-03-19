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
    # Compute the x value.
    x = get_clean_interp_index(da, dim)
    x = np.vander(x, deg + 1)

    def _nanpolyfit_1d(arr, x, rcond=None):
        out = np.full((x.shape[1],), np.nan)
        mask = np.isnan(arr)
        if not np.all(mask):
            out[:], _, _, _ = np.linalg.lstsq(x[~mask, :], arr[~mask], rcond=rcond)
        return out

    # Fit the parameters (lazy computation)
    coefs = dask.array.apply_along_axis(
        _nanpolyfit_1d, da.get_axis_num(dim), da, x, shape=(deg + 1,), dtype=float
    )

    coords = dict(da.coords.items())
    coords.pop(dim)
    coords["degree"] = range(deg, -1, -1)

    dims = list(da.dims)
    dims.remove(dim)
    dims.insert(0, "degree")

    out = xr.DataArray(
        data=coefs, coords=coords, dims=dims, name="polyfit_coefficients"
    )
    return out


# TODO: use xr.polyval once it's implemented.
def polyval(coord, coeffs, degree_dim="degree"):
    """Evaluate a polynomial at specific values
    Parameters
    ----------
    coord : DataArray
        The 1D coordinate along which to evaluate the polynomial.
    coeffs : DataArray
        Coefficients of the polynomials.
    degree_dim : str, default "degree"
        Name of the polynomial degree dimension in `coeffs`.
    See also
    --------
    xarray.DataArray.polyfit
    numpy.polyval
    """
    x = get_clean_interp_index(coord, coord.name)

    deg_coord = coeffs[degree_dim]

    lhs = xr.DataArray(
        np.vander(x, int(deg_coord.max()) + 1),
        dims=(coord.name, degree_dim),
        coords={coord.name: coord, degree_dim: np.arange(deg_coord.max() + 1)[::-1]},
    )
    return (lhs * coeffs).sum(degree_dim)

"""Detrending objects"""
import os
import tempfile
from pathlib import Path
from typing import Optional
from typing import Union

import xarray as xr

from .base import Grouper
from .base import Parametrizable
from .base import parse_group
from .utils import ADDITIVE
from .utils import apply_correction
from .utils import invert

# from .utils import loffsets


class BaseDetrend(Parametrizable):
    """Base class for detrending objects

    Defines three methods:

    fit(da)      : Compute trend from da and return a new _fitted_ Detrend object.
    get_trend(da): Return the fitted trend along da's coordinate.
    detrend(da)  : Return detrended array.
    retrend(da)  : Puts trend back on da.

    * Subclasses should implement _fit() and _get_trend(). Both will be called in a `group.apply()`.
    `_fit()` is called with the dataarray and str `dim` that indicates the fitting dimension,
        it should return a dataset that will be set as `.fitds`.
    `_get_trend()` is called with .fitds broadcasted on the main dim of the input DataArray.
    """

    @parse_group
    def __init__(
        self, *, group: Union[Grouper, str] = "time", kind: str = "+", **kwargs
    ):
        """Initialize Detrending object.

        Parameters
        ----------
        group : Union[str, Grouper]
            The grouping information. See :py:class:`xclim.sdba.base.Grouper` for details.
            The fit is performed along the group's main dim.
        kind : {'*', '+'}
            The way the trend is removed or added, either additive or multiplicative.
        """
        self.__fitted = False
        self.__ds_is_tempfile = False
        super().__init__(group=group, kind=kind, **kwargs)

    def fit(self, da: xr.DataArray):
        """Extract the trend of a DataArray along a specific dimension.

        Returns a new object storing the fit data that can be used for detrending and retrending.
        """
        new = self.copy()
        new._set_fitds(new.group.apply(new._fit, da, main_only=True))
        new.__fitted = True
        return new

    def get_trend(self, da: xr.DataArray):
        """Get the trend computed from the fit, the fitting dim as found on da.

        If da is a DataArray (and has a "dtype" attribute), the trend is casted to have the same dtype.
        """
        out = self.group.apply(
            self._get_trend,
            {self.group.dim: da[self.group.dim], **self.fit_ds.data_vars},
            main_only=True,
        )
        if hasattr(da, 'dtype'):
            out = out.astype(da.dtype)
        return out

    def detrend(self, da: xr.DataArray):
        """Removes the previously fitted trend from a DataArray."""
        if not self.__fitted:
            raise ValueError("You must call fit() before detrending.")
        trend = self.get_trend(da)
        return self._detrend(da, trend)

    def retrend(self, da: xr.DataArray):
        """Puts back the previsouly fitted trend on a DataArray."""
        if not self.__fitted:
            raise ValueError("You must call fit() before retrending")
        trend = self.get_trend(da)
        return self._retrend(da, trend)

    def save_fit(
        self, filename: Optional[Union[Path, str]] = None, tempdir: Optional[str] = None
    ):
        """Save fit data to a (temporary) file.

        Save to a temporary file if `filename` is not given. The file will be
        deleted when this Adjustment instance is deleted.

        The dataset is immediately reload from file. This is meant to help divide dask's
        workload when needed.

        Parameters
        ----------
        filename : Optional[Union[Path, str]]
          Filename of the saved file. When given, the file is not considered "temporary"
          and is not deleted when the Detrending object is deleted by Python.
        tempdir : Optional[str]
          The path to a directory where to save the temporary file. Ignored if `filename`
          is given.
        """
        if filename is None:
            # We use mkstemp to be sure the filename is reserved.
            fid, filename = tempfile.mkstemp(suffix=".nc", dir=tempdir)
            os.close(fid)  # Passing file-like objects is too restrictive with xarray.
            self.__ds_is_tempfile = True  # So that the file is deleted when this instance is garbage collected

        self._ds_file = Path(filename)
        previous_chunking = (
            self.fit_ds.chunks
        )  # Expected behavior is to conserve chunking
        self.fit_ds.to_netcdf(self._ds_file)
        # chunks: non-dask data will return an empty set on ds.chunks, but that means 1 chunk for open_dataset
        # `previous_chunking or None` returns None is ds.chunks was an empty set
        self.fit_ds = xr.open_dataset(self._ds_file, chunks=previous_chunking or None)

    def _set_fitds(self, ds):
        self.fit_ds = ds

    def _detrend(self, da, trend):
        # Remove trend from series
        return apply_correction(da, invert(trend, self.kind), self.kind)

    def _retrend(self, da, trend):
        # Add trend to series
        return apply_correction(da, trend, self.kind)

    def _get_trend(self, grpd, dim="time"):
        raise NotImplementedError

    def _fit(self, da):
        raise NotImplementedError

    def __del__(self):
        # Delete the training data file if it was saved to a temporary file.
        if self.__ds_is_tempfile and hasattr(self, "_ds_file"):
            self._ds_file.unlink()


class NoDetrend(BaseDetrend):
    """Convenience class for polymorphism. Does nothing."""

    def _fit(self, da, dim=None):
        return da.isel({dim: 0})

    def _detrend(self, da, trend):
        return da

    def _retrend(self, da, trend):
        return da


class MeanDetrend(BaseDetrend):
    """Simple detrending removing only the mean from the data, quite similar to normalizing."""

    def _fit(self, da, dim="time"):
        mean = da.mean(dim=dim)
        mean.name = "mean"
        return mean

    def _get_trend(self, grpd, dim="time"):
        return grpd.mean


class PolyDetrend(BaseDetrend):
    """
    Detrend time series using a polynomial regression.

    Parameters
    ----------
    group : Union[str, Grouper]
        The grouping information. See :py:class:`xclim.sdba.base.Grouper` for details.
        The fit is performed along the group's main dim.
    kind : {'*', '+'}
        The way the trend is removed or added, either additive or multiplicative.
    degree : int
        The order of the polynomial to fit.
    preserve_mean : bool
        Whether to preserve the mean when de/re-trending. If True, the trend has its mean
        removed before it is used.
    """

    def __init__(self, group="time", kind=ADDITIVE, degree=4, preserve_mean=False):
        super().__init__(
            group=group, kind=kind, degree=degree, preserve_mean=preserve_mean
        )

    def _fit(self, da, dim="time"):
        return da.polyfit(dim=dim, deg=self.degree)

    def _get_trend(self, grpd, dim="time"):
        # Estimate trend over da
        trend = xr.polyval(coord=grpd[dim], coeffs=grpd.polyfit_coefficients)

        if self.preserve_mean:
            trend = apply_correction(
                trend, invert(trend.mean(dim=dim), self.kind), self.kind
            )

        return trend

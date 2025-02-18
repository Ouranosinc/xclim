from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from xclim.sdba import nbutils as nbu


class TestQuantiles:
    @pytest.mark.parametrize("uses_dask", [True, False])
    def test_quantile(self, open_dataset, uses_dask):
        da = (open_dataset("sdba/CanESM2_1950-2100.nc").sel(time=slice("1950", "1955")).pr).load()
        if uses_dask:
            da = da.chunk({"location": 1})
        else:
            da = da.load()
        q = np.linspace(0.1, 0.99, 50)
        out_nbu = nbu.quantile(da, q, dim="time").transpose("location", ...)
        out_xr = da.quantile(q=q, dim="time").transpose("location", ...)
        np.testing.assert_array_almost_equal(out_nbu.values, out_xr.values)

    def test_edge_cases(self, open_dataset):
        q = np.linspace(0.1, 0.99, 50)

        # only 1 non-null value
        da = xr.DataArray([1] + [np.nan] * 100, dims="dim_0")
        out_nbu = nbu.quantile(da, q, dim="dim_0")
        np.testing.assert_array_equal(out_nbu.values, np.full_like(q, 1))

        # only NANs
        da = xr.DataArray([np.nan] * 100, dims="dim_0")
        out_nbu = nbu.quantile(da, q, dim="dim_0")
        np.testing.assert_array_equal(out_nbu.values, np.full_like(q, np.nan))

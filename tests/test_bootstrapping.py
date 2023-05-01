from __future__ import annotations

from typing import Callable

import numpy as np
import pytest
from xarray.core.dataarray import DataArray

from xclim.core.calendar import percentile_doy
from xclim.indices import (
    cold_spell_duration_index,
    days_over_precip_thresh,
    fraction_over_precip_thresh,
    tg10p,
    tg90p,
    tn10p,
    tn90p,
    tx10p,
    tx90p,
    warm_spell_duration_index,
)


class Test_bootstrap:
    @pytest.mark.slow
    @pytest.mark.parametrize("use_dask", [True, False])
    def test_bootstrap_tg90p(self, tas_series, use_dask):
        self.bootstrap_testor(
            tas_series,
            98,
            lambda x, y, z: tg90p(x, y, freq="MS", bootstrap=z),
            use_dask=use_dask,
        )

    @pytest.mark.slow
    @pytest.mark.parametrize("use_dask", [True, False])
    def test_bootstrap_tn90p_anchored_freq(self, tasmin_series, use_dask):
        self.bootstrap_testor(
            tasmin_series,
            98,
            lambda x, y, z: tn90p(x, y, freq="A-JUL", bootstrap=z),
            use_dask=use_dask,
        )

    @pytest.mark.slow
    @pytest.mark.parametrize("use_dask", [True, False])
    def test_bootstrap_tx90p_anchored_freq(self, tasmax_series, use_dask):
        self.bootstrap_testor(
            tasmax_series,
            98,
            lambda x, y, z: tx90p(x, y, freq="Q-APR", bootstrap=z),
            use_dask=use_dask,
        )

    @pytest.mark.slow
    @pytest.mark.parametrize("use_dask", [True, False])
    def test_bootstrap_tn10p(self, tasmin_series, use_dask):
        self.bootstrap_testor(
            tasmin_series,
            2,
            lambda x, y, z: tn10p(x, y, freq="MS", bootstrap=z),
            use_dask=use_dask,
        )

    @pytest.mark.slow
    @pytest.mark.parametrize("use_dask", [True, False])
    def test_bootstrap_tx10p(self, tasmax_series, use_dask):
        self.bootstrap_testor(
            tasmax_series,
            2,
            lambda x, y, z: tx10p(x, y, freq="YS", bootstrap=z),
            use_dask=use_dask,
        )

    @pytest.mark.slow
    @pytest.mark.parametrize("use_dask", [True, False])
    def test_bootstrap_tg10p(self, tas_series, use_dask):
        self.bootstrap_testor(
            tas_series,
            2,
            lambda x, y, z: tg10p(x, y, freq="MS", bootstrap=z),
            use_dask=use_dask,
        )

    @pytest.mark.slow
    @pytest.mark.parametrize("use_dask", [True, False])
    def test_bootstrap_warm_spell_duration_index(self, tasmax_series, use_dask):
        self.bootstrap_testor(
            tasmax_series,
            98,
            lambda x, y, z: warm_spell_duration_index(
                x, y, window=6, freq="MS", bootstrap=z
            ),
            use_dask=use_dask,
        )

    @pytest.mark.slow
    @pytest.mark.parametrize("use_dask", [True, False])
    def test_bootstrap_cold_spell_duration_index(self, tasmin_series, use_dask):
        self.bootstrap_testor(
            tasmin_series,
            2,
            lambda x, y, z: cold_spell_duration_index(
                x, y, window=6, freq="MS", bootstrap=z
            ),
            use_dask=use_dask,
        )

    @pytest.mark.slow
    @pytest.mark.parametrize("use_dask", [True, False])
    def test_bootstrap_days_over_precip_thresh(self, pr_series, use_dask):
        self.bootstrap_testor(
            pr_series,
            99,
            lambda x, y, z: days_over_precip_thresh(x, y, freq="MS", bootstrap=z),
            positive_values=True,
            use_dask=use_dask,
        )

    @pytest.mark.slow
    @pytest.mark.parametrize("use_dask", [True, False])
    def test_bootstrap_fraction_over_precip_thresh(self, pr_series, use_dask):
        self.bootstrap_testor(
            pr_series,
            98,
            lambda x, y, z: fraction_over_precip_thresh(x, y, freq="MS", bootstrap=z),
            positive_values=True,
            use_dask=use_dask,
        )

    def test_bootstrap_cftime(self, bootstrap_series):
        self.bootstrap_testor(
            lambda values, start: bootstrap_series(values, start, cf_time=True),
            98,
            lambda x, y, z: fraction_over_precip_thresh(x, y, freq="MS", bootstrap=z),
            positive_values=True,
            use_dask=False,
        )

    @pytest.mark.slow
    def test_bootstrap_fraction_over_precip_error_no_doy(self, pr_series):
        with pytest.raises(KeyError):
            # no "dayofyear" coords on per
            fraction_over_precip_thresh(
                pr_series([1, 2]), pr_series([1, 2]), bootstrap=True
            )

    def test_bootstrap_days_over_precip_thresh_error_no_doy(self, pr_series):
        with pytest.raises(KeyError):
            # no "dayofyear" coords on per
            days_over_precip_thresh(
                pr_series([1, 2]), pr_series([1, 2]), bootstrap=True
            )

    def test_bootstrap_no_doy(self, tas_series):
        # no "dayofyear" coords on per
        with pytest.raises(KeyError):
            tg10p(tas_series([42]), tas_series([42]), freq="MS", bootstrap=True)

    def test_bootstrap_full_overlap(self, tas_series):
        # bootstrap is unnecessary when in base and out of base fully overlap
        # -- GIVEN
        tas = tas_series(self.ar1(alpha=0.8, n=int(4 * 365.25)), start="2000-01-01")
        per = percentile_doy(tas, per=90)
        # -- THEN
        with pytest.raises(KeyError):
            # -- WHEN
            tg10p(tas, per, freq="YS", bootstrap=True)

    @pytest.mark.slow
    def test_bootstrap_no_overlap(self, tas_series):
        # bootstrap is unnecessary when in base and out of base fully overlap
        # -- GIVEN
        tas = tas_series(self.ar1(alpha=0.8, n=int(4 * 365.25)), start="2000-01-01")
        tas_in_base = tas.sel(time=slice("2000-01-01", "2001-12-31"))
        tas_out_base = tas.sel(time=slice("2002-01-01", "2001-12-31"))
        per = percentile_doy(tas_in_base, per=90)
        # -- THEN
        with pytest.raises(KeyError):
            # -- WHEN
            tg10p(tas_out_base, per, freq="MS", bootstrap=True)

    @pytest.mark.slow
    def test_multi_per(self, open_dataset):
        tas = open_dataset("ERA5/daily_surface_cancities_1990-1993.nc").tas
        t90 = percentile_doy(
            tas.sel(time=slice("1990-01-01", "1991-12-31")), window=5, per=[90, 91]
        )
        res = tg90p(tas=tas, tas_per=t90, freq="YS", bootstrap=True)
        np.testing.assert_array_equal([90, 91], res.percentiles)

    @pytest.mark.slow
    def test_doctest_ndims(self, open_dataset):
        """Replicates doctest to facilitate debugging."""
        tas = open_dataset("ERA5/daily_surface_cancities_1990-1993.nc").tas
        t90 = percentile_doy(
            tas.sel(time=slice("1990-01-01", "1991-12-31")), window=5, per=90
        )
        tg90p(tas=tas, tas_per=t90.isel(percentiles=0), freq="YS", bootstrap=True)
        tg90p(tas=tas, tas_per=t90, freq="YS", bootstrap=True)

    def bootstrap_testor(
        self,
        series,
        per: int,
        index_fun: Callable[[DataArray, DataArray, bool], DataArray],
        positive_values=False,
        use_dask=False,
    ):
        # -- GIVEN
        arr = self.ar1(alpha=0.8, n=int(4 * 365.25), positive_values=positive_values)
        climate_var = series(arr, start="2000-01-01")
        if use_dask:
            climate_var = climate_var.chunk(dict(time=50))
        in_base_slice = slice("2000-01-01", "2001-12-31")
        out_base_slice = slice("2002-01-01", "2003-12-31")
        per = percentile_doy(climate_var.sel(time=in_base_slice), per=per)
        # -- WHEN
        no_bootstrap = index_fun(climate_var, per, False)
        no_bs_in_base = no_bootstrap.sel(time=in_base_slice)
        no_bs_out_base = no_bootstrap.sel(time=out_base_slice)
        bootstrap = index_fun(climate_var, per, True)
        bootstrapped_in_base = bootstrap.sel(time=in_base_slice)
        bs_out_base = bootstrap.sel(time=out_base_slice)
        # -- THEN
        # Bootstrapping should increase the computed index values within the overlapping
        # period. However, this will not work on unrealistic values such as a constant
        # temperature.
        # Beside, bootstrapping is particularly effective on extreme percentiles, but
        # the closer the target percentile is to the median the less bootstrapping is
        # necessary.
        # Following assertions may even fail if chosen percentile is close to 50.
        assert np.count_nonzero(
            bootstrapped_in_base > no_bs_in_base
        ) > np.count_nonzero(bootstrapped_in_base < no_bs_in_base)
        # bootstrapping should leave the out of base unchanged,
        # but precision above 15th decimal might differ.
        np.testing.assert_array_almost_equal(no_bs_out_base, bs_out_base, 15)

    def ar1(self, alpha, n, positive_values=False):
        """Return random AR1 DataArray."""
        # White noise
        wn = np.random.randn(n - 1) * np.sqrt(1 - alpha**2)
        # Autoregressive series of order 1
        out = np.empty(n)
        out[0] = np.random.randn()
        for i, w in enumerate(wn):
            if positive_values:
                out[i + 1] = np.abs(alpha * out[i] + w)
            else:
                out[i + 1] = alpha * out[i] + w
        return out

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Tests for `xclim` package.
#
# We want to tests multiple things here:
#  - that data results are correct
#  - that metadata is correct and complete
#  - that missing data are handled appropriately
#  - that various calendar formats and supported
#  - that non-valid input frequencies or holes in the time series are detected
#
#
# For correctness, I think it would be useful to use a small dataset and run the original ICCLIM indicators on it,
# saving the results in a reference netcdf dataset. We could then compare the hailstorm output to this reference as
# a first line of defense.
import glob
import os

import numpy as np
import pandas as pd
import xarray as xr

from xclim import ensembles

TESTS_HOME = os.path.abspath(os.path.dirname(__file__))
TESTS_DATA = os.path.join(TESTS_HOME, "testdata")


class TestEnsembleStats:
    nc_files_simple = glob.glob(
        os.path.join(TESTS_DATA, "EnsembleStats", "*1950-2100*.nc")
    )
    nc_files = glob.glob(os.path.join(TESTS_DATA, "EnsembleStats", "*.nc"))

    def test_checktimes(self):
        time_flag, time_all = ensembles._ens_checktimes(self.nc_files)
        assert time_flag
        assert pd.DatetimeIndex(time_all).min() == pd.Timestamp("1950-01-01 00:00:00")
        assert pd.DatetimeIndex(time_all).max() == pd.Timestamp("2100-01-01 00:00:00")

        # verify short time-series file
        time_flag, time_all1 = ensembles._ens_checktimes(
            [i for i in self.nc_files if "1970-2050" in i]
        )
        assert time_flag
        assert pd.DatetimeIndex(time_all1).min() > pd.DatetimeIndex(time_all).min()
        assert pd.DatetimeIndex(time_all1).max() < pd.DatetimeIndex(time_all).max()

        # no time
        ds = xr.open_dataset(self.nc_files[0])
        ds = ds.groupby(ds.time.dt.month).mean("time", keep_attrs=True)
        time_flag, time_all = ensembles._ens_checktimes([ds])
        assert not time_flag
        assert time_all is None

    def test_create_ensemble(self):
        ens = ensembles.create_ensemble(self.nc_files_simple)
        assert len(ens.realization) == len(self.nc_files_simple)

        # create again using xr.Dataset objects
        ds_all = []
        for n in self.nc_files_simple:
            ds = xr.open_dataset(n, decode_times=False)
            ds["time"] = xr.decode_cf(ds).time
            ds_all.append(ds)

        ens1 = ensembles.create_ensemble(ds_all)
        coords = list(ens1.coords)
        coords.extend(list(ens1.data_vars))
        for c in coords:
            np.testing.assert_array_equal(ens[c], ens1[c])

    def test_no_time(self):

        # create again using xr.Dataset objects
        ds_all = []
        for n in self.nc_files_simple:
            ds = xr.open_dataset(n, decode_times=False)
            ds["time"] = xr.decode_cf(ds).time
            ds_all.append(ds.groupby(ds.time.dt.month).mean("time", keep_attrs=True))

        ens = ensembles.create_ensemble(ds_all)
        assert len(ens.realization) == len(self.nc_files_simple)

    def test_create_unequal_times(self):
        ens = ensembles.create_ensemble(self.nc_files)
        assert len(ens.realization) == len(self.nc_files)
        assert ens.time.dt.year.min() == 1950
        assert ens.time.dt.year.max() == 2100

        ii = [i for i, s in enumerate(self.nc_files) if "1970-2050" in s]
        # assert padded with nans
        assert np.all(
            np.isnan(ens.tg_mean.isel(realization=ii).sel(time=ens.time.dt.year < 1970))
        )
        assert np.all(
            np.isnan(ens.tg_mean.isel(realization=ii).sel(time=ens.time.dt.year > 2050))
        )

        ens_mean = ens.tg_mean.mean(dim=["realization", "lon", "lat"], skipna=False)
        assert ens_mean.where(~np.isnan(ens_mean), drop=True).time.dt.year.min() == 1970
        assert ens_mean.where(~np.isnan(ens_mean), drop=True).time.dt.year.max() == 2050

    def test_calc_perc(self):
        ens = ensembles.create_ensemble(self.nc_files_simple)
        out1 = ensembles.ensemble_percentiles(ens)
        np.testing.assert_array_equal(
            np.percentile(ens["tg_mean"][:, 0, 5, 5], 10), out1["tg_mean_p10"][0, 5, 5]
        )
        np.testing.assert_array_equal(
            np.percentile(ens["tg_mean"][:, 0, 5, 5], 50), out1["tg_mean_p50"][0, 5, 5]
        )
        np.testing.assert_array_equal(
            np.percentile(ens["tg_mean"][:, 0, 5, 5], 90), out1["tg_mean_p90"][0, 5, 5]
        )
        assert np.all(out1["tg_mean_p90"] > out1["tg_mean_p50"])
        assert np.all(out1["tg_mean_p50"] > out1["tg_mean_p10"])
        out1 = ensembles.ensemble_percentiles(ens, values=(25, 75))
        assert np.all(out1["tg_mean_p75"] > out1["tg_mean_p25"])

    def test_calc_perc_blocks(self):
        ens = ensembles.create_ensemble(self.nc_files_simple)
        out1 = ensembles.ensemble_percentiles(ens)
        out2 = ensembles.ensemble_percentiles(ens, values=(10, 50, 90), time_block=10)
        np.testing.assert_array_equal(out1["tg_mean_p10"], out2["tg_mean_p10"])
        np.testing.assert_array_equal(out1["tg_mean_p50"], out2["tg_mean_p50"])
        np.testing.assert_array_equal(out1["tg_mean_p90"], out2["tg_mean_p90"])

    def test_calc_perc_nans(self):
        ens = ensembles.create_ensemble(self.nc_files_simple).load()

        ens.tg_mean[2, 0, 5, 5] = np.nan
        ens.tg_mean[2, 7, 5, 5] = np.nan
        out1 = ensembles.ensemble_percentiles(ens)
        np.testing.assert_array_equal(
            np.percentile(ens["tg_mean"][:, 0, 5, 5], 10), np.nan
        )
        np.testing.assert_array_equal(
            np.percentile(ens["tg_mean"][:, 7, 5, 5], 10), np.nan
        )
        np.testing.assert_array_equal(
            np.nanpercentile(ens["tg_mean"][:, 0, 5, 5], 10),
            out1["tg_mean_p10"][0, 5, 5],
        )
        np.testing.assert_array_equal(
            np.nanpercentile(ens["tg_mean"][:, 7, 5, 5], 10),
            out1["tg_mean_p10"][7, 5, 5],
        )
        assert np.all(out1["tg_mean_p90"] > out1["tg_mean_p50"])
        assert np.all(out1["tg_mean_p50"] > out1["tg_mean_p10"])

    def test_calc_mean_std_min_max(self):
        ens = ensembles.create_ensemble(self.nc_files_simple)
        out1 = ensembles.ensemble_mean_std_max_min(ens)
        np.testing.assert_array_equal(
            ens["tg_mean"][:, 0, 5, 5].mean(dim="realization"),
            out1.tg_mean_mean[0, 5, 5],
        )
        np.testing.assert_array_equal(
            ens["tg_mean"][:, 0, 5, 5].std(dim="realization"),
            out1.tg_mean_stdev[0, 5, 5],
        )
        np.testing.assert_array_equal(
            ens["tg_mean"][:, 0, 5, 5].max(dim="realization"), out1.tg_mean_max[0, 5, 5]
        )
        np.testing.assert_array_equal(
            ens["tg_mean"][:, 0, 5, 5].min(dim="realization"), out1.tg_mean_min[0, 5, 5]
        )


class TestEnsembleReduction:
    nc_file = os.path.join(TESTS_DATA, "EnsembleReduce", "TestEnsReduceCriteria.nc")

    def test_kmeans_rsqcutoff(self):
        ds = xr.open_dataset(self.nc_file)
        for n in np.arange(0, 20):
            print(n)
            # use random state variable to ensure consistent clustering in tests:
            [ids, cluster] = ensembles.kmeans_reduce_ensemble(
                ds.data, method={"rsq_cutoff": 0.9}, random_state=42
            )

            assert ids == [0, 1, 3, 4, 6, 7, 8, 10, 11, 15, 18, 20, 22]
            assert len(ids) == 13

    def test_kmeans_rsqopt(self):
        ds = xr.open_dataset(self.nc_file)
        [ids, cluster] = ensembles.kmeans_reduce_ensemble(
            ds.data, method={"rsq_optimize": None}
        )
        assert ids == [4, 5, 7, 10, 11, 12, 13]
        assert len(ids) == 7

    def test_kmeans_nclust(self):
        ds = xr.open_dataset(self.nc_file)
        [ids, cluster] = ensembles.kmeans_reduce_ensemble(
            ds.data, method={"n_clusters": 4}
        )
        assert ids == [4, 5, 7, 10, 11, 12, 13]
        assert len(ids) == 7

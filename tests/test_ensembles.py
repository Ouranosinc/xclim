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
import os
import sys
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from xclim import ensembles
from xclim.testing import open_dataset


class TestEnsembleStats:
    nc_files = [
        "BCCAQv2+ANUSPLIN300_ACCESS1-0_historical+rcp45_r1i1p1_1950-2100_tg_mean_YS.nc",
        "BCCAQv2+ANUSPLIN300_BNU-ESM_historical+rcp45_r1i1p1_1950-2100_tg_mean_YS.nc",
        "BCCAQv2+ANUSPLIN300_CCSM4_historical+rcp45_r1i1p1_1950-2100_tg_mean_YS.nc",
        "BCCAQv2+ANUSPLIN300_CCSM4_historical+rcp45_r2i1p1_1950-2100_tg_mean_YS.nc",
    ]

    nc_file_extra = (
        "BCCAQv2+ANUSPLIN300_CNRM-CM5_historical+rcp45_r1i1p1_1970-2050_tg_mean_YS.nc"
    )

    nc_datasets_simple = [
        open_dataset(os.path.join("EnsembleStats", f)) for f in nc_files
    ]
    nc_datasets = deepcopy(nc_datasets_simple)
    nc_datasets.append(open_dataset(os.path.join("EnsembleStats", nc_file_extra)))

    def test_create_ensemble(self):
        ens = ensembles.create_ensemble(self.nc_datasets_simple)
        assert len(ens.realization) == len(self.nc_datasets_simple)
        assert len(ens.time) == 151

        # create again using xr.Dataset objects
        ds_all = []
        for n in self.nc_files:
            ds = open_dataset(os.path.join("EnsembleStats", n), decode_times=False)
            ds["time"] = xr.decode_cf(ds).time
            ds_all.append(ds)

        ens1 = ensembles.create_ensemble(ds_all)
        coords = list(ens1.coords)
        coords.extend(list(ens1.data_vars))
        for c in coords:
            np.testing.assert_array_equal(ens[c], ens1[c])

        for i in np.arange(0, len(ens1.realization)):
            np.testing.assert_array_equal(
                ens1.isel(realization=i).tg_mean.values, ds_all[i].tg_mean.values
            )

    def test_no_time(self):
        # create again using xr.Dataset objects
        ds_all = []
        for n in self.nc_files:
            ds = open_dataset(os.path.join("EnsembleStats", n), decode_times=False)
            ds["time"] = xr.decode_cf(ds).time
            ds_all.append(ds.groupby(ds.time.dt.month).mean("time", keep_attrs=True))

        ens = ensembles.create_ensemble(ds_all)
        assert len(ens.realization) == len(self.nc_files)

    def test_create_unequal_times(self):
        ens = ensembles.create_ensemble(self.nc_datasets)
        assert len(ens.realization) == len(self.nc_datasets)
        assert ens.time.dt.year.min() == 1950
        assert ens.time.dt.year.max() == 2100
        assert len(ens.time) == 151

        ii = [i for i, s in enumerate(self.nc_datasets) if "1970-2050" in s]
        # assert padded with nans
        assert np.all(
            np.isnan(ens.tg_mean.isel(realization=ii).sel(time=ens.time.dt.year < 1970))
        )
        assert np.all(
            np.isnan(ens.tg_mean.isel(realization=ii).sel(time=ens.time.dt.year > 2050))
        )

        ens_mean = ens.tg_mean.mean(dim=["realization", "lon", "lat"], skipna=False)
        assert (
            ens_mean.where(~(np.isnan(ens_mean)), drop=True).time.dt.year.min() == 1970
        )
        assert (
            ens_mean.where(~(np.isnan(ens_mean)), drop=True).time.dt.year.max() == 2050
        )

    @pytest.mark.parametrize(
        "timegen,calkw",
        [(xr.cftime_range, {"calendar": "360_day"}), (pd.date_range, {})],
    )
    def test_create_unaligned_times(self, timegen, calkw):
        t1 = timegen("2000-01-01", periods=24, freq="M", **calkw)
        t2 = timegen("2000-01-01", periods=24, freq="MS", **calkw)

        d1 = xr.DataArray(
            np.arange(24), dims=("time",), coords={"time": t1}, name="tas"
        )
        d2 = xr.DataArray(
            np.arange(24), dims=("time",), coords={"time": t2}, name="tas"
        )

        if t1.dtype != "O":
            ens = ensembles.create_ensemble((d1, d2))
            assert ens.time.size == 48
            np.testing.assert_equal(ens.isel(time=0), [np.nan, 0])

        ens = ensembles.create_ensemble((d1, d2), resample_freq="MS")
        assert ens.time.size == 24
        np.testing.assert_equal(ens.isel(time=0), [0, 0])

    @pytest.mark.parametrize("transpose", [False, True])
    def test_calc_perc(self, transpose):
        ens = ensembles.create_ensemble(self.nc_datasets_simple)
        if transpose:
            ens = ens.transpose()

        out1 = ensembles.ensemble_percentiles(ens, split=True)
        np.testing.assert_array_equal(
            np.percentile(ens["tg_mean"].isel(time=0, lon=5, lat=5), 10),
            out1["tg_mean_p10"].isel(time=0, lon=5, lat=5),
        )
        np.testing.assert_array_equal(
            np.percentile(ens["tg_mean"].isel(time=0, lon=5, lat=5), 50),
            out1["tg_mean_p50"].isel(time=0, lon=5, lat=5),
        )
        np.testing.assert_array_equal(
            np.percentile(ens["tg_mean"].isel(time=0, lon=5, lat=5), 90),
            out1["tg_mean_p90"].isel(time=0, lon=5, lat=5),
        )
        assert np.all(out1["tg_mean_p90"] > out1["tg_mean_p50"])
        assert np.all(out1["tg_mean_p50"] > out1["tg_mean_p10"])

        out2 = ensembles.ensemble_percentiles(ens, values=(25, 75))
        assert np.all(out2["tg_mean_p75"] > out2["tg_mean_p25"])
        assert "Computation of the percentiles on" in out1.attrs["xclim_history"]

        out3 = ensembles.ensemble_percentiles(ens, split=False)
        xr.testing.assert_equal(
            out1["tg_mean_p10"], out3.tg_mean.sel(percentiles=10, drop=True)
        )

    @pytest.mark.parametrize("keep_chunk_size", [False, True, None])
    def test_calc_perc_dask(self, keep_chunk_size):
        ens = ensembles.create_ensemble(self.nc_datasets_simple)
        out2 = ensembles.ensemble_percentiles(
            ens.chunk({"time": 2}), keep_chunk_size=keep_chunk_size, split=False
        )
        out1 = ensembles.ensemble_percentiles(ens.load(), split=False)
        np.testing.assert_array_equal(out1["tg_mean"], out2["tg_mean"])

    def test_calc_perc_nans(self):
        ens = ensembles.create_ensemble(self.nc_datasets_simple).load()

        ens.tg_mean[2, 0, 5, 5] = np.nan
        ens.tg_mean[2, 7, 5, 5] = np.nan
        out1 = ensembles.ensemble_percentiles(ens, split=True)
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
        ens = ensembles.create_ensemble(self.nc_datasets_simple)
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
        assert "Computation of statistics on" in out1.attrs["xclim_history"]


class TestEnsembleReduction:
    nc_file = os.path.join("EnsembleReduce", "TestEnsReduceCriteria.nc")

    def test_kmeans_rsqcutoff(self):
        pytest.importorskip("sklearn", minversion="0.22")
        ds = open_dataset(self.nc_file)

        # use random state variable to ensure consistent clustering in tests:
        [ids, cluster, fig_data] = ensembles.kmeans_reduce_ensemble(
            data=ds.data, method={"rsq_cutoff": 0.9}, random_state=42, make_graph=False
        )

        assert ids == [0, 1, 3, 4, 6, 7, 8, 9, 10, 11, 12, 15, 20, 22]
        assert len(ids) == 14

        # Test max cluster option
        [ids, cluster, fig_data] = ensembles.kmeans_reduce_ensemble(
            data=ds.data,
            method={"rsq_cutoff": 0.9},
            random_state=42,
            make_graph=False,
            max_clusters=10,
        )
        assert ids == [0, 1, 3, 4, 6, 7, 10, 11, 18, 20]
        assert len(ids) == 10

    def test_kmeans_rsqopt(self):
        pytest.importorskip("sklearn", minversion="0.22")
        ds = open_dataset(self.nc_file)
        [ids, cluster, fig_data] = ensembles.kmeans_reduce_ensemble(
            data=ds.data,
            method={"rsq_optimize": None},
            random_state=42,
            make_graph=False,
        )
        assert ids == [4, 5, 7, 10, 11, 12, 13]
        assert len(ids) == 7

    def test_kmeans_nclust(self):
        ds = open_dataset(self.nc_file)

        [ids, cluster, fig_data] = ensembles.kmeans_reduce_ensemble(
            data=ds.data, method={"n_clusters": 4}, random_state=42, make_graph=False
        )
        assert ids == [4, 7, 10, 23]
        assert len(ids) == 4

        [ids, cluster, fig_data] = ensembles.kmeans_reduce_ensemble(
            ds.data, method={"n_clusters": 9}, random_state=42, make_graph=False
        )
        assert ids == [0, 3, 4, 6, 7, 10, 11, 12, 13]
        assert len(ids) == 9

    def test_kmeans_sampleweights(self):
        ds = open_dataset(self.nc_file)
        # Test sample weights
        sample_weights = np.ones(ds.data.shape[0])
        # boost weights for some sims
        sample_weights[[0, 20]] = 15

        [ids, cluster, fig_data] = ensembles.kmeans_reduce_ensemble(
            data=ds.data,
            method={"rsq_cutoff": 0.9},
            random_state=42,
            make_graph=False,
            sample_weights=sample_weights,
        )
        assert ids == [0, 1, 3, 4, 5, 6, 7, 10, 11, 18, 20]
        assert len(ids) == 11

        # RSQ optimize
        sample_weights = np.ones(ds.data.shape[0])
        # try zero weights
        sample_weights[[6, 18, 22]] = 0

        [ids, cluster, fig_data] = ensembles.kmeans_reduce_ensemble(
            data=ds.data,
            method={"rsq_optimize": None},
            random_state=0,
            make_graph=False,
            sample_weights=sample_weights,
        )

        assert ids == [4, 5, 7, 10, 11, 12, 13]
        assert len(ids) == 7

        sample_weights = np.ones(ds.data.shape[0])
        # try zero weights
        sample_weights[[6, 18, 22]] = 0
        [ids, cluster, fig_data] = ensembles.kmeans_reduce_ensemble(
            data=ds.data,
            method={"rsq_optimize": None},
            random_state=42,
            make_graph=False,
            sample_weights=sample_weights,
        )
        assert ids == [4, 5, 7, 10, 12, 13]
        assert len(ids) == 6

    def test_kmeans_variweights(self):
        pytest.importorskip("sklearn", minversion="0.22")
        ds = open_dataset(self.nc_file)
        # Test sample weights
        var_weights = np.ones(ds.data.shape[1])
        # reduce weights for some variables
        var_weights[3:] = 0.25

        [ids, cluster, fig_data] = ensembles.kmeans_reduce_ensemble(
            data=ds.data,
            method={"rsq_cutoff": 0.9},
            random_state=42,
            make_graph=False,
            variable_weights=var_weights,
        )
        assert ids == [1, 3, 8, 10, 13, 14, 16, 19, 20]
        assert len(ids) == 9

        # using RSQ optimize
        [ids, cluster, fig_data] = ensembles.kmeans_reduce_ensemble(
            data=ds.data,
            method={"rsq_optimize": None},
            random_state=42,
            make_graph=False,
            variable_weights=var_weights,
        )

        assert ids == [2, 4, 8, 13, 14, 22]
        assert len(ids) == 6

        # try zero weights
        var_weights = np.ones(ds.data.shape[1])
        var_weights[[1, 4]] = 0

        [ids, cluster, fig_data] = ensembles.kmeans_reduce_ensemble(
            data=ds.data,
            method={"rsq_optimize": None},
            random_state=42,
            make_graph=False,
            variable_weights=var_weights,
        )
        # Results here may change according to sklearn version, hence the *isin* intead of ==
        assert all(np.isin([4, 12, 13, 16], ids))
        assert len(ids) == 5

    def test_kmeans_modelweights(self):
        ds = open_dataset(self.nc_file)
        # Test sample weights
        model_weights = np.ones(ds.data.shape[0])
        model_weights[[4, 7, 10, 23]] = 0

        # set to zero for some models that are selected in n_cluster test - these models should not be selected now
        [ids, cluster, fig_data] = ensembles.kmeans_reduce_ensemble(
            data=ds.data,
            method={"n_clusters": 4},
            random_state=42,
            make_graph=False,
            model_weights=model_weights,
        )

        for i in np.where(model_weights == 0)[0]:
            # as long as the cluster has more than one member the models w/ weight==0 should not be present
            if np.sum(cluster == cluster[i]) > 1:
                assert i not in ids

        model_weights = np.ones(ds.data.shape[0])
        model_weights[[0, 3, 4, 6, 7, 10, 11, 12, 13]] = 0
        [ids, cluster, fig_data] = ensembles.kmeans_reduce_ensemble(
            data=ds.data,
            method={"n_clusters": 9},
            random_state=42,
            make_graph=False,
            model_weights=model_weights,
        )
        for i in np.where(model_weights == 0)[0]:
            # as long as the cluster has more than one member the models w/ weight==0 should not be present
            if np.sum(cluster == cluster[i]) > 1:
                assert i not in ids

    @pytest.mark.skipif(
        "matplotlib.pyplot" not in sys.modules, reason="matplotlib.pyplot is required"
    )
    def test_kmeans_rsqcutoff_with_graphs(self):
        pytest.importorskip("sklearn", minversion="0.22")
        ds = open_dataset(self.nc_file)

        # use random state variable to ensure consistent clustering in tests:
        [ids, cluster, fig_data] = ensembles.kmeans_reduce_ensemble(
            data=ds.data, method={"rsq_cutoff": 0.9}, random_state=42, make_graph=True
        )

        assert ids == [0, 1, 3, 4, 6, 7, 8, 9, 10, 11, 12, 15, 20, 22]
        assert len(ids) == 14

        # Test max cluster option
        [ids, cluster, fig_data] = ensembles.kmeans_reduce_ensemble(
            data=ds.data,
            method={"rsq_cutoff": 0.9},
            random_state=42,
            make_graph=True,
            max_clusters=10,
        )
        assert ids == [0, 1, 3, 4, 6, 7, 10, 11, 18, 20]
        assert len(ids) == 10

    @pytest.mark.parametrize(
        "crit,num_select,expected",
        [
            ([0, 1], 5, [16, 19, 20, 15, 9]),
            ([0, 1], 4, [16, 19, 20, 15]),
            (np.arange(6), 8, [23, 19, 10, 14, 11, 8, 20, 3]),
            ([4, 5], 4, [15, 10, 14, 1]),
            ([4, 5], 1, [15]),
        ],
    )
    def test_kkz_simple(self, crit, num_select, expected):
        ens = open_dataset(self.nc_file)
        data = ens.data.isel(criteria=crit)

        selected = ensembles.kkz_reduce_ensemble(data, num_select)
        assert selected == expected

    def test_kkz_standardize(self):
        ens = open_dataset(self.nc_file)
        data = ens.data.isel(criteria=[1, 3, 5])

        sel_std = ensembles.kkz_reduce_ensemble(data, 4, standardize=True)
        sel_no = ensembles.kkz_reduce_ensemble(data, 4, standardize=False)
        assert sel_std == [23, 10, 19, 14]
        assert sel_no == [23, 1, 14, 10]

    def test_kkz_change_metric(self):
        # This test uses stupid values but is meant to test is kwargs are passed and if dist_method is used.
        ens = open_dataset(self.nc_file)
        data = ens.data.isel(criteria=[1, 3, 5])

        sel_euc = ensembles.kkz_reduce_ensemble(data, 4, dist_method="euclidean")
        sel_mah = ensembles.kkz_reduce_ensemble(
            data, 4, dist_method="mahalanobis", VI=np.arange(24)
        )
        assert sel_euc == [23, 10, 19, 14]
        assert sel_mah == [5, 3, 4, 0]

    def test_standardize_seuclidean(self):
        # This test the odd choice of standardizing data for a standardized distance metric
        ens = open_dataset(self.nc_file)
        data = ens.data
        for n in np.arange(1, len(data)):
            sel1 = ensembles.kkz_reduce_ensemble(
                data, n, dist_method="seuclidean", standardize=True
            )
            sel2 = ensembles.kkz_reduce_ensemble(
                data, n, dist_method="seuclidean", standardize=False
            )
            sel3 = ensembles.kkz_reduce_ensemble(
                data, n, dist_method="euclidean", standardize=True
            )
            assert sel1 == sel2
            assert sel1 == sel3

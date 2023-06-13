#!/usr/bin/env python
# Tests for `xclim` package.
#
# We want to test multiple things here:
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
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from pkg_resources import parse_version
from scipy import __version__ as __scipy_version__
from scipy.stats.mstats import mquantiles

from xclim import ensembles
from xclim.indices.stats import get_dist
from xclim.testing.helpers import TESTDATA_BRANCH


class TestEnsembleStats:
    def test_create_ensemble(
        self, open_dataset, ensemble_dataset_objects, threadsafe_data_dir
    ):
        ds_all = []
        for n in ensemble_dataset_objects["nc_files_simple"]:
            ds = open_dataset(n, decode_times=False)
            ds["time"] = xr.decode_cf(ds).time
            ds_all.append(ds)
        ens = ensembles.create_ensemble(ds_all)

        assert len(ens.realization) == len(ensemble_dataset_objects["nc_files_simple"])
        assert len(ens.time) == 151
        for i in np.arange(0, len(ens.realization)):
            np.testing.assert_array_equal(
                ens.isel(realization=i).tg_mean.values, ds_all[i].tg_mean.values
            )

        reals = [
            "_".join(Path(f).name.split("_")[1:4:2])
            for f in ensemble_dataset_objects["nc_files_simple"]
        ]
        ens1 = ensembles.create_ensemble(ds_all, realizations=reals)

        # Kinda a hack? Alternative is to open and rewrite in a temp folder.
        files = [
            threadsafe_data_dir / TESTDATA_BRANCH / "EnsembleStats" / Path(f).name
            for f in ensemble_dataset_objects["nc_files_simple"]
        ]
        ens2 = ensembles.create_ensemble(dict(zip(reals, files)))
        xr.testing.assert_identical(ens1, ens2)

    def test_no_time(self, tmp_path, ensemble_dataset_objects, open_dataset):
        # create again using xr.Dataset objects
        f1 = Path(tmp_path / "notime")
        f1.mkdir()
        ds_all = []
        for n in ensemble_dataset_objects["nc_files"]:
            ds = open_dataset(n, decode_times=False)
            ds["time"] = xr.decode_cf(ds).time
            ds_all.append(ds.groupby(ds.time.dt.month).mean("time", keep_attrs=True))
            ds.groupby(ds.time.dt.month).mean("time", keep_attrs=True).to_netcdf(
                f1.joinpath(Path(n).name)
            )
        ens = ensembles.create_ensemble(ds_all)

        assert len(ens.realization) == len(ensemble_dataset_objects["nc_files"])

        in_ncs = list(Path(f1).glob("*.nc"))
        ens = ensembles.create_ensemble(in_ncs)
        assert len(ens.realization) == len(ensemble_dataset_objects["nc_files"])

    def test_create_unequal_times(self, ensemble_dataset_objects, open_dataset):
        ds_all = []
        for n in ensemble_dataset_objects["nc_files"]:
            ds = open_dataset(n)
            ds_all.append(ds)
        ens = ensembles.create_ensemble(ds_all)

        assert len(ens.realization) == len(ds_all)
        assert ens.time.dt.year.min() == 1950
        assert ens.time.dt.year.max() == 2100
        assert len(ens.time) == 151

        ii = [
            i
            for i, s in enumerate(ensemble_dataset_objects["nc_files"])
            if "1970-2050" in s
        ]
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
    def test_calc_perc(self, transpose, ensemble_dataset_objects, open_dataset):
        ds_all = []
        for n in ensemble_dataset_objects["nc_files_simple"]:
            ds = open_dataset(n)
            ds_all.append(ds)
        ens = ensembles.create_ensemble(ds_all)

        if transpose:
            ens = ens.transpose()

        out1 = ensembles.ensemble_percentiles(ens, split=True)
        np.testing.assert_array_almost_equal(
            mquantiles(
                ens["tg_mean"].isel(time=0, lon=5, lat=5), 0.1, alphap=1, betap=1
            ),
            out1["tg_mean_p10"].isel(time=0, lon=5, lat=5),
        )
        np.testing.assert_array_almost_equal(
            mquantiles(
                ens["tg_mean"].isel(time=0, lon=5, lat=5), alphap=1, betap=1, prob=0.50
            ),
            out1["tg_mean_p50"].isel(time=0, lon=5, lat=5),
        )
        np.testing.assert_array_almost_equal(
            mquantiles(
                ens["tg_mean"].isel(time=0, lon=5, lat=5), alphap=1, betap=1, prob=0.90
            ),
            out1["tg_mean_p90"].isel(time=0, lon=5, lat=5),
        )
        assert np.all(out1["tg_mean_p90"] > out1["tg_mean_p50"])
        assert np.all(out1["tg_mean_p50"] > out1["tg_mean_p10"])

        out2 = ensembles.ensemble_percentiles(ens, values=(25, 75))
        assert np.all(out2["tg_mean_p75"] > out2["tg_mean_p25"])
        assert "Computation of the percentiles on" in out1.attrs["history"]

        out3 = ensembles.ensemble_percentiles(ens, split=False)
        xr.testing.assert_equal(
            out1["tg_mean_p10"], out3.tg_mean.sel(percentiles=10, drop=True)
        )

        weights = xr.DataArray(
            [1, 0.1, 3.5, 5], coords={"realization": ens.realization}
        )
        out4 = ensembles.ensemble_percentiles(ens, weights=weights)
        np.testing.assert_array_almost_equal(
            ens["tg_mean"].isel(time=0, lon=5, lat=5).weighted(weights).quantile(0.5),
            out4["tg_mean_p50"].isel(time=0, lon=5, lat=5),
        )
        np.testing.assert_array_almost_equal(
            ens["tg_mean"].isel(time=0, lon=5, lat=5).weighted(weights).quantile(0.1),
            out4["tg_mean_p10"].isel(time=0, lon=5, lat=5),
        )
        np.testing.assert_array_almost_equal(
            ens["tg_mean"].isel(time=0, lon=5, lat=5).weighted(weights).quantile(0.9),
            out4["tg_mean_p90"].isel(time=0, lon=5, lat=5),
        )
        assert np.all(out4["tg_mean_p90"] > out4["tg_mean_p10"])

    @pytest.mark.parametrize("keep_chunk_size", [False, True, None])
    def test_calc_perc_dask(
        self, keep_chunk_size, ensemble_dataset_objects, open_dataset
    ):
        ds_all = []
        for n in ensemble_dataset_objects["nc_files_simple"]:
            ds = open_dataset(n)
            ds_all.append(ds)
        ens = ensembles.create_ensemble(ds_all)

        out2 = ensembles.ensemble_percentiles(
            ens.chunk({"time": 2}), keep_chunk_size=keep_chunk_size, split=False
        )
        out1 = ensembles.ensemble_percentiles(ens.load(), split=False)
        np.testing.assert_array_equal(out1["tg_mean"], out2["tg_mean"])

    def test_calc_perc_nans(self, ensemble_dataset_objects, open_dataset):
        ds_all = []
        for n in ensemble_dataset_objects["nc_files_simple"]:
            ds = open_dataset(n)
            ds_all.append(ds)
        ens = ensembles.create_ensemble(ds_all).load()

        ens.tg_mean[2, 0, 5, 5] = np.nan
        ens.tg_mean[2, 7, 5, 5] = np.nan
        out1 = ensembles.ensemble_percentiles(ens, split=True)
        masked_arr = np.ma.fix_invalid(ens["tg_mean"][:, 0, 5, 5])
        np.testing.assert_array_almost_equal(
            mquantiles(masked_arr, 0.10, alphap=1, betap=1),
            out1["tg_mean_p10"][0, 5, 5],
        )
        masked_arr = np.ma.fix_invalid(ens["tg_mean"][:, 7, 5, 5])
        np.testing.assert_array_almost_equal(
            mquantiles(masked_arr, 0.10, alphap=1, betap=1),
            out1["tg_mean_p10"][7, 5, 5],
        )
        assert np.all(out1["tg_mean_p90"] > out1["tg_mean_p50"])
        assert np.all(out1["tg_mean_p50"] > out1["tg_mean_p10"])

    def test_calc_mean_std_min_max(self, ensemble_dataset_objects, open_dataset):
        ds_all = []
        for n in ensemble_dataset_objects["nc_files_simple"]:
            ds = open_dataset(n)
            ds_all.append(ds)
        with pytest.warns(FutureWarning):
            ens = ensembles.create_ensemble(ds_all, mf_flag=False)

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
        assert "Computation of statistics on" in out1.attrs["history"]

        weights = xr.DataArray(
            [1, 0.1, 3.5, 5], coords={"realization": ens.realization}
        )
        out2 = ensembles.ensemble_mean_std_max_min(ens, weights=weights)
        values = ens["tg_mean"][:, 0, 5, 5]
        np.testing.assert_array_equal(
            (values[0] * 1 + values[1] * 0.1 + values[2] * 3.5 + values[3] * 5)
            / np.sum(weights),
            out2.tg_mean_mean[0, 5, 5],
        )
        np.testing.assert_array_equal(
            ens["tg_mean"][:, 0, 5, 5].weighted(weights).std(dim="realization"),
            out2.tg_mean_stdev[0, 5, 5],
        )
        np.testing.assert_array_equal(
            out1.tg_mean_max[0, 5, 5], out2.tg_mean_max[0, 5, 5]
        )
        np.testing.assert_array_equal(
            out1.tg_mean_min[0, 5, 5], out2.tg_mean_min[0, 5, 5]
        )


@pytest.mark.slow
class TestEnsembleReduction:
    nc_file = os.path.join("EnsembleReduce", "TestEnsReduceCriteria.nc")

    def test_kmeans_rsqcutoff(self, open_dataset):
        pytest.importorskip("sklearn", minversion="0.24.1")
        ds = open_dataset(self.nc_file)

        # use random state variable to ensure consistent clustering in tests:
        [ids, cluster, fig_data] = ensembles.kmeans_reduce_ensemble(
            data=ds.data, method={"rsq_cutoff": 0.5}, random_state=42, make_graph=False
        )

        assert ids == [4, 7, 10, 23]
        assert len(ids) == 4

        # Test max cluster option
        [ids, cluster, fig_data] = ensembles.kmeans_reduce_ensemble(
            data=ds.data,
            method={"rsq_cutoff": 0.5},
            random_state=42,
            make_graph=False,
            max_clusters=3,
        )
        assert ids == [4, 7, 23]
        assert len(ids) == 3

    def test_kmeans_rsqopt(self, open_dataset):
        pytest.importorskip("sklearn", minversion="0.24.1")
        ds = open_dataset(self.nc_file)
        [ids, cluster, fig_data] = ensembles.kmeans_reduce_ensemble(
            data=ds.data,
            method={"rsq_optimize": None},
            random_state=42,
            make_graph=False,
        )
        assert ids == [3, 4, 5, 7, 10, 11, 12, 13]
        assert len(ids) == 8

    def test_kmeans_nclust(self, open_dataset):
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

    def test_kmeans_sampleweights(self, open_dataset):
        ds = open_dataset(self.nc_file)
        # Test sample weights
        sample_weights = np.ones(ds.data.shape[0])
        # boost weights for some sims
        sample_weights[[0, 20]] = 15

        [ids, cluster, fig_data] = ensembles.kmeans_reduce_ensemble(
            data=ds.data,
            method={"rsq_cutoff": 0.5},
            random_state=42,
            make_graph=False,
            sample_weights=sample_weights,
        )
        assert ids == [0, 20, 23]
        assert len(ids) == 3

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

    def test_kmeans_variweights(self, open_dataset):
        pytest.importorskip("sklearn", minversion="0.24.1")
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

        # using RSQ optimize and try zero weights
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
        assert all(np.isin([12, 13, 16], ids))
        assert len(ids) == 6

    def test_kmeans_modelweights(self, open_dataset):
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

    @pytest.mark.skipif(
        "matplotlib.pyplot" not in sys.modules, reason="matplotlib.pyplot is required"
    )
    def test_kmeans_rsqcutoff_with_graphs(self, open_dataset):
        pytest.importorskip("sklearn", minversion="0.24.1")
        ds = open_dataset(self.nc_file)

        # use random state variable to ensure consistent clustering in tests:
        [ids, cluster, fig_data] = ensembles.kmeans_reduce_ensemble(
            data=ds.data, method={"rsq_cutoff": 0.5}, random_state=42, make_graph=True
        )

        assert ids == [4, 7, 10, 23]
        assert len(ids) == 4

        # Test max cluster option
        [ids, cluster, fig_data] = ensembles.kmeans_reduce_ensemble(
            data=ds.data,
            method={"rsq_cutoff": 0.5},
            random_state=42,
            make_graph=True,
            max_clusters=3,
        )
        assert ids == [4, 7, 23]
        assert len(ids) == 3

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
    def test_kkz_simple(self, open_dataset, crit, num_select, expected):
        ens = open_dataset(self.nc_file)
        data = ens.data.isel(criteria=crit)

        selected = ensembles.kkz_reduce_ensemble(data, num_select)
        assert selected == expected

    def test_kkz_standardize(self, open_dataset):
        ens = open_dataset(self.nc_file)
        data = ens.data.isel(criteria=[1, 3, 5])

        sel_std = ensembles.kkz_reduce_ensemble(data, 4, standardize=True)
        sel_no = ensembles.kkz_reduce_ensemble(data, 4, standardize=False)
        assert sel_std == [23, 10, 19, 14]
        assert sel_no == [23, 1, 14, 10]

    def test_kkz_change_metric(self, open_dataset):
        # This test uses stupid values but is meant to test is kwargs are passed and if dist_method is used.
        ens = open_dataset(self.nc_file)
        data = ens.data.isel(criteria=[1, 3, 5])

        sel_euc = ensembles.kkz_reduce_ensemble(data, 4, dist_method="euclidean")
        sel_mah = ensembles.kkz_reduce_ensemble(
            data, 4, dist_method="mahalanobis", VI=np.arange(24)
        )
        assert sel_euc == [23, 10, 19, 14]
        assert sel_mah == [5, 3, 4, 0]

    def test_standardize_seuclidean(self, open_dataset):
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

    def test_make_criteria(self, tas_series):
        ds = xr.Dataset(
            data_vars={
                "var_a": tas_series([0, 1, 2, 3]),
                "var_b": tas_series([0, 1, 2, 3]).expand_dims(lat=[45, 47]),
                "var_c": tas_series([0, 1, 2, 3]),
            }
        ).expand_dims(realization=["A", "B", "C"])

        crit = ensembles.make_criteria(ds)
        assert crit.dims == ("realization", "criteria")
        assert crit.criteria.size == 16
        uncrit = crit.unstack("criteria").to_dataset("variables")
        assert set(uncrit.data_vars.keys()) == {"var_a", "var_b", "var_c"}
        assert set(uncrit.var_a.dims) == {"realization", "lat", "time"}

        crit = ensembles.make_criteria(ds.var_b)
        assert crit.dims == ("realization", "criteria")
        assert crit.criteria.size == 8
        uncrit = crit.unstack("criteria")
        assert set(uncrit.dims) == {"realization", "lat", "time"}


# ## Tests for Robustness ##
@pytest.fixture(params=[True, False])
def robust_data(request):
    norm = get_dist("norm")
    ref = np.tile(
        np.array(
            [
                norm.rvs(loc=274, scale=0.8, size=(40,), random_state=r)
                for r in [101083, 19377, 473820, 483625]
            ]
        ),
        (4, 1, 1),
    )
    fut = np.array(
        [
            [
                norm.rvs(loc=loc, scale=sc, size=(40,), random_state=r)
                for loc, sc, r in shps
            ]
            for shps in (
                [
                    (274.0, 0.7, 176378),
                    (274.0, 0.6, 839789),
                    (274.0, 0.7, 393239),
                    (275.6, 1.1, 747390),
                ],  # 3 no change, 1 positive change
                [
                    (272.5, 1.2, 743920),
                    (272.4, 0.8, 138489),
                    (275.5, 0.8, 673683),
                    (275.6, 1.1, 969383),
                ],  # 2 neg change
                [
                    (275.6, 0.8, 696857),
                    (275.8, 1.2, 379949),
                    (276.5, 0.8, 268395),
                    (277.6, 1.1, 456544),
                ],  # All pos change
                [
                    (np.nan, 0.3, 746323),
                    (np.nan, 1.2, 5643723),
                    (275.5, 0.8, 118294),
                    (275.6, 1.1, 574732),
                ],  # Some NaN
            )
        ]
    )
    ref = xr.DataArray(ref, dims=("lon", "realization", "time"), name="tas")
    ref["time"] = xr.cftime_range("2000-01-01", periods=40, freq="YS")
    fut = xr.DataArray(fut, dims=("lon", "realization", "time"), name="tas")
    fut["time"] = xr.cftime_range("2040-01-01", periods=40, freq="YS")
    if request.param:
        ref = ref.chunk({"lon": 1}).to_dataset()
        fut = fut.chunk({"lon": 1}).to_dataset()
    return ref, fut


@pytest.mark.parametrize(
    "test,exp_chng_frac,exp_pos_frac,exp_changed,kws",
    [
        (
            "ttest",
            [0.25, 1, 1, 1],
            [1, 0.5, 1, 1],
            [
                [False, False, False, True],
                [True, True, True, True],
                [True, True, True, True],
                [False, False, True, True],
            ],
            {},
        ),
        (
            "welch-ttest",
            [0.25, 1, 1, 1],
            [1, 0.5, 1, 1],
            [
                [False, False, False, True],
                [True, True, True, True],
                [True, True, True, True],
                [False, False, True, True],
            ],
            {},
        ),
        (
            "mannwhitney-utest",
            [0.25, 1, 1, 1],
            [1, 0.5, 1, 1],
            [
                [False, False, False, True],
                [True, True, True, True],
                [True, True, True, True],
                [False, False, True, True],
            ],
            {},
        ),
        (
            "brownforsythe-test",
            [0.5, 0.25, 0.5, 0.5],
            [0.5, 0.0, 1, 1],
            [
                [False, True, False, True],
                [True, False, False, False],
                [False, True, False, True],
                [False, False, False, True],
            ],
            {},
        ),
        (
            "threshold",
            [0.25, 1, 1, 1],
            [1, 0.5, 1, 1],
            [
                [False, False, False, True],
                [True, True, True, True],
                [True, True, True, True],
                [False, False, True, True],
            ],
            {"rel_thresh": 0.002},
        ),
        (
            "threshold",
            [0, 0, 0.5, 0],
            [np.nan, np.nan, 1, np.nan],
            [
                [False, False, False, False],
                [False, False, False, False],
                [False, False, True, True],
                [False, False, False, False],
            ],
            {"abs_thresh": 2},
        ),
        (
            None,
            [1, 1, 1, 1],
            [0.5, 0.5, 1, 1],
            [],
            {},
        ),
    ],
)
def test_change_significance(
    robust_data, test, exp_chng_frac, exp_pos_frac, exp_changed, kws
):
    ref, fut = robust_data

    if test == "ttest" and parse_version(__scipy_version__) < parse_version("1.9.0"):
        with pytest.warns(FutureWarning):
            chng_frac, pos_frac = ensembles.change_significance(
                fut, ref, test=test, **kws
            )
    else:
        chng_frac, pos_frac = ensembles.change_significance(fut, ref, test=test, **kws)

    assert chng_frac.attrs["test"] == str(test)
    if isinstance(ref, xr.Dataset):
        chng_frac = chng_frac.tas
        pos_frac = pos_frac.tas

    np.testing.assert_array_almost_equal(chng_frac, exp_chng_frac)
    np.testing.assert_array_almost_equal(pos_frac, exp_pos_frac)

    # With p-values
    chng, sign, pvals = ensembles.change_significance(
        fut, ref, test=test, p_vals=True, **kws
    )
    if pvals is not None:
        if isinstance(ref, xr.Dataset):
            pvals = pvals.tas
        pvals.load()  # Otherwise it acts weirdly for welch-ttest
        # 0.05 is the default p_change parameter
        changed = pvals < 0.05
        np.testing.assert_array_almost_equal(changed, exp_changed)


def test_change_significance_weighted(robust_data):
    ref, fut = robust_data
    weights = xr.DataArray([1, 0.1, 3.5, 5], coords={"realization": ref.realization})
    chng_frac, pos_frac = ensembles.change_significance(
        fut, ref, test=None, weights=weights
    )
    assert chng_frac.attrs["test"] == "None"
    if isinstance(ref, xr.Dataset):
        chng_frac = chng_frac.tas
        pos_frac = pos_frac.tas

    np.testing.assert_array_equal(chng_frac, [1, 1, 1, 1])
    np.testing.assert_array_almost_equal(pos_frac, [0.88541667, 0.88541667, 1.0, 1.0])


def test_change_significance_delta(robust_data):
    ref, fut = robust_data
    delta = fut.mean("time") - ref.mean("time")
    chng_frac, pos_frac = ensembles.change_significance(
        delta, test="threshold", abs_thresh=2
    )

    if isinstance(ref, xr.Dataset):
        chng_frac = chng_frac.tas
        pos_frac = pos_frac.tas

    exp_chng_frac = [0, 0, 0.5, 0]
    exp_pos_frac = [np.nan, np.nan, 1, np.nan]
    np.testing.assert_array_equal(chng_frac, exp_chng_frac)
    np.testing.assert_array_equal(pos_frac, exp_pos_frac)

    weights = xr.DataArray([1, 0.1, 3.5, 5], coords={"realization": delta.realization})
    chng_frac, pos_frac = ensembles.change_significance(
        delta, test="threshold", abs_thresh=2, weights=weights
    )
    if isinstance(ref, xr.Dataset):
        chng_frac = chng_frac.tas
        pos_frac = pos_frac.tas

    exp_chng_frac = [0, 0, 0.88541667, 0]
    exp_pos_frac = [np.nan, np.nan, 1, np.nan]

    np.testing.assert_array_almost_equal(chng_frac, exp_chng_frac)
    np.testing.assert_array_equal(pos_frac, exp_pos_frac)


def test_change_significance_empty():
    """Test that NaN is returned if input arrays are full of NaNs."""
    r = np.full((20, 10), np.nan)
    f = np.full((20, 10), np.nan)

    ref = xr.DataArray(r, dims=("realization", "time"), name="tas")
    fut = xr.DataArray(f, dims=("realization", "time"), name="tas")

    c, f = ensembles.change_significance(fut, ref, test="ttest")
    assert np.all(np.isnan(c))


def test_robustness_coefficient():
    # High
    ref = xr.DataArray([274, 275, 274.5, 276, 274.3, 273.3], dims=("time",), name="tas")
    fut = xr.DataArray(
        [
            [277, 277.1, 278, 278.4, 278.1, 276.9],
            [275, 275.8, 276, 275.2, 276.2, 275.7],
        ],
        dims=("realization", "time"),
        name="tas",
    )
    R = ensembles.robustness_coefficient(fut, ref)
    np.testing.assert_almost_equal(R, 0.91972477)

    fut = xr.DataArray(
        [
            [277, 277.1, 278, 278.4, 278.1, 276.9],
            [274, 274.8, 273.7, 274.2, 273.9, 274.5],
        ],
        dims=("realization", "time"),
        name="tas",
    )
    R = ensembles.robustness_coefficient(fut, ref)
    np.testing.assert_almost_equal(R, 0.83743842)

    R = ensembles.robustness_coefficient(fut.to_dataset(), ref.to_dataset())
    np.testing.assert_almost_equal(R.tas, 0.83743842)

import numpy as np
import pandas as pd
import scipy
import xarray as xr
from sklearn.cluster import KMeans

# Avoid having to include matplotlib in xclim requirements
try:
    import matplotlib.pyplot as plt

    make_graph = True
except ModuleNotFoundError:
    make_graph = False


def create_ensemble(datasets, mf_flag=False):
    """Create an xarray datset of ensemble of climate simulation from a list of netcdf files. Input data is
    concatenated along a newly created data dimension ('realization')

    Returns a xarray dataset object containing input data from the list of netcdf files concatenated along
    a new dimension (name:'realization'). In the case where input files have unequal time dimensions output
    ensemble dataset is created for maximum time-step interval of all input files.  Before concatenation datasets not
    covering the entire time span have their data padded with `nan` values

    Parameters
    ----------
    datasets : sequence
      List of netcdf file paths or xr.Datasets . If mf_flag is true ncfiles should be a list of lists where
    each sublist contains input .nc files of a multifile dataset

    mf_flag : Boolean . If True climate simulations are treated as multifile datasets before concatenation.
    Only applicable when `datasets` is a sequence of file paths

    Returns
    -------
    xarray dataset containing concatenated data from all input files

    Notes
    -----
    Input netcdf files require equal spatial dimension size (e.g. lon, lat dimensions)
    If input data contains multiple cftime calendar types they must be at monthly or coarser frequency

    Examples
    --------
    >>> from xclim import ensembles
    >>> import glob
    >>> ncfiles = glob.glob('/*.nc')
    >>> ens = ensembles.create_ensemble(datasets)
    >>> print(ens)
    Using multifile datasets:
    simulation 1 is a list of .nc files (e.g. separated by time)
    >>> datasets = glob.glob('dir/*.nc')
    simulation 2 is also a list of .nc files
    >>> datasets.append(glob.glob('dir2/*.nc'))
    >>> ens = utils.create_ensemble(datasets, mf_flag=True)
    """

    dim = "realization"

    time_flag, time_all = _ens_checktimes(datasets, mf_flag)

    ds1 = _ens_align_datasets(datasets, mf_flag, time_flag, time_all)

    # print('concatenating files : adding dimension ', dim, )
    ens = xr.concat(ds1, dim=dim)

    return ens


def ensemble_mean_std_max_min(ens):
    """Calculate ensemble statistics between a results from an ensemble of climate simulations

    Returns a dataset containing ensemble mean, standard-deviation,
    minimum and maximum for input climate simulations.

    Parameters
    ----------
    ens : Ensemble dataset (see xclim.ensembles.create_ensemble)

    Returns
    -------
    xarray dataset with containing data variables of ensemble statistics

    Examples
    --------
    >>> from xclim import ensembles
    >>> import glob
    >>> ncfiles = glob.glob('/*tas*.nc')
    Create ensemble dataset
    >>> ens = ensembles.create_ensemble(ncfiles)
    Calculate ensemble statistics
    >>> ens_means_std = ensembles.ensemble_mean_std_max_min(ens)
    >>> print(ens_mean_std['tas_mean'])
    """
    dsOut = ens.drop(ens.data_vars)
    for v in ens.data_vars:

        dsOut[v + "_mean"] = ens[v].mean(dim="realization")
        dsOut[v + "_stdev"] = ens[v].std(dim="realization")
        dsOut[v + "_max"] = ens[v].max(dim="realization")
        dsOut[v + "_min"] = ens[v].min(dim="realization")
        for vv in dsOut.data_vars:
            dsOut[vv].attrs = ens[v].attrs

            if "description" in dsOut[vv].attrs.keys():
                vv.split()
                dsOut[vv].attrs["description"] = (
                    dsOut[vv].attrs["description"]
                    + " : "
                    + vv.split("_")[-1]
                    + " of ensemble"
                )

    return dsOut


def ensemble_percentiles(ens, values=(10, 50, 90), time_block=None):
    """Calculate ensemble statistics between a results from an ensemble of climate simulations

    Returns a dataset containing ensemble statistics for input climate simulations.
    Alternatively calculate ensemble percentiles (default) or ensemble mean and standard deviation

    Parameters
    ----------
    ens : Ensemble dataset (see xclim.ensembles.create_ensemble)
    values : tuple of integers - percentile values to calculate  : default : (10, 50, 90)
    time_block : integer
      for large ensembles iteratively calculate percentiles in time-step blocks (n==time_block).
       If not defined the function tries to estimate an appropriate value

    Returns
    -------
    xarray dataset with containing data variables of requested ensemble statistics

    Examples
    --------
    >>> from xclim import ensembles
    >>> import glob
    >>> ncfiles = glob.glob('/*tas*.nc')
    Create ensemble dataset
    >>> ens = ensembles.create_ensemble(ncfiles)
    Calculate default ensemble percentiles
    >>> ens_percs = ensembles.ensemble_statistics(ens)
    >>> print(ens_percs['tas_p10'])
    Calculate non-default percentiles (25th and 75th)
    >>> ens_percs = ensembles.ensemble_statistics(ens, values=(25,75))
    >>> print(ens_percs['tas_p25'])
    Calculate by time blocks (n=10) if ensemble size is too large to load in memory
    >>> ens_percs = ensembles.ensemble_statistics(ens, time_block=10)
    >>> print(ens_percs['tas_p25'])

    """

    ds_out = ens.drop(ens.data_vars)
    dims = list(ens.dims)
    for v in ens.data_vars:
        # Percentile calculation requires load to memory : automate size for large ensemble objects
        if not time_block:
            time_block = round(
                2e8 / (ens[v].size / ens[v].shape[dims.index("time")]), -1
            )  # 2E8

        if time_block > len(ens[v].time):
            out = _calc_percentiles_simple(ens, v, values)

        else:
            # loop through blocks
            Warning(
                "large ensemble size detected : statistics will be calculated in blocks of ",
                int(time_block),
                " time-steps",
            )
            out = _calc_percentiles_blocks(ens, v, values, time_block)
        for vv in out.data_vars:
            ds_out[vv] = out[vv]
    return ds_out


def _ens_checktimes(datasets, mf_flag=False):
    """Check list of datasets and determine if they hava a time dimension. If present return the
    maximum time-step interval of all input files

    Parameters
    ----------
    datasets : sequence
      List of netcdf file paths or xr.Datasets . If mf_flag is true ncfiles should be a list of lists where
    each sublist contains input .nc files of a multifile dataset

    mf_flag : Boolean . If True climate simulations are treated as multifile datasets before concatenation.
    Only applicable when `datasets` is a sequence of file paths

    Returns
    -------
    time_flag : bool; True if time dimension is present in the dataset list otherwise false.

    time_all : array of datetime64; Series of unique time-steps covering all input datasets.
    """

    time_flag = False
    time_all = []
    for n in datasets:
        if mf_flag:
            ds = xr.open_mfdataset(
                n, concat_dim="time", decode_times=False, chunks={"time": 10}
            )
        else:
            if isinstance(n, xr.Dataset):
                ds = n
            else:
                ds = xr.open_dataset(n, decode_times=False)

        if hasattr(ds, "time"):
            ds["time"] = xr.decode_cf(ds).time
            time_flag = True

            # get times - use common
            time1 = pd.to_datetime(
                {
                    "year": ds.time.dt.year,
                    "month": ds.time.dt.month,
                    "day": ds.time.dt.day,
                }
            )

            time_all.extend(time1.values)
    if time_flag:
        time_all = pd.unique(time_all)
        time_all.sort()
    else:
        time_all = None
    return time_flag, time_all


def _ens_align_datasets(datasets, mf_flag=False, time_flag=False, time_all=None):
    """Create a list of aligned xr.Datasets for ensemble dataset creation. If `time_flag == True` input datasets are
    given a common time dimension defined by `time_all`. Datasets not covering the entire time span have their data
    padded with `nan` values

    Parameters
    ----------
    datasets : sequence
      List of netcdf file paths or xr.Datasets . If mf_flag is true ncfiles should be a list of lists where
    each sublist contains input .nc files of a multifile dataset

    mf_flag : Boolean . If True climate simulations are treated as multifile datasets before concatenation.
    Only applicable when `datasets` is a sequence of file paths

    time_flag : bool; True if time dimension is present in the dataset list otherwise false.

    time_all : array of datetime64; Series of unique time-steps covering all input datasets.

    Returns
    -------
    ds_all : list; list of xr.Datasets

    """

    ds_all = []
    for n in datasets:
        # print('accessing file ', ncfiles.index(n) + 1, ' of ', len(ncfiles))
        if mf_flag:
            ds = xr.open_mfdataset(
                n, concat_dim="time", decode_times=False, chunks={"time": 10}
            )
        else:
            if isinstance(n, xr.Dataset):
                ds = n
            else:
                ds = xr.open_dataset(n, decode_times=False, chunks={"time": 10})

        if time_flag:

            ds["time"] = xr.decode_cf(ds).time

            ds["time"].values = pd.to_datetime(
                {
                    "year": ds.time.dt.year,
                    "month": ds.time.dt.month,
                    "day": ds.time.dt.day,
                }
            )
            # if dataset does not have the same time steps pad with nans
            if ds.time.min() > time_all.min() or ds.time.max() < time_all.max():
                coords = {}
                for c in [c for c in ds.coords if "time" not in c]:
                    coords[c] = ds.coords[c]
                coords["time"] = time_all
                dsTmp = xr.Dataset(data_vars=None, coords=coords, attrs=ds.attrs)
                for v in ds.data_vars:
                    dsTmp[v] = ds[v]
                ds = dsTmp
            # ds = ds.where((ds.time >= start1) & (ds.time <= end1), drop=True)
        ds_all.append(ds)

    return ds_all


def _calc_percentiles_simple(ens, v, values):
    ds_out = ens.drop(ens.data_vars)
    dims = list(ens[v].dims)
    outdims = [x for x in dims if "realization" not in x]

    # print('loading ensemble data to memory')
    arr = ens[v].load()  # percentile calc requires loading the array
    coords = {}
    for c in outdims:
        coords[c] = arr[c]
    for p in values:
        outvar = v + "_p" + str(p)

        out1 = _calc_perc(arr, p)

        ds_out[outvar] = xr.DataArray(out1, dims=outdims, coords=coords)
        ds_out[outvar].attrs = ens[v].attrs
        if "description" in ds_out[outvar].attrs.keys():
            ds_out[outvar].attrs[
                "description"
            ] = "{} : {}th percentile of ensemble".format(
                ds_out[outvar].attrs["description"], str(p)
            )
        else:
            ds_out[outvar].attrs["description"] = "{}th percentile of ensemble".format(
                str(p)
            )

    return ds_out


def _calc_percentiles_blocks(ens, v, values, time_block):
    ds_out = ens.drop(ens.data_vars)
    dims = list(ens[v].dims)
    outdims = [x for x in dims if "realization" not in x]

    blocks = list(range(0, len(ens.time) + 1, int(time_block)))
    if blocks[-1] != len(ens[v].time):
        blocks.append(len(ens[v].time))
    arr_p_all = {}
    for t in range(0, len(blocks) - 1):
        # print('Calculating block ', t + 1, ' of ', len(blocks) - 1)
        time_sel = slice(blocks[t], blocks[t + 1])
        arr = (
            ens[v].isel(time=time_sel).load()
        )  # percentile calc requires loading the array
        coords = {}
        for c in outdims:
            coords[c] = arr[c]
        for p in values:

            out1 = _calc_perc(arr, p)

            if t == 0:
                arr_p_all[str(p)] = xr.DataArray(out1, dims=outdims, coords=coords)
            else:
                arr_p_all[str(p)] = xr.concat(
                    [
                        arr_p_all[str(p)],
                        xr.DataArray(out1, dims=outdims, coords=coords),
                    ],
                    dim="time",
                )
    for p in values:
        outvar = v + "_p" + str(p)
        ds_out[outvar] = arr_p_all[str(p)]
        ds_out[outvar].attrs = ens[v].attrs
        if "description" in ds_out[outvar].attrs.keys():
            ds_out[outvar].attrs[
                "description"
            ] = "{} : {}th percentile of ensemble".format(
                ds_out[outvar].attrs["description"], str(p)
            )
        else:
            ds_out[outvar].attrs["description"] = "{}th percentile of ensemble".format(
                str(p)
            )

    return ds_out


def _calc_perc(arr, p):
    dims = arr.dims
    # make sure time is the second dimension
    if dims.index("time") != 1:
        dims1 = [dims[dims.index("realization")], dims[dims.index("time")]]
        for d in dims:
            if d not in dims1:
                dims1.append(d)
        arr = arr.transpose(*dims1)
        dims = dims1

    nan_count = np.isnan(arr).sum(axis=dims.index("realization"))
    out = np.percentile(arr, p, axis=dims.index("realization"))
    if np.any((nan_count > 0) & (nan_count < arr.shape[dims.index("realization")])):
        arr1 = arr.values.reshape(
            arr.shape[dims.index("realization")],
            int(arr.size / arr.shape[dims.index("realization")]),
        )
        # only use nanpercentile where we need it (slow performace compared to standard) :
        nan_index = np.where(
            (nan_count > 0) & (nan_count < arr.shape[dims.index("realization")])
        )
        t = np.ravel_multi_index(nan_index, nan_count.shape)
        out[np.unravel_index(t, nan_count.shape)] = np.nanpercentile(
            arr1[:, t], p, axis=dims.index("realization")
        )

    return out


def kmeans_reduce_ensemble(
    sel_criteria,
    method=None,
    max_clusters=None,
    variable_weights=None,
    sample_weights=None,
    model_weights=None,
    make_graph=make_graph,
    random_state=None,
):
    """Return a sample (selection) of ensemble members using k-means clustering. The algorithm attempts to
    reduce the total number of ensemble members while maintaining adequate coverage of the ensemble
    uncertainty (variance) in a N-dimensional data space (sel_criteria). K-Means clustering is carried out on the input
    selection criteria data-array in order to group individual ensemble members into a reduced number of similar groups
    Subsequently a single representative simulation is identified from each group.


    Parameters
    ----------
    sel_criteria : xr.DataArray (NxP array)  ---  Selecton criteria data. These are the values used for clustering.
        N is the number of realizations in the original ensemble and P the number of variables/indicators used in the grouping
        algorithm

    method : dict. Dictionary defining selection method and associated value (when required). One of the following:

        {'rsq_optimize':None} : Default - Optimize the cost (number of ensemble members) versus benefit
            (variance coverage) relationship. For details see supporting information (S2 text) in
            https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0152495

        {'rsq_cutoff': val} : threshold Coefficient of variation (R² value) above which to cover with the
            grouping. val : float between 0 and 1. The R² indicates the proportion of the total variance in sel_criteria
            that is explained by the grouping

        {'n_clusters': val} : Create a user determined number of clusters. val : integer between 1 and N

    Optional parameters:
    max_clusters : integer  --  Maximum number of members to include in the output ensemble selection.
        When using 'rsq_optimize' or 'rsq_cutoff' methods, limit the final selection to a maximum number even if method
        results indicate a higher value. Defaults to N (number ensemble members)

    variable_weights: xr.DataArray of size P  --  This weighting can be used to influence of weight of the climate
        indices on the clustering itself

    sample_weights: xr.DataArray of size N  --  This weighting can be used to influence of weight of simulations on
        the clustering itself. For example, putting a weight of 0 on a simulation will completely exclude it from the
        clustering

    model_weights: xr.DataArray of size N  --  This weighting can be used to influence which model is selected within
        each cluster. This parameter has no influence whatsoever on the clustering itself.

    graph: boolean --  displays a plot of R² vs. the number of clusters

    random_state -- sklearn.cluster.KMeans() random_state parameter. Determines random number generation for centroid
        initialization. Use an int to make the randomness deterministic.
        see https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html

    Returns
    -------

    out : list -- Selected model indexes (positions)
    clusters : KMeans clustering results



    References
    -----
    Casajus et al. 2016. https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0152495


    Examples
    --------
    >
    """

    # initialize the variables
    n_sim = np.shape(sel_criteria)[0]  # number of simulations
    n_idx = np.shape(sel_criteria)[1]  # number of indicators

    # normalize the data matrix
    z = xr.DataArray(
        scipy.stats.zscore(sel_criteria, axis=0, ddof=1)
    )  # ddof=1 to be the same as Matlab's zscore

    if sample_weights is None:
        sample_weights = np.ones(n_sim)
    else:
        # TODO KMeans sample weights of zero cause errors occasionally - set to 1e-15 for now
        sample_weights[sample_weights == 0] = 1e-15
    if model_weights is None:
        model_weights = np.ones(n_sim)
    if variable_weights is None:
        variable_weights = np.ones(shape=(1, n_idx))
    if max_clusters is None:
        max_clusters = n_sim
    if method is None:
        method = {"rsq_optimize": None}

    # normalize the weights (note: I don't know if this is really useful... this was in the MATLAB code)
    sample_weights = sample_weights / np.sum(sample_weights)
    model_weights = model_weights / np.sum(model_weights)
    variable_weights = variable_weights / np.sum(variable_weights)

    z = z * variable_weights
    rsq = _calc_rsq(z, method, make_graph, n_sim, random_state, sample_weights)
    if make_graph:
        # make a plot of rsq profile
        plt.figure(figsize=(10, 6))
        plt.plot(
            range(1, n_sim + 1), rsq, "k-o", label="R²", linewidth=0.8, markersize=4
        )
        # plt.plot(np.arange(1.5, n_sim + 0.5), np.diff(rsq), 'r', label='ΔR²')
        axes = plt.gca()
        axes.set_xlim([0, n_sim])
        axes.set_ylim([0, 1])
        plt.xlabel("Number of groups")
        plt.ylabel("R²")
        plt.legend(loc="lower right")
        plt.title("R² of groups vs. full ensemble")

    n_clusters = _get_nclust(method, n_sim, rsq, make_graph, max_clusters)

    # Finale k-means clustering with 1000 iterations to avoid instabilities in the choice of final scenarios
    kmeans = KMeans(n_clusters=n_clusters, n_init=1000, max_iter=600)
    # we use 'fit_' only once, otherwise it computes everything again
    clusters = kmeans.fit_predict(z, sample_weight=sample_weights)

    # squared distance to centroids
    d = np.square(
        kmeans.transform(z)
    )  # squared distance between each point and each centroid

    out = np.empty(
        shape=n_clusters
    )  # prepare an empty array in which to store the results
    r = np.arange(n_sim)

    # in each cluster, find the closest (weighted) simulation and select it
    for i in range(n_clusters):
        d_i = d[
            clusters == i, i
        ]  # distance to the centroid for all simulations within the cluster 'i'
        if d_i.shape[0] > 2:
            sig = np.std(
                d_i, ddof=1
            )  # standard deviation of those distances (ddof = 1 gives the same as Matlab's std function)
            like = (
                scipy.stats.norm.pdf(d_i, 0, sig) * model_weights[clusters == i]
            )  # weighted likelihood
            argmax = np.argmax(like)  # index of the maximum likelihood
        elif d_i.shape[0] == 2:
            sig = (
                1
            )  # standard deviation would be 0 for a 2-simulation cluster, meaning that model_weights would be ignored.
            like = (
                scipy.stats.norm.pdf(d_i, 0, sig) * model_weights[clusters == i]
            )  # weighted likelihood
            argmax = np.argmax(like)  # index of the maximum likelihood
        else:
            argmax = 0
        r_clust = r[
            clusters == i
        ]  # index of the cluster simulations within the full ensemble

        out[i] = r_clust[argmax]

    out = sorted(out.astype(int))
    # display graph - don't block code execution
    if make_graph:
        plt.show(block=False)
    return out, clusters


def _calc_rsq(z, method, make_graph, n_sim, random_state, sample_weights):
    rsq = None
    if list(method.keys())[0] != "n_clusters" or make_graph is True:
        # generate r2 profile data
        sumd = np.zeros(shape=n_sim) + np.nan
        for nclust in range(n_sim):
            # This is k-means with only 10 iterations, to limit the computation times
            kmeans = KMeans(
                n_clusters=nclust + 1,
                n_init=15,
                max_iter=300,
                random_state=random_state,
            )
            kmeans = kmeans.fit(z, sample_weight=sample_weights)
            sumd[
                nclust
            ] = (
                kmeans.inertia_
            )  # sum of the squared distance between each simulation and the nearest cluster centroid

        # R² of the groups vs. the full ensemble
        rsq = (sumd[0] - sumd) / sumd[0]

    return rsq


def _get_nclust(method=None, n_sim=None, rsq=None, make_graph=None, max_clusters=None):
    """Subfunction to kmean_reduce_ensemble.
       Determine number of clusters to create depending on various methods
    """

    # if we actually need to find the optimal number of clusters, this is where it is done
    if list(method.keys())[0] == "rsq_cutoff":
        # argmax finds the first occurence of rsq > rsq_cutoff,but we need to add 1 b/c of python indexing
        n_clusters = np.argmax(rsq > method["rsq_cutoff"]) + 1
        if make_graph:
            plt.plot(
                (0, n_clusters, n_clusters),
                (rsq[n_clusters - 1], rsq[n_clusters - 1], 0),
                "k--",
                label="R² > {rsq_cut} (n = {n_clusters})".format(
                    rsq_cut=method["rsq_cutoff"], n_clusters=n_clusters
                ),
                linewidth=0.75,
            )
            plt.legend(loc="lower right")
    elif list(method.keys())[0] == "rsq_optimize":
        # create constant benefits curve (one to one)
        onetoone = -1 * (1.0 / (n_sim - 1)) + np.arange(1, n_sim + 1) * (
            1.0 / (n_sim - 1)
        )
        n_clusters = np.argmax(rsq - onetoone) + 1
        if make_graph:
            plt.plot(
                range(1, n_sim + 1),
                onetoone,
                color=[0.25, 0.25, 0.75],
                label="Theoretical constant increase in R²",
                linewidth=0.5,
            )
            plt.plot(
                range(1, n_sim + 1),
                rsq - onetoone,
                color=[0.75, 0.25, 0.25],
                label="Real benefits (R² - theoretical)",
                linewidth=0.5,
            )
            plt.plot(
                (0, n_clusters, n_clusters),
                (rsq[n_clusters - 1], rsq[n_clusters - 1], 0),
                "k--",
                linewidth=0.75,
                label="Optimized R² cost / benefit (n = {n_clusters})".format(
                    n_clusters=n_clusters
                ),
            )
            plt.legend(loc="center right")
            # plt.show()
    elif list(method.keys())[0] == "n_clusters":
        n_clusters = method["n_clusters"]
    else:
        raise Exception(
            "unknown selection method : {meth}".format(meth=list(method.keys()))
        )
    if n_clusters > max_clusters:
        print(
            str(n_clusters)
            + " clusters has been found to be the optimal number of clusters, but limiting "
            "to " + str(max_clusters) + " as required by user provided max_clusters"
        )
        n_clusters = max_clusters
    return n_clusters

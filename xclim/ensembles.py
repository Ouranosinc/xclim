from pathlib import Path
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import pandas as pd
import scipy
import xarray as xr
from sklearn.cluster import KMeans

# Avoid having to include matplotlib in xclim requirements
try:
    import matplotlib.pyplot as plt

    MPL_INSTALLED = True

except ImportError:
    MPL_INSTALLED = False


def create_ensemble(
    datasets: List[Union[xr.Dataset, Path, str, List[Union[Path, str]]]],
    mf_flag: bool = False,
) -> xr.Dataset:

    """Create an xarray dataset of an ensemble of climate simulation from a list of netcdf files. Input data is
    concatenated along a newly created data dimension ('realization')

    Returns an xarray dataset object containing input data from the list of netcdf files concatenated along
    a new dimension (name:'realization'). In the case where input files have unequal time dimensions, the output
    ensemble Dataset is created for maximum time-step interval of all input files.  Before concatenation, datasets not
    covering the entire time span have their data padded with NaN values.

    Parameters
    ----------
    datasets : List[Union[xr.Dataset, Path, str, List[Path, str]]]
      List of netcdf file paths or xarray DataSet objects . If mf_flag is True, ncfiles should be a list of lists where
      each sublist contains input .nc files of an xarray multifile Dataset.

    mf_flag : bool
      If True, climate simulations are treated as xarray multifile Datasets before concatenation.
      Only applicable when "datasets" is a sequence of file paths.

    Returns
    -------
    xr.Dataset
      Dataset containing concatenated data from all input files.

    Notes
    -----
    Input netcdf files require equal spatial dimension size (e.g. lon, lat dimensions).
    If input data contains multiple cftime calendar types they must be at monthly or coarser frequency.

    Examples
    --------
    >>> from xclim import ensembles
    >>> import glob
    >>> ncfiles = glob.glob('/*.nc')
    >>> ens = ensembles.create_ensemble(datasets)
    >>> print(ens)
    # Using multifile datasets:
    # simulation 1 is a list of .nc files (e.g. separated by time)
    >>> datasets = glob.glob('dir/*.nc')
    # simulation 2 is also a list of .nc files
    >>> datasets.append(glob.glob('dir2/*.nc'))
    >>> ens = utils.create_ensemble(datasets, mf_flag=True)
    """

    dim = "realization"

    time_flag, time_all = _ens_checktimes(datasets, mf_flag)

    ds1 = _ens_align_datasets(datasets, mf_flag, time_flag, time_all)

    for v in list(ds1[0].data_vars):
        list1 = [ds[v] for ds in ds1]
        data = xr.concat(list1, dim=dim)
        if v == list(ds1[0].data_vars)[0]:
            ens = xr.Dataset(data_vars=None, coords=data.coords, attrs=ds1[0].attrs)
        ens[v] = data

    return ens


def ensemble_mean_std_max_min(ens: xr.Dataset) -> xr.Dataset:
    """Calculate ensemble statistics between a results from an ensemble of climate simulations

    Returns an xarray Dataset containing ensemble mean, standard-deviation, minimum and maximum for input climate
    simulations.

    Parameters
    ----------
    ens: xr.Dataset
      Ensemble dataset (see xclim.ensembles.create_ensemble).

    Returns
    -------
    xr.Dataset
      Dataset with data variables of ensemble statistics.

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


def ensemble_percentiles(
    ens: xr.Dataset,
    values: Tuple[int, int, int] = (10, 50, 90),
    time_block: Optional[int] = None,
) -> xr.Dataset:
    """Calculate ensemble statistics between a results from an ensemble of climate simulations.

    Returns a Dataset containing ensemble statistics for input climate simulations.
    Alternatively calculate ensemble percentiles (default) or ensemble mean and standard deviation.

    Parameters
    ----------
    ens: xr.Dataset
      Ensemble dataset (see xclim.ensembles.create_ensemble).
    values : Tuple[int, int, int]
      Percentile values to calculate. Default: (10, 50, 90).
    time_block : Optional[int]
      For large ensembles, iteratively calculate percentiles in time-step blocks (n==time_block).
      If not defined, the function tries to estimate an appropriate value.

    Returns
    -------
    xr.Dataset
      Dataset with containing data variables of requested ensemble statistics

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


def _ens_checktimes(
    datasets: List[Union[xr.Dataset, Path, str, List[Union[Path, str]]]],
    mf_flag: bool = False,
) -> Tuple[bool, np.ndarray]:
    """Check list of xarray Datasets and determine if they hava a time dimension. If present, returns the
    maximum time-step interval of all input files.

    Parameters
    ----------
    datasets : List[Union[xr.Dataset, Path, str, List[Path, str]]]
      List of netcdf file paths or xr.DataSet objects . If mf_flag is True, ncfiles should be a list of lists where
      each sublist contains input .nc files of an xarray multifile Dataset.
    mf_flag : bool
      If True climate simulations are treated as xarray multifile Datasets before concatenation.
      Only applicable when :datasets: is a sequence of file paths.

    Returns
    -------
    bool
      True if time dimension is present in the dataset list; Otherwise False.
    array of datetime64
      Series of unique time-steps covering all input datasets.
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


def _ens_align_datasets(
    datasets: List[Union[xr.Dataset, Path, str, List[Union[Path, str]]]],
    mf_flag: bool = False,
    time_flag: bool = False,
    time_all=None,
) -> xr.Dataset:
    """Create a list of aligned xarray Datasets for ensemble Dataset creation. If (time_flag == True), input Datasets
    are given a common time dimension defined by "time_all". Datasets not covering the entire time span have their data
    padded with NaN values

    Parameters
    ----------
    datasets : List[Union[xr.Dataset, Path, str, List[Path, str]]]
      List of netcdf file paths or xarray Dataset objects . If mf_flag is True, ncfiles should be a list of lists where
      each sublist contains input .nc files of an xarray multifile Dataset.
    mf_flag : bool
      If True climate simulations are treated as xarray multifile datasets before concatenation.
      Only applicable when datasets is a sequence of file paths.
    time_flag : bool
      True if time dimension is present among the "datasets"; Otherwise false.
    time_all : array of datetime64
      Series of unique time-steps covering all input Datasets.

    Returns
    -------
    List[xr.Dataset]
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
    data: xr.DataArray,
    *,
    method: dict = None,
    make_graph: bool = MPL_INSTALLED,
    max_clusters: Optional[int] = None,
    variable_weights: Optional[np.ndarray] = None,
    model_weights: Optional[np.ndarray] = None,
    sample_weights: Optional[np.ndarray] = None,
    random_state: Optional[Union[int, np.random.RandomState]] = None
) -> Tuple[list, np.ndarray, dict]:
    """Return a sample of ensemble members using k-means clustering. The algorithm attempts to
    reduce the total number of ensemble members while maintaining adequate coverage of the ensemble
    uncertainty in a N-dimensional data space. K-Means clustering is carried out on the input
    selection criteria data-array in order to group individual ensemble members into a reduced number of similar groups.
    Subsequently a single representative simulation is retained from each group.


    Parameters
    ----------
    data : xr.DataArray
      Selecton criteria data : 2-D xr.DataArray with dimensions 'realization' (N) and
      'criteria' (P). These are the values used for clustering. Realizations represent the individual original
      ensemble members and criteria the variables/indicators used in the grouping algorithm.
    method : dict
      Dictionary defining selection method and associated value when required. See Notes.
    max_clusters : Optional[int]
      Maximum number of members to include in the output ensemble selection.
      When using 'rsq_optimize' or 'rsq_cutoff' methods, limit the final selection to a maximum number even if method
      results indicate a higher value. Defaults to N.
    variable_weights: Optional[np.ndarray]
      An array of size P. This weighting can be used to influence of weight of the climate indices (criteria dimension)
      on the clustering itself.
    model_weights: Optional[np.ndarray]
      An array of size N. This weighting can be used to influence which realization is selected
      from within each cluster. This parameter has no influence on the clustering itself.
    sample_weights: Optional[np.ndarray]
      An array of size N. sklearn.cluster.KMeans() sample_weights parameter. This weighting can be
      used to influence of weight of simulations on the clustering itself.
      See: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
    random_state: Optional[Union[int, np.random.RandomState]]
      sklearn.cluster.KMeans() random_state parameter. Determines random number generation for centroid
      initialization. Use an int to make the randomness deterministic.
      See: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
    make_graph: bool
      output a dictionary of input for displays a plot of R² vs. the number of clusters

    Notes
    -----
    Parameters for method in call must follow these conventions:

    rsq_optimize
        Calculate coefficient of variation (R²) of cluster results for n = 1 to N clusters and determine
        an optimal number of clusters that balances cost / benefit tradeoffs. This is the default setting.
        See supporting information S2 text in https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0152495

        method={'rsq_optimize':None}

    rsq_cutoff
        Calculate Coefficient of variation (R²) of cluster results for n = 1 to N clusters and determine
        the minimum numbers of clusters needed for R² > val.

        val : float between 0 and 1. R² value that must be exceeded by clustering results.

        method={'rsq_cutoff': val}

    n_clusters
        Create a user determined number of clusters.

        val : integer between 1 and N

        method={'n_clusters': val}

    Returns
    -------
    list
      Selected model indexes (positions)
    np.ndarray
      KMeans clustering results
    dict
      Dictionary of input data for creating R² profile plot. 'None' when make_graph=False

    References
    -----
    Casajus et al. 2016. https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0152495


    Examples
    --------
    >>> from xclim import ensembles
    # Start with ensemble datasets for temperature and precipitation
    >>> ensTas = ensembles.create_ensemble(temperature_datasets)
    >>> ensPr = ensembles.create_ensemble(precip_datasets)
    # Calculate selection criteria -- Use annual climate change Δ fields between 2071-2100 and 1981-2010 normals
    # Total annual precipation
    >>> HistPr = ensPr.pr.sel(time=slice('1981','2010')).sum(dim='time').mean(dim=['lat','lon'])
    >>> FutPr = ensPr.pr.sel(time=slice('2071','2100')).sum(dim='time').mean(dim=['lat','lon'])
    >>> dPr = 100*((FutPr / HistPr) - 1)  # expressed in percent change
    # Average annual temperature
    >>> HistTas = ensTas.tas.sel(time=slice('1981','2010')).mean(dim=['time','lat','lon'])
    >>> FutTas = ensTas.tas.sel(time=slice('2071','2100')).mean(dim=['time','lat','lon'])
    >>> dTas = FutTas - HistTas
    # Create selection criteria xr.DataArray
    >>> crit = xr.concat((dTas,dPr), dim='criteria')
    # Create clusters and select realization ids of reduced ensemble
    >>> [ids, cluster, fig_data] = ensembles.kmeans_reduce_ensemble(data=crit, method={'rsq_cutoff':0.9}, random_state=42, make_graph=False)
    >>> [ids, cluster, fig_data] = ensembles.kmeans_reduce_ensemble(data=crit, method={'rsq_optimize':None}, random_state=42, make_graph=True)
    """
    if make_graph:
        fig_data = {}
        if max_clusters is not None:
            fig_data["max_clusters"] = max_clusters

    data = data.transpose("realization", "criteria")
    # initialize the variables
    n_sim = np.shape(data)[0]  # number of simulations
    n_idx = np.shape(data)[1]  # number of indicators

    # normalize the data matrix
    z = xr.DataArray(
        scipy.stats.zscore(data, axis=0, ddof=1), coords=data.coords
    )  # ddof=1 to be the same as Matlab's zscore

    if sample_weights is None:
        sample_weights = np.ones(n_sim)
    else:
        # KMeans sample weights of zero cause errors occasionally - set to 1e-15 for now
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

    n_clusters = _get_nclust(method, n_sim, rsq, max_clusters)

    if make_graph:

        fig_data["method"] = method
        fig_data["rsq"] = rsq
        fig_data["n_clusters"] = n_clusters
        fig_data["realizations"] = n_sim

    else:
        fig_data = None

    # Final k-means clustering with 1000 iterations to avoid instabilities in the choice of final scenarios
    kmeans = KMeans(
        n_clusters=n_clusters, n_init=1000, max_iter=600, random_state=random_state
    )
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
        if d_i.shape[0] >= 2:
            if d_i.shape[0] == 2:
                sig = 1
            else:
                sig = np.std(
                    d_i, ddof=1
                )  # standard deviation of those distances (ddof = 1 gives the same as Matlab's std function)

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

    return out, clusters, fig_data


def _calc_rsq(z, method, make_graph, n_sim, random_state, sample_weights):
    """Subfunction to kmeans_reduce_ensemble. Calculates r-square profile (r-square versus number of clusters."""
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


def _get_nclust(method=None, n_sim=None, rsq=None, max_clusters=None):
    """Subfunction to kmeans_reduce_ensemble. Determine number of clusters to create depending on various methods."""

    # if we actually need to find the optimal number of clusters, this is where it is done
    if list(method.keys())[0] == "rsq_cutoff":
        # argmax finds the first occurence of rsq > rsq_cutoff,but we need to add 1 b/c of python indexing
        n_clusters = np.argmax(rsq > method["rsq_cutoff"]) + 1

    elif list(method.keys())[0] == "rsq_optimize":
        # create constant benefits curve (one to one)
        onetoone = -1 * (1.0 / (n_sim - 1)) + np.arange(1, n_sim + 1) * (
            1.0 / (n_sim - 1)
        )

        n_clusters = np.argmax(rsq - onetoone) + 1

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


def plot_rsqprofile(fig_data):
    """Create an R² profile plot using kmeans_reduce_ensemble output. The R² plot allows evaluation of the proportion
    of total uncertainty in the original ensemble that is provided by the reduced selected.

    Examples
    --------
    >>> [ids, cluster, fig_data] = ensembles.kmeans_reduce_ensemble(data=crit, method={'rsq_cutoff':0.9}, random_state=42, make_graph=False)
    >>> plot_rsqprofile(fig_data)
    """

    rsq = fig_data["rsq"]
    n_sim = fig_data["realizations"]
    n_clusters = fig_data["n_clusters"]
    # make a plot of rsq profile
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, n_sim + 1), rsq, "k-o", label="R²", linewidth=0.8, markersize=4)
    # plt.plot(np.arange(1.5, n_sim + 0.5), np.diff(rsq), 'r', label='ΔR²')
    axes = plt.gca()
    axes.set_xlim([0, fig_data["realizations"]])
    axes.set_ylim([0, 1])
    plt.xlabel("Number of groups")
    plt.ylabel("R²")
    plt.legend(loc="lower right")
    plt.title("R² of groups vs. full ensemble")
    if "rsq_cutoff" in fig_data["method"].keys():
        col = "k--"
        label = "R² selection > {rsq_cut} (n = {n_clusters})".format(
            rsq_cut=fig_data["method"]["rsq_cutoff"], n_clusters=n_clusters
        )

        if "max_clusters" in fig_data.keys():

            if rsq[n_clusters - 1] < fig_data["method"]["rsq_cutoff"]:
                col = "r--"
                label = "R² selection = {rsq} (n = {n_clusters}) : Max cluster set to {max_clust}".format(
                    rsq=rsq[n_clusters - 1].round(2),
                    n_clusters=n_clusters,
                    max_clust=fig_data["max_clusters"],
                )
            else:
                label = "R² selection > {rsq_cut} (n = {n_clusters}) : Max cluster set to {max_clust}".format(
                    rsq_cut=fig_data["method"]["rsq_cutoff"],
                    n_clusters=n_clusters,
                    max_clust=fig_data["max_clusters"],
                )

        plt.plot(
            (0, n_clusters, n_clusters),
            (rsq[n_clusters - 1], rsq[n_clusters - 1], 0),
            col,
            label=label,
            linewidth=0.75,
        )
        plt.legend(loc="lower right")
    elif "rsq_optimize" in fig_data["method"].keys():
        onetoone = -1 * (1.0 / (n_sim - 1)) + np.arange(1, n_sim + 1) * (
            1.0 / (n_sim - 1)
        )
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
        col = "k--"
        label = "Optimized R² cost / benefit (n = {n_clusters})".format(
            n_clusters=n_clusters
        )
        if "max_clusters" in fig_data.keys():
            opt = rsq - onetoone
            imax = np.where(opt == opt.max())[0]

            if rsq[n_clusters - 1] < rsq[imax]:
                col = "r--"
            label = "R² selection = {rsq} (n = {n_clusters}) : Max cluster set to {max_clust}".format(
                rsq=rsq[n_clusters - 1].round(2),
                n_clusters=n_clusters,
                max_clust=fig_data["max_clusters"],
            )

        plt.plot(
            (0, n_clusters, n_clusters),
            (rsq[n_clusters - 1], rsq[n_clusters - 1], 0),
            col,
            linewidth=0.75,
            label=label,
        )
        plt.legend(loc="center right")
    else:
        plt.plot(
            (0, n_clusters, n_clusters),
            (rsq[n_clusters - 1], rsq[n_clusters - 1], 0),
            "k--",
            label="n = {n_clusters} (R² selection = {rsq})".format(
                rsq=rsq[n_clusters - 1].round(2), n_clusters=n_clusters
            ),
            linewidth=0.75,
        )
        plt.legend(loc="lower right")

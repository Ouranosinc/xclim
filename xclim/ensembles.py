"""Ensembles Module."""
import logging
import warnings
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import scipy.stats
import xarray as xr
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans

from xclim.core.calendar import convert_calendar, get_calendar
from xclim.core.formatting import update_history

# Avoid having to include matplotlib in xclim requirements
try:
    from matplotlib import pyplot as plt

    logging.info("Matplotlib installed. Setting make_graph to True.")
    MPL_INSTALLED = True

except ImportError:
    logging.info("Matplotlib not found. No graph data will be produced.")
    MPL_INSTALLED = False


def create_ensemble(
    datasets: List[Union[xr.Dataset, xr.DataArray, Path, str, List[Union[Path, str]]]],
    mf_flag: bool = False,
    resample_freq: Optional[str] = None,
    calendar: str = "default",
    **xr_kwargs,
) -> xr.Dataset:
    """Create an xarray dataset of an ensemble of climate simulation from a list of netcdf files.

    Input data is concatenated along a newly created data dimension ('realization'). Returns an xarray dataset object
    containing input data from the list of netcdf files concatenated along a new dimension (name:'realization').
    In the case where input files have unequal time dimensions, the output ensemble Dataset is created for maximum
    time-step interval of all input files.  Before concatenation, datasets not covering the entire time span have
    their data padded with NaN values. Dataset and variable attributes of the first dataset are copied to the
    resulting dataset.

    Parameters
    ----------
    datasets : List[Union[xr.Dataset, Path, str, List[Path, str]]]
      List of netcdf file paths or xarray Dataset/DataArray objects . If mf_flag is True, ncfiles should be a list of lists where
      each sublist contains input .nc files of an xarray multifile Dataset.
      If DataArray object are passed, they should have a name in order to be transformed into Datasets.

    mf_flag : bool
      If True, climate simulations are treated as xarray multifile Datasets before concatenation.
      Only applicable when "datasets" is a sequence of file paths.

    resample_freq : Optional[str]
      If the members of the ensemble have the same frequency but not the same offset, they cannot be properly aligned.
      If resample_freq is set, the time coordinate of each members will be modified to fit this frequency.

    calendar : str
      The calendar of the time coordinate of the ensemble. For conversions involving '360_day', the align_on='date' option is used.
      See `xclim.core.calendar.convert_calendar`. 'default' is the standard calendar using np.datetime64 objects.

    xr_kwargs :
      Any keyword arguments to be given to `xr.open_dataset` when opening the files (or to `xr.open_mfdataset` if mf_flag is True)

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
    >>> from xclim.ensembles import create_ensemble  # doctest: +SKIP
    >>> ens = create_ensemble(temperature_datasets)  # doctest: +SKIP

    Using multifile datasets:
    Simulation 1 is a list of .nc files (e.g. separated by time)
    >>> datasets = glob.glob('/dir/*.nc')  # doctest: +SKIP

    Simulation 2 is also a list of .nc files
    >>> datasets.append(glob.glob('/dir2/*.nc'))  # doctest: +SKIP
    >>> ens = create_ensemble(datasets, mf_flag=True)  # doctest: +SKIP
    """
    ds = _ens_align_datasets(
        datasets, mf_flag, resample_freq, calendar=calendar, **xr_kwargs
    )

    dim = xr.IndexVariable("realization", np.arange(len(ds)), attrs={"axis": "E"})

    ens = xr.concat(ds, dim)
    for vname, var in ds[0].variables.items():
        ens[vname].attrs.update(**var.attrs)
    ens.attrs.update(**ds[0].attrs)

    return ens


def ensemble_mean_std_max_min(ens: xr.Dataset) -> xr.Dataset:
    """Calculate ensemble statistics between a results from an ensemble of climate simulations.

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
    >>> from xclim.ensembles import create_ensemble, ensemble_mean_std_max_min

    # Create ensemble dataset
    >>> ens = create_ensemble(temperature_datasets)

    # Calculate ensemble statistics
    >>> ens_mean_std = ensemble_mean_std_max_min(ens)
    """
    ds_out = xr.Dataset(attrs=ens.attrs)
    for v in ens.data_vars:

        ds_out[f"{v}_mean"] = ens[v].mean(dim="realization")
        ds_out[f"{v}_stdev"] = ens[v].std(dim="realization")
        ds_out[f"{v}_max"] = ens[v].max(dim="realization")
        ds_out[f"{v}_min"] = ens[v].min(dim="realization")
        for vv in ds_out.data_vars:
            ds_out[vv].attrs = ens[v].attrs
            if "description" in ds_out[vv].attrs.keys():
                vv.split()
                ds_out[vv].attrs["description"] = (
                    ds_out[vv].attrs["description"]
                    + " : "
                    + vv.split("_")[-1]
                    + " of ensemble"
                )
    ds_out.attrs["xclim_history"] = update_history(
        f"Computation of statistics on {ens.realization.size} ensemble members.", ds_out
    )
    return ds_out


def ensemble_percentiles(
    ens: Union[xr.Dataset, xr.DataArray],
    values: Sequence[int] = (10, 50, 90),
    keep_chunk_size: Optional[bool] = None,
    split: bool = True,
) -> xr.Dataset:
    """Calculate ensemble statistics between a results from an ensemble of climate simulations.

    Returns a Dataset containing ensemble percentiles for input climate simulations.

    Parameters
    ----------
    ens: Union[xr.Dataset, xr.DataArray]
      Ensemble dataset or dataarray (see xclim.ensembles.create_ensemble).
    values : Tuple[int, int, int]
      Percentile values to calculate. Default: (10, 50, 90).
    keep_chunk_size : Optional[bool]
      For ensembles using dask arrays, all chunks along the 'realization' axis are merged.
      If True, the dataset is rechunked along the dimension with the largest chunks, so that the chunks keep the same size (approx)
      If False, no shrinking is performed, resulting in much larger chunks
      If not defined, the function decides which is best
    split : bool
      Whether to split each percentile into a new variable of concatenate the ouput along a new
      "percentiles" dimension.

    Returns
    -------
    Union[xr.Dataset, xr.DataArray]
      If split is True, same type as ens; dataset otherwise,
      containing data variable(s) of requested ensemble statistics

    Examples
    --------
    >>> from xclim.ensembles import create_ensemble, ensemble_percentiles

    # Create ensemble dataset
    >>> ens = create_ensemble(temperature_datasets)

    # Calculate default ensemble percentiles
    >>> ens_percs = ensemble_percentiles(ens)

    # Calculate non-default percentiles (25th and 75th)
    >>> ens_percs = ensemble_percentiles(ens, values=(25, 50, 75))

    # If the original array has many small chunks, it might be more efficient to do:
    >>> ens_percs = ensemble_percentiles(ens, keep_chunk_size=False)
    """
    if isinstance(ens, xr.Dataset):
        out = xr.merge(
            [
                ensemble_percentiles(
                    da, values, keep_chunk_size=keep_chunk_size, split=split
                )
                for da in ens.data_vars.values()
                if "realization" in da.dims
            ]
        )
        out.attrs.update(ens.attrs)
        out.attrs["xclim_history"] = update_history(
            f"Computation of the percentiles on {ens.realization.size} ensemble members.",
            ens,
        )

        return out

    # Percentile calculation forbids any chunks along realization
    if ens.chunks and len(ens.chunks[ens.get_axis_num("realization")]) > 1:
        if keep_chunk_size is None:
            # Enable smart rechunking is chunksize exceed 2E8 elements after merging along realization
            keep_chunk_size = (
                np.prod(ens.isel(realization=0).data.chunksize) * ens.realization.size
                > 2e8
            )
        if keep_chunk_size:
            # Smart rechunk on dimension where chunks are the largest
            chkDim, chks = max(
                enumerate(ens.chunks),
                key=lambda kv: 0
                if kv[0] == ens.get_axis_num("realization")
                else max(kv[1]),
            )
            ens = ens.chunk(
                {"realization": -1, ens.dims[chkDim]: len(chks) * ens.realization.size}
            )
        else:
            ens = ens.chunk({"realization": -1})

    out = xr.apply_ufunc(
        _calc_perc,
        ens,
        input_core_dims=[["realization"]],
        output_core_dims=[["percentiles"]],
        keep_attrs=True,
        kwargs=dict(p=values),
        dask="parallelized",
        output_dtypes=[ens.dtype],
        output_sizes={"percentiles": len(values)},
    )

    out = out.assign_coords(
        percentiles=xr.DataArray(list(values), dims=("percentiles",))
    )

    if split:
        out = out.to_dataset(dim="percentiles")
        for p, perc in out.data_vars.items():
            perc.attrs.update(ens.attrs)
            perc.attrs["description"] = (
                perc.attrs.get("description", "") + f" {p}th percentile of ensemble."
            )
            out[p] = perc
            out = out.rename(name_dict={p: f"{ens.name}_p{int(p):02d}"})

    out.attrs["xclim_history"] = update_history(
        f"Computation of the percentiles on {ens.realization.size} ensemble members.",
        ens,
    )

    return out


def _ens_align_datasets(
    datasets: List[Union[xr.Dataset, Path, str, List[Union[Path, str]]]],
    mf_flag: bool = False,
    resample_freq: str = None,
    calendar: str = "default",
    **xr_kwargs,
) -> List[xr.Dataset]:
    """Create a list of aligned xarray Datasets for ensemble Dataset creation.

    Parameters
    ----------
    datasets : List[Union[xr.Dataset, xr.DataArray, Path, str, List[Path, str]]]
      List of netcdf file paths or xarray Dataset/DataArray objects . If mf_flag is True, ncfiles should be a list of lists where
      each sublist contains input .nc files of an xarray multifile Dataset. DataArrays should have a name so they can be converted to datasets.
    mf_flag : bool
      If True climate simulations are treated as xarray multifile datasets before concatenation.
      Only applicable when datasets is a sequence of file paths.
    resample_freq : Optional[str]
      If the members of the ensemble have the same frequency but not the same offset, they cannot be properly aligned.
      If resample_freq is set, the time coordinate of each members will be modified to fit this frequency.
    calendar : str
      The calendar of the time coordinate of the ensemble. For conversions involving '360_day', the align_on='date' option is used.
      See `xclim.core.calendar.convert_calendar`. 'default' is the standard calendar using np.datetime64 objects.
    xr_kwargs :
      Any keyword arguments to be given to xarray when opening the files.

    Returns
    -------
    List[xr.Dataset]
    """
    xr_kwargs.setdefault("chunks", {"time": 10})
    xr_kwargs.setdefault("decode_times", False)

    ds_all = []
    for i, n in enumerate(datasets):
        logging.info(f"Accessing {n} of {len(datasets)}")
        if mf_flag:
            ds = xr.open_mfdataset(n, combine="by_coords", **xr_kwargs)
        else:
            if isinstance(n, xr.Dataset):
                ds = n
            elif isinstance(n, xr.DataArray):
                ds = n.to_dataset()
            else:
                ds = xr.open_dataset(n, **xr_kwargs)

        if "time" in ds.coords:
            time = xr.decode_cf(ds).time

            if resample_freq is not None:
                counts = time.resample(time=resample_freq).count()
                if any(counts > 1):
                    raise ValueError(
                        f"Alignment of dataset #{i:02d} failed : its time axis cannot be resampled to freq {resample_freq}."
                    )
                time = counts.time

            ds["time"] = time

            cal = get_calendar(time)
            ds = convert_calendar(
                ds,
                calendar,
                align_on="date" if "360_day" in [cal, calendar] else None,
            )

        ds_all.append(ds)

    return ds_all


def _calc_perc(arr, p=[50]):
    """Ufunc-like computing a percentile over the last axis of the array.

    Processes cases with invalid values separately, which makes it more efficent than np.nanpercentile for array with only a few invalid points.

    Parameters
    ----------
    arr : np.array
        Percentile is computed over the last axis.
    p : scalar
        Percentile to compute, between 0 and 100. (the default is 50)

    Returns
    -------
    np.array
    """
    nan_count = np.isnan(arr).sum(axis=-1)
    out = np.moveaxis(np.percentile(arr, p, axis=-1), 0, -1)
    nans = (nan_count > 0) & (nan_count < arr.shape[-1])
    if np.any(nans):
        out_mask = np.stack([nans] * len(p), axis=-1)
        # arr1 = arr.reshape(int(arr.size / arr.shape[-1]), arr.shape[-1])
        # only use nanpercentile where we need it (slow performance compared to standard) :
        out[out_mask] = np.moveaxis(
            np.nanpercentile(arr[nans], p, axis=-1), 0, -1
        ).ravel()
    return out


def kkz_reduce_ensemble(
    data: xr.DataArray,
    num_select: int,
    *,
    dist_method: str = "euclidean",
    standardize: bool = True,
    **cdist_kwargs,
) -> list:
    """Return a sample of ensemble members using KKZ selection.

    The algorithm selects `num_select` ensemble members spanning the overall range of the ensemble.
    The selection is ordered, smaller groups are always subsets of larger ones for given criteria.
    The first selected member is the one nearest to the centroid of the ensemble, all subsequent members
    are selected in a way maximizing the phase-space coverage of the group. Algorithm taken from [CannonKKZ]_.

    Parameters
    ----------
    data : xr.DataArray
      Selecton criteria data : 2-D xr.DataArray with dimensions 'realization' (N) and
      'criteria' (P). These are the values used for clustering. Realizations represent the individual original
      ensemble members and criteria the variables/indicators used in the grouping algorithm.
    num_select : int
      The number of members to select.
    dist_method : str
      Any distance metric name accepted by `scipy.spatial.distance.cdist`.
    standardize : bool
      Whether to standardize the input before running the selection or not.
      Standardization consists in translation as to have a zero mean and scaling as to have a unit standard deviation.
    **cdist_kwargs
      All extra arguments are passed as-is to `scipy.spatial.distance.cdist`, see its docs for more information.

    Returns
    -------
    list
        Selected model indices along the `realization` dimension.

    References
    ----------
    .. [CannonKKZ] Cannon, Alex J. (2015). Selecting GCM Scenarios that Span the Range of Changes in a Multimodel Ensemble: Application to CMIP5 Climate Extremes Indices. Journal of Climate, (28)3, 1260-1267. https://doi.org/10.1175/JCLI-D-14-00636.1
    .. Kastsavounidis, I, Kuo, C.-C. Jay, Zhang, Zhen (1994). A new initialization technique for generalized Lloyd iteration. IEEE Signal Processing Letters, 1(10), 144-146. https://doi.org/10.1109/97.329844
    """
    if standardize:
        data = (data - data.mean("realization")) / data.std("realization")

    data = data.transpose("realization", "criteria")
    data["realization"] = np.arange(data.realization.size)

    unselected = list(data.realization.values)
    selected = []

    dist0 = cdist(
        data.mean("realization").expand_dims("realization"),
        data,
        metric=dist_method,
        **cdist_kwargs,
    )
    selected.append(unselected.pop(dist0.argmin()))

    for i in range(1, num_select):
        dist = cdist(
            data.isel(realization=selected),
            data.isel(realization=unselected),
            metric=dist_method,
            **cdist_kwargs,
        )
        dist = dist.min(axis=0)
        selected.append(unselected.pop(dist.argmax()))

    return selected


def kmeans_reduce_ensemble(
    data: xr.DataArray,
    *,
    method: dict = None,
    make_graph: bool = MPL_INSTALLED,
    max_clusters: Optional[int] = None,
    variable_weights: Optional[np.ndarray] = None,
    model_weights: Optional[np.ndarray] = None,
    sample_weights: Optional[np.ndarray] = None,
    random_state: Optional[Union[int, np.random.RandomState]] = None,
) -> Tuple[list, np.ndarray, dict]:
    """Return a sample of ensemble members using k-means clustering.

    The algorithm attempts to reduce the total number of ensemble members while maintaining adequate coverage of
    the ensemble uncertainty in a N-dimensional data space. K-Means clustering is carried out on the input
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
      output a dictionary of input for displays a plot of R² vs. the number of clusters.
      Defaults to True if matplotlib is installed in runtime environment.

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
    ----------
    Casajus et al. 2016. https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0152495

    Examples
    --------
    >>> from xclim.ensembles import create_ensemble, kmeans_reduce_ensemble
    >>> from xclim.indicators.atmos import tg_mean, hot_spell_frequency

    # Start with ensemble datasets for temperature
    >>> ensTas = create_ensemble(temperature_datasets)

    # Calculate selection criteria -- Use annual climate change Δ fields between 2071-2100 and 1981-2010 normals
    # Average annual temperature
    >>> tg = tg_mean(tas=ensTas.tas)
    >>> his_tg = tg.sel(time=slice('1990','2019')).mean(dim='time')
    >>> fut_tg = tg.sel(time=slice('2020','2050')).mean(dim='time')
    >>> dtg = fut_tg - his_tg

    # Hotspell frequency as second indicator
    >>> hs = hot_spell_frequency(tasmax=ensTas.tas, window=2, thresh_tasmax='10 degC')
    >>> his_hs = hs.sel(time=slice('1990','2019')).mean(dim='time')
    >>> fut_hs = hs.sel(time=slice('2020','2050')).mean(dim='time')
    >>> dhs = fut_hs - his_hs

    # Create selection criteria xr.DataArray
    >>> crit = xr.concat((dtg, dhs), dim='criteria')

    # Create clusters and select realization ids of reduced ensemble
    >>> ids, cluster, fig_data = kmeans_reduce_ensemble(data=crit, method={'rsq_cutoff':0.9}, random_state=42, make_graph=False)
    >>> ids, cluster, fig_data = kmeans_reduce_ensemble(data=crit, method={'rsq_optimize':None}, random_state=42, make_graph=True)
    """
    if make_graph:
        fig_data = {}
        if max_clusters is not None:
            fig_data["max_clusters"] = max_clusters
    else:
        fig_data = None

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
        raise Exception(f"Unknown selection method : {list(method.keys())}")
    if n_clusters > max_clusters:
        warnings.warn(
            f"{n_clusters} clusters has been found to be the optimal number of clusters, but limiting "
            f"to {max_clusters} as required by user provided max_clusters",
            UserWarning,
            stacklevel=2,
        )
        n_clusters = max_clusters
    return n_clusters


def plot_rsqprofile(fig_data):
    """Create an R² profile plot using kmeans_reduce_ensemble output.

    The R² plot allows evaluation of the proportion of total uncertainty in the original ensemble that is provided
    by the reduced selected.

    Examples
    --------
    >>> from xclim.ensembles import kmeans_reduce_ensemble, plot_rsqprofile
    >>> is_matplotlib_installed()
    >>> crit = xr.open_dataset(path_to_ensemble_file).data
    >>> ids, cluster, fig_data = kmeans_reduce_ensemble(data=crit, method={'rsq_cutoff':0.9}, random_state=42, make_graph=True)
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
        label = f"R² selection > {fig_data['method']['rsq_cutoff']} (n = {n_clusters})"
        if "max_clusters" in fig_data.keys():

            if rsq[n_clusters - 1] < fig_data["method"]["rsq_cutoff"]:
                col = "r--"
                label = (
                    f"R² selection = {rsq[n_clusters - 1].round(2)} (n = {n_clusters}) :"
                    f" Max cluster set to {fig_data['max_clusters']}"
                )
            else:
                label = (
                    f"R² selection > {fig_data['method']['rsq_cutoff']} (n = {n_clusters}) :"
                    f" Max cluster set to {fig_data['max_clusters']}"
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
        label = f"Optimized R² cost / benefit (n = {n_clusters})"
        if "max_clusters" in fig_data.keys():
            opt = rsq - onetoone
            imax = np.where(opt == opt.max())[0]

            if rsq[n_clusters - 1] < rsq[imax]:
                col = "r--"
            label = (
                f"R² selection = {rsq[n_clusters - 1].round(2)} (n = {n_clusters}) :"
                f" Max cluster set to {fig_data['max_clusters']}"
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
            label=f"n = {n_clusters} (R² selection = {rsq[n_clusters - 1].round(2)})",
            linewidth=0.75,
        )
        plt.legend(loc="lower right")

"""
Ensemble Reduction
==================

Ensemble reduction is the process of selecting a subset of members from an ensemble in
order to reduce the volume of computation needed while still covering a good portion of
the simulated climate variability.
"""
from __future__ import annotations

from warnings import warn

import numpy as np
import pandas as pd
import scipy.stats
import xarray
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans

# Avoid having to include matplotlib in xclim requirements
try:
    from matplotlib import pyplot as plt  # noqa

    MPL_INSTALLED = True
except ImportError:
    plt = None
    MPL_INSTALLED = False


def make_criteria(ds: xarray.Dataset | xarray.DataArray):
    """Reshapes the input into a criteria 2D DataArray.

    The reshaping preserves the "realization" dimension but stacks all other
    dimensions and variables into a new "criteria" dimension, as expected by
    functions :py:func:`xclim.ensembles._reduce.kkz_reduce_ensemble`
    and :py:func:`xclim.ensembles._reduce.kmeans_reduce_ensemble`.

    Parameters
    ----------
    ds : Dataset or DataArray
        Must at least have a "realization" dimension. All values are considered independent "criterion" for the ensemble reduction.
        If a Dataset, variables may have different sizes, but all must include the "realization" dimension.

    Returns
    -------
    crit : DataArray
        Same data, reshaped. Old coordinates (and variables) are available as a multiindex.

    Notes
    -----
    One can get back to the original dataset with

    .. code-block:: python

        crit = make_criteria(ds)
        ds2 = crit.unstack("criteria").to_dataset("variables")

    `ds2` will have all variables with the same dimensions, so if the original dataset had variables with different dimensions, the
    added dimensions are filled with NaNs. The `to_dataset` part can be skipped if the original input was a DataArray.
    """

    def _make_crit(da):
        """Variable version : stack non-realization dims."""
        return da.stack(criteria=set(da.dims) - {"realization"})

    if isinstance(ds, xarray.Dataset):
        # When variables have different set of dims, missing dims on one variable results in duplicated values when a simple stack is done.
        # To avoid that: stack each variable independently add a new "variables" dim
        stacked = {
            da.name: _make_crit(da.expand_dims(variables=[da.name]))
            for da in ds.data_vars.values()
        }
        # Get name of all stacked coords
        stacked_coords = set.union(
            *[set(da.indexes["criteria"].names) for da in stacked.values()]
        )
        # Concat the variables by dropping old stacked index and related coords
        crit = xarray.concat(
            [
                da.reset_index("criteria").drop_vars(stacked_coords, errors="ignore")
                for k, da in stacked.items()
            ],
            "criteria",
        )
        # Reconstruct proper stacked coordinates. When a variable is missing one of the coords, give NaNss
        coords = [
            (
                crd,
                np.concatenate(
                    [
                        da[crd].values
                        if crd in da.coords
                        else [np.NaN] * da.criteria.size
                        for da in stacked.values()
                    ],
                ),
            )
            for crd in stacked_coords
        ]
        crit["criteria"] = pd.MultiIndex.from_arrays(
            [arr for name, arr in coords], names=[name for name, arr in coords]
        )
        # Previous ops gave the first variable's attributes, replace by the original dataset ones.
        crit.attrs = ds.attrs
    else:
        # Easy peasy, skip all the convoluted stuff
        crit = _make_crit(ds)
    return crit.rename("criteria")


def kkz_reduce_ensemble(
    data: xarray.DataArray,
    num_select: int,
    *,
    dist_method: str = "euclidean",
    standardize: bool = True,
    **cdist_kwargs,
) -> list:
    r"""Return a sample of ensemble members using KKZ selection.

    The algorithm selects `num_select` ensemble members spanning the overall range of the ensemble.
    The selection is ordered, smaller groups are always subsets of larger ones for given criteria.
    The first selected member is the one nearest to the centroid of the ensemble, all subsequent members
    are selected in a way maximizing the phase-space coverage of the group.
    Algorithm taken from :cite:t:`cannon_selecting_2015`.

    Parameters
    ----------
    data : xr.DataArray
        Selection criteria data : 2-D xr.DataArray with dimensions 'realization' (N) and
        'criteria' (P). These are the values used for clustering. Realizations represent the individual original
        ensemble members and criteria the variables/indicators used in the grouping algorithm.
    num_select : int
        The number of members to select.
    dist_method : str
        Any distance metric name accepted by `scipy.spatial.distance.cdist`.
    standardize : bool
        Whether to standardize the input before running the selection or not.
        Standardization consists in translation as to have a zero mean and scaling as to have a unit standard deviation.
    \*\*cdist_kwargs
        All extra arguments are passed as-is to `scipy.spatial.distance.cdist`, see its docs for more information.

    Returns
    -------
    list
        Selected model indices along the `realization` dimension.

    References
    ----------
    :cite:cts:`cannon_selecting_2015,katsavounidis_new_1994`
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

    for _ in range(1, num_select):
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
    data: xarray.DataArray,
    *,
    method: dict = None,
    make_graph: bool = MPL_INSTALLED,
    max_clusters: int | None = None,
    variable_weights: np.ndarray | None = None,
    model_weights: np.ndarray | None = None,
    sample_weights: np.ndarray | None = None,
    random_state: int | np.random.RandomState | None = None,
) -> tuple[list, np.ndarray, dict]:
    """Return a sample of ensemble members using k-means clustering.

    The algorithm attempts to reduce the total number of ensemble members while maintaining adequate coverage of
    the ensemble uncertainty in an N-dimensional data space. K-Means clustering is carried out on the input
    selection criteria data-array in order to group individual ensemble members into a reduced number of similar groups.
    Subsequently, a single representative simulation is retained from each group.

    Parameters
    ----------
    data : xr.DataArray
        Selection criteria data : 2-D xr.DataArray with dimensions 'realization' (N) and
        'criteria' (P). These are the values used for clustering. Realizations represent the individual original
        ensemble members and criteria the variables/indicators used in the grouping algorithm.
    method : dict, optional
        Dictionary defining selection method and associated value when required. See Notes.
    max_clusters : int, optional
        Maximum number of members to include in the output ensemble selection.
        When using 'rsq_optimize' or 'rsq_cutoff' methods, limit the final selection to a maximum number even if method
        results indicate a higher value. Defaults to N.
    variable_weights : np.ndarray, optional
        An array of size P. This weighting can be used to influence of weight of the climate indices (criteria dimension)
        on the clustering itself.
    model_weights : np.ndarray, optional
        An array of size N. This weighting can be used to influence which realization is selected
        from within each cluster. This parameter has no influence on the clustering itself.
    sample_weights : np.ndarray, optional
        An array of size N. sklearn.cluster.KMeans() sample_weights parameter. This weighting can be
        used to influence of weight of simulations on the clustering itself.
        See: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
    random_state: int or np.random.RandomState, optional
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
        an optimal number of clusters that balances cost/benefit tradeoffs. This is the default setting.
        See supporting information S2 text in :cite:t:`casajus_objective_2016`.

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
    :cite:cts:`casajus_objective_2016`

    Examples
    --------
    .. code-block::

        import xclim
        from xclim.ensembles import create_ensemble, kmeans_reduce_ensemble
        from xclim.indices import hot_spell_frequency

        # Start with ensemble datasets for temperature:

        ensTas = create_ensemble(temperature_datasets)

        # Calculate selection criteria -- Use annual climate change Δ fields between 2071-2100 and 1981-2010 normals.
        # First, average annual temperature:

        tg = xclim.atmos.tg_mean(tas=ensTas.tas)
        his_tg = tg.sel(time=slice("1990", "2019")).mean(dim="time")
        fut_tg = tg.sel(time=slice("2020", "2050")).mean(dim="time")
        dtg = fut_tg - his_tg

        # Then, hot spell frequency as second indicator:

        hs = hot_spell_frequency(tasmax=ensTas.tas, window=2, thresh_tasmax="10 degC")
        his_hs = hs.sel(time=slice("1990", "2019")).mean(dim="time")
        fut_hs = hs.sel(time=slice("2020", "2050")).mean(dim="time")
        dhs = fut_hs - his_hs

        # Create a selection criteria xr.DataArray:

        from xarray import concat

        crit = concat((dtg, dhs), dim="criteria")

        # Finally, create clusters and select realization ids of reduced ensemble:

        ids, cluster, fig_data = kmeans_reduce_ensemble(
            data=crit, method={"rsq_cutoff": 0.9}, random_state=42, make_graph=False
        )
        ids, cluster, fig_data = kmeans_reduce_ensemble(
            data=crit, method={"rsq_optimize": None}, random_state=42, make_graph=True
        )
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
    std = data.std(dim="realization", ddof=1)
    z = (data - data.mean(dim="realization")) / std.where(std > 0)

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
    """Sub-function to kmeans_reduce_ensemble. Calculates r-square profile (r-square versus number of clusters."""
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
    """Sub-function to kmeans_reduce_ensemble. Determine number of clusters to create depending on various methods."""
    # if we actually need to find the optimal number of clusters, this is where it is done
    if list(method.keys())[0] == "rsq_cutoff":
        # argmax finds the first occurrence of rsq > rsq_cutoff,but we need to add 1 b/c of python indexing
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
        warn(
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
    >>> ids, cluster, fig_data = kmeans_reduce_ensemble(
    ...     data=crit, method={"rsq_cutoff": 0.9}, random_state=42, make_graph=True
    ... )
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

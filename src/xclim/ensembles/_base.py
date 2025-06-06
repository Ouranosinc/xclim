"""
Ensembles Creation and Statistics
=================================
"""

from __future__ import annotations

from collections.abc import Sequence
from glob import glob
from pathlib import Path
from typing import Any, Literal

import numpy as np
import xarray as xr

from xclim.core.calendar import common_calendar, get_calendar
from xclim.core.formatting import update_history
from xclim.core.utils import calc_perc

# The alpha and beta parameters for the quantile function
_quantile_params = {
    "interpolated_inverted_cdf": (0, 1),
    "hazen": (0.5, 0.5),
    "weibull": (0, 0),
    "linear": (1, 1),
    "median_unbiased": (1 / 3, 1 / 3),
    "normal_unbiased": (3 / 8, 3 / 8),
}


def create_ensemble(
    datasets: Any,
    multifile: bool = False,
    resample_freq: str | None = None,
    calendar: str | None = None,
    realizations: Sequence[Any] | None = None,
    cal_kwargs: dict | None = None,
    **xr_kwargs,
) -> xr.Dataset:
    r"""
    Create an xarray dataset of an ensemble of climate simulation from a list of netcdf files.

    Input data is concatenated along a newly created data dimension ('realization'). Returns an xarray dataset object
    containing input data from the list of netcdf files concatenated along a new dimension (name:'realization').
    In the case where input files have unequal time dimensions, the output ensemble Dataset is created for maximum
    time-step interval of all input files.  Before concatenation, datasets not covering the entire time span have
    their data padded with NaN values. Dataset and variable attributes of the first dataset are copied to the
    resulting dataset.

    Parameters
    ----------
    datasets : list or dict or str
        List of netcdf file paths or xarray Dataset/DataArray objects . If `multifile` is True, ncfiles should be a
        list of lists where each sublist contains input .nc files of an xarray multifile Dataset.
        If DataArray objects are passed, they should have a name in order to be transformed into Datasets.
        A dictionary can be passed instead of a list, in which case the keys are used as coordinates along the new
        `realization` axis.
        If a string is passed, it is assumed to be a glob pattern for finding datasets.
    multifile : bool
        If True, climate simulations are treated as xarray multifile Datasets before concatenation.
        Only applicable when "datasets" is sequence of list of file paths. Default: False.
    resample_freq : Optional[str]
        If the members of the ensemble have the same frequency but not the same offset,
        they cannot be properly aligned.
        If resample_freq is set, the time coordinate of each member will be modified to fit this frequency.
    calendar : str, optional
        The calendar of the time coordinate of the ensemble.
        By default, the biggest calendar (in number of days by year) is chosen.
        For example, a mixed input of "noleap" and "360_day" will default to "noleap".
        'default' is the standard calendar using np.datetime64 objects (xarray's "standard" with `use_cftime=False`).
    realizations : sequence, optional
        The coordinate values for the new `realization` axis.
        If None (default), the new axis has a simple integer coordinate.
        This argument shouldn't be used if `datasets` is a glob pattern as the dataset order is random.
    cal_kwargs : dict, optional
        Additional arguments to pass to py:func:`xclim.core.calendar.convert_calendar`.
        For conversions involving '360_day', the align_on='date' option is used by default.
    **xr_kwargs : dict
        Any keyword arguments to be given to `xr.open_dataset` when opening the files
        (or to `xr.open_mfdataset` if `multifile` is True).

    Returns
    -------
    xr.Dataset
        A Dataset containing concatenated data from all input files.

    Notes
    -----
    Input netcdf files require equal spatial dimension size (e.g. lon, lat dimensions).
    If input data contains multiple cftime calendar types they must be at monthly or coarser frequency.

    Examples
    --------
    .. code-block:: python

        from pathlib import Path
        from xclim.ensembles import create_ensemble

        ens = create_ensemble(temperature_datasets)

        # Using multifile datasets, through glob patterns.
        # Simulation 1 is a list of .nc files (e.g. separated by time):
        datasets = list(Path("/dir").glob("*.nc"))

        # Simulation 2 is also a list of .nc files:
        datasets.extend(Path("/dir2").glob("*.nc"))
        ens = create_ensemble(datasets, multifile=True)
    """
    if isinstance(datasets, dict):
        if realizations is None:
            realizations, datasets = zip(*datasets.items(), strict=False)
        else:
            datasets = datasets.values()
    elif isinstance(datasets, str) and realizations is not None:
        raise ValueError(
            "Passing `realizations` is not supported when `datasets` is a glob pattern, as the final order is random."
        )

    ds = _ens_align_datasets(
        datasets,
        multifile,
        resample_freq,
        calendar=calendar,
        cal_kwargs=cal_kwargs or {},
        **xr_kwargs,
    )

    if realizations is None:
        realizations = np.arange(len(ds))

    dim = xr.IndexVariable("realization", list(realizations), attrs={"axis": "E"})

    ens = xr.concat(ds, dim)
    for var_name, var in ds[0].variables.items():
        ens[var_name].attrs.update(**var.attrs)
    ens.attrs.update(**ds[0].attrs)

    return ens


def ensemble_mean_std_max_min(
    ens: xr.Dataset, min_members: int | None = 1, weights: xr.DataArray | None = None
) -> xr.Dataset:
    """
    Calculate ensemble statistics between a results from an ensemble of climate simulations.

    Returns an xarray Dataset containing ensemble mean, standard-deviation, minimum and maximum for input climate
    simulations.

    Parameters
    ----------
    ens : xr.Dataset
        Ensemble dataset (see xclim.ensembles.create_ensemble).
    min_members : int, optional
        The minimum number of valid ensemble members for a statistic to be valid.
        Passing None is equivalent to setting min_members to the size of the realization dimension.
        The default (1) essentially skips this check.
    weights : xr.DataArray, optional
        Weights to apply along the 'realization' dimension. This array cannot contain missing values.

    Returns
    -------
    xr.Dataset
        Dataset with data variables of ensemble statistics.

    Examples
    --------
    .. code-block:: python

        from xclim.ensembles import create_ensemble, ensemble_mean_std_max_min

        # Create the ensemble dataset:
        ens = create_ensemble(temperature_datasets)

        # Calculate ensemble statistics:
        ens_mean_std = ensemble_mean_std_max_min(ens)
    """
    if min_members is None:
        min_members = ens.realization.size
    ds_out = xr.Dataset(attrs=ens.attrs)
    for v in ens.data_vars:
        if weights is None:
            ds_out[f"{v}_mean"] = ens[v].mean(dim="realization")
            ds_out[f"{v}_stdev"] = ens[v].std(dim="realization")
        else:
            with xr.set_options(keep_attrs=True):
                ds_out[f"{v}_mean"] = ens[v].weighted(weights).mean(dim="realization")
                ds_out[f"{v}_stdev"] = ens[v].weighted(weights).std(dim="realization")
        ds_out[f"{v}_max"] = ens[v].max(dim="realization")
        ds_out[f"{v}_min"] = ens[v].min(dim="realization")

        enough = None
        if min_members != 1:
            enough = ens[v].notnull().sum("realization") >= min_members

        # Re-add attributes
        for stat in ["mean", "stdev", "max", "min"]:
            vv = f"{v}_{stat}"
            if min_members != 1:
                ds_out[vv] = ds_out[vv].where(enough)
            ds_out[vv].attrs = ens[v].attrs
            if "description" in ds_out[vv].attrs.keys():
                vv.split()
                ds_out[vv].attrs["description"] = (
                    ds_out[vv].attrs["description"] + " : " + vv.split("_", maxsplit=1)[-1] + " of ensemble"
                )

    ds_out.attrs["history"] = update_history(
        f"Computation of statistics on {ens.realization.size} ensemble members.", ds_out
    )
    return ds_out


def ensemble_percentiles(
    ens: xr.Dataset | xr.DataArray,
    values: Sequence[int] | None = None,
    keep_chunk_size: bool | None = None,
    min_members: int | None = 1,
    weights: xr.DataArray | None = None,
    split: bool = True,
    method: Literal[
        "linear",
        "interpolated_inverted_cdf",
        "hazen",
        "weibull",
        "median_unbiased",
        "normal_unbiased",
    ] = "linear",
) -> xr.DataArray | xr.Dataset:
    """
    Calculate ensemble statistics between a results from an ensemble of climate simulations.

    Returns a Dataset containing ensemble percentiles for input climate simulations.

    Parameters
    ----------
    ens : xr.Dataset or xr.DataArray
        Ensemble Dataset or DataArray (see xclim.ensembles.create_ensemble).
    values : Sequence[int], optional
        Percentile values to calculate. Default: (10, 50, 90).
    keep_chunk_size : bool, optional
        For ensembles using dask arrays, all chunks along the 'realization' axis are merged.
        If True, the dataset is rechunked along the dimension with the largest chunks,
        so that the chunks keep the same size (approximately).
        If False, no shrinking is performed, resulting in much larger chunks.
        If not defined, the function decides which is best.
    min_members : int, optional
        The minimum number of valid ensemble members for a statistic to be valid.
        Passing None is equivalent to setting min_members to the size of the realization dimension.
        The default (1) essentially skips this check.
    weights : xr.DataArray, optional
        Weights to apply along the 'realization' dimension. This array cannot contain missing values.
        When given, the function uses xarray's quantile method which is slower than xclim's NaN-optimized algorithm,
        and does not support `method` values other than `linear`.
    split : bool
        Whether to split each percentile into a new variable
        or concatenate the output along a new "percentiles" dimension.
    method : {"linear", "interpolated_inverted_cdf", "hazen", "weibull", "median_unbiased", "normal_unbiased"}
        Method to use for estimating the percentile, see the `numpy.percentile` documentation for more information.

    Returns
    -------
    xr.Dataset or xr.DataArray
        If split is True, same type as ens;
        Otherwise, a dataset containing data variable(s) of requested ensemble statistics.

    Examples
    --------
    .. code-block:: python

        from xclim.ensembles import create_ensemble, ensemble_percentiles

        # Create ensemble dataset:
        ens = create_ensemble(temperature_datasets)

        # Calculate default ensemble percentiles:
        ens_percs = ensemble_percentiles(ens)

        # Calculate non-default percentiles (25th and 75th)
        ens_percs = ensemble_percentiles(ens, values=(25, 50, 75))

        # If the original array has many small chunks, it might be more efficient to do:
        ens_percs = ensemble_percentiles(ens, keep_chunk_size=False)
    """
    if values is None:
        values = [10, 50, 90]
    if min_members is None:
        min_members = ens.realization.size

    if isinstance(ens, xr.Dataset):
        out = xr.merge(
            [
                ensemble_percentiles(
                    da,
                    values,
                    keep_chunk_size=keep_chunk_size,
                    split=split,
                    min_members=min_members,
                    weights=weights,
                    method=method,
                )
                for da in ens.data_vars.values()
                if "realization" in da.dims
            ]
        )
        out.attrs.update(ens.attrs)
        out.attrs["history"] = update_history(
            f"Computation of the percentiles on {ens.realization.size} ensemble members.",
            ens,
        )

        return out

    # Percentile calculation forbids any chunks along realization
    if ens.chunks and len(ens.chunks[ens.get_axis_num("realization")]) > 1:
        if keep_chunk_size is None:
            # Enable smart rechunking is chunksize exceed 2E8 elements after merging along realization
            keep_chunk_size = np.prod(ens.isel(realization=0).data.chunksize) * ens.realization.size > 2e8
        if keep_chunk_size:
            # Smart rechunk on dimension where chunks are the largest
            chk_dim, chks = max(
                enumerate(ens.chunks),
                key=lambda kv: (0 if kv[0] == ens.get_axis_num("realization") else max(kv[1])),
            )
            ens = ens.chunk({"realization": -1, ens.dims[chk_dim]: len(chks) * ens.realization.size})
        else:
            ens = ens.chunk({"realization": -1})

    if weights is None:
        alpha, beta = _quantile_params[method]

        out = xr.apply_ufunc(
            calc_perc,
            ens,
            input_core_dims=[["realization"]],
            output_core_dims=[["percentiles"]],
            keep_attrs=True,
            kwargs={"percentiles": values, "alpha": alpha, "beta": beta},
            dask="parallelized",
            output_dtypes=[ens.dtype],
            dask_gufunc_kwargs={"output_sizes": {"percentiles": len(values)}},
        )
    else:
        if method != "linear":
            raise ValueError("Only the 'linear' method is supported when using weights.")

        with xr.set_options(keep_attrs=True):
            # xclim's calc_perc does not support weighted arrays, so xarray's native function is used instead.
            qt = np.array(values) / 100  # xarray requires values between 0 and 1
            out = (
                ens.weighted(weights)
                .quantile(qt, dim="realization", keep_attrs=True)
                .rename({"quantile": "percentiles"})
            )

    if min_members != 1:
        out = out.where(ens.notnull().sum("realization") >= min_members)
    out = out.assign_coords(percentiles=xr.DataArray(list(values), dims=("percentiles",)))

    if split:
        out = out.to_dataset(dim="percentiles")
        for p, perc in out.data_vars.items():
            perc.attrs.update(ens.attrs)
            perc.attrs["description"] = perc.attrs.get("description", "") + f" {p}th percentile of ensemble."
            out[p] = perc
            out = out.rename(name_dict={p: f"{ens.name}_p{int(p):02d}"})

    out.attrs["history"] = update_history(
        f"Computation of the percentiles on {ens.realization.size} ensemble members using method {method}.",
        ens,
    )

    return out


def _ens_align_datasets(
    datasets: list[xr.Dataset | Path | str | list[Path | str]] | str,
    multifile: bool = False,
    resample_freq: str | None = None,
    calendar: str = "default",
    cal_kwargs: dict | None = None,
    **xr_kwargs,
) -> list[xr.Dataset]:
    r"""
    Create a list of aligned xarray Datasets for ensemble Dataset creation.

    Parameters
    ----------
    datasets : list[xr.Dataset | xr.DataArray | Path | str | list[Path | str]] or str
        List of netcdf file paths or xarray Dataset/DataArray objects . If `multifile` is True, 'datasets' should be a
        list of lists where each sublist contains input NetCDF files of a xarray multi-file Dataset.
        DataArrays should have a name, so they can be converted to datasets.
        If a string, it is assumed to be a glob pattern for finding datasets.
    multifile : bool
        If True climate simulations are treated as xarray multi-file datasets before concatenation.
        Only applicable when 'datasets' is a sequence of file paths.
    resample_freq : str, optional
        If the members of the ensemble have the same frequency but not the same offset, they cannot be properly aligned.
        If resample_freq is set, the time coordinate of each member will be modified to fit this frequency.
    calendar : str
        The calendar of the time coordinate of the ensemble. For conversions involving '360_day',
        the align_on='date' option is used.
        See :py:func:`xclim.core.calendar.convert_calendar`.
        'default' is the standard calendar using np.datetime64 objects.
    cal_kwargs : dict, optional
        Any keyword to be given to used when setting calendar options.
    **xr_kwargs : dict
        Any keyword arguments to be given to xarray when opening the files.

    Returns
    -------
    list[xr.Dataset]
    """
    xr_kwargs.setdefault("chunks", "auto")
    xr_kwargs.setdefault("decode_times", False)

    if isinstance(datasets, str):
        datasets = glob(datasets)

    ds_all: list[xr.Dataset] = []
    calendars = []
    for i, n in enumerate(datasets):
        ds: xr.Dataset
        if multifile:
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
                # Cast to bool to avoid bug in flox/numpy_groupies (xarray-contrib/flox#137)
                counts = time.astype(bool).resample(time=resample_freq).count()
                if any(counts > 1):
                    raise ValueError(
                        f"Alignment of dataset #{i:02d} failed: Time axis cannot be resampled to freq {resample_freq}."
                    )
                time = counts.time

            ds["time"] = time
            calendars.append(get_calendar(time))

        ds_all.append(ds)

    if not calendars:
        # no time
        return ds_all

    if calendar is None:
        calendar = common_calendar(calendars, join="outer")
    cal_kwargs.setdefault("align_on", "date")
    return [ds.convert_calendar(calendar, **cal_kwargs) for ds in ds_all]

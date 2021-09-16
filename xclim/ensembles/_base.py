"""Ensembles creation and statistics."""
import logging
from pathlib import Path
from typing import List, Optional, Sequence, Union

import numpy as np
import xarray as xr

from xclim.core.calendar import convert_calendar, get_calendar
from xclim.core.formatting import update_history
from xclim.core.utils import calc_perc


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
    >>> from xclim.ensembles import create_ensemble
    >>> ens = create_ensemble(temperature_datasets)

    Using multifile datasets, through glob patterns.
    Simulation 1 is a list of .nc files (e.g. separated by time):

    >>> datasets = glob.glob('/dir/*.nc')  # doctest: +SKIP

    Simulation 2 is also a list of .nc files:

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

    Create the ensemble dataset:

    >>> ens = create_ensemble(temperature_datasets)

    Calculate ensemble statistics:

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
    ds_out.attrs["history"] = update_history(
        f"Computation of statistics on {ens.realization.size} ensemble members.", ds_out
    )
    return ds_out


def ensemble_percentiles(
    ens: Union[xr.Dataset, xr.DataArray],
    values: Sequence[float] = [10, 50, 90],
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

    Create ensemble dataset:

    >>> ens = create_ensemble(temperature_datasets)

    Calculate default ensemble percentiles:

    >>> ens_percs = ensemble_percentiles(ens)

    Calculate non-default percentiles (25th and 75th)

    >>> ens_percs = ensemble_percentiles(ens, values=(25, 50, 75))

    If the original array has many small chunks, it might be more efficient to do:

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
        out.attrs["history"] = update_history(
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
            chk_dim, chks = max(
                enumerate(ens.chunks),
                key=lambda kv: 0
                if kv[0] == ens.get_axis_num("realization")
                else max(kv[1]),
            )
            ens = ens.chunk(
                {"realization": -1, ens.dims[chk_dim]: len(chks) * ens.realization.size}
            )
        else:
            ens = ens.chunk({"realization": -1})

    out = xr.apply_ufunc(
        calc_perc,
        ens,
        input_core_dims=[["realization"]],
        output_core_dims=[["percentiles"]],
        keep_attrs=True,
        kwargs=dict(
            percentiles=values,
        ),
        dask="parallelized",
        output_dtypes=[ens.dtype],
        dask_gufunc_kwargs=dict(output_sizes={"percentiles": len(values)}),
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

    out.attrs["history"] = update_history(
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
    xr_kwargs.setdefault("chunks", "auto")
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

import numpy as np
import pandas as pd
import xarray as xr


def create_ensemble(ncfiles, mf_flag=False):
    """Create an xarray datset of ensemble of climate simulation from a list of netcdf files. Input data is
    concatenated along a newly created data dimension ('realization')

    Returns a xarray dataset object containing input data from the list of netcdf files concatenated along
    a new dimension (name:'realization'). In the case where input files have unequal time dimensions output
    ensemble dataset is created for overlapping time-steps common to all input files

    Parameters
    ----------
    ncfiles : sequence
      List of netcdf file paths. If mf_flag is true ncfiles should be a list of lists where
    each sublist contains input .nc files of a multifile dataset

    mf_flag : Boolean . If true climate simulations are treated as multifile datasets before concatenation

    Returns
    -------
    xarray dataset containing concatenated data from all input files

    Notes
    -----
    Input netcdf files require equal spatial dimension size (e.g. lon, lat dimensions)
    If input data contains multiple cftime calendar types they must be at monthly or coarser frequency

    Examples
    --------
    >>> from xclim import utils
    >>> import glob
    >>> ncfiles = glob.glob('/*.nc')
    >>> ens = utils.create_ensemble(ncfiles)
    >>> print(ens)
    Using multifile datasets:
    simulation 1 is a list of .nc files (e.g. separated by time)
    >>> ncfiles = glob.glob('dir/*.nc')
    simulation 2 is also a list of .nc files
    >>> ens = utils.create_ensemble(ncfiles)
    """
    dim = 'realization'
    ds1 = []
    start_end_flag = True
    print('finding common time-steps')
    for n in ncfiles:
        if mf_flag:
            ds = xr.open_mfdataset(n, concat_dim='time', decode_times=False, chunks={'time': 10})
            ds['time'] = xr.open_mfdataset(n).time
        else:
            ds = xr.open_dataset(n, decode_times=False)
            ds['time'] = xr.decode_cf(ds).time
        # get times - use common
        time1 = pd.to_datetime({'year': ds.time.dt.year, 'month': ds.time.dt.month, 'day': ds.time.dt.day})
        if start_end_flag:
            start1 = time1.values[0]
            end1 = time1.values[-1]
            start_end_flag = False
        if time1.values.min() > start1:
            start1 = time1.values.min()
        if time1.values.max() < end1:
            end1 = time1.values.max()

    for n in ncfiles:
        print('accessing file ', ncfiles.index(n) + 1, ' of ', len(ncfiles))
        if mf_flag:
            ds = xr.open_mfdataset(n, concat_dim='time', decode_times=False, chunks={'time': 10})
            ds['time'] = xr.open_mfdataset(n).time
        else:
            ds = xr.open_dataset(n, decode_times=False, chunks={'time': 10})
            ds['time'] = xr.decode_cf(ds).time

        ds['time'].values = pd.to_datetime({'year': ds.time.dt.year, 'month': ds.time.dt.month, 'day': ds.time.dt.day})

        ds = ds.where((ds.time >= start1) & (ds.time <= end1), drop=True)

        ds1.append(ds.drop('time'))
    print('concatenating files : adding dimension ', dim, )
    ens = xr.concat(ds1, dim=dim)
    # assign time coords
    ens = ens.assign_coords(time=ds.time.values)
    return ens


def ensemble_mean_std_max_min(ens):
    """Calculate ensemble statistics between a results from an ensemble of climate simulations

    Returns a dataset containing ensemble mean, standard-deviation,
    minimum and maximum for input climate simulations.

    Parameters
    ----------
    ens : Ensemble dataset (see xclim.utils.create_ensemble)

    Returns
    -------
    xarray dataset with containing data variables of ensemble statistics

    Examples
    --------
    >>> from xclim import utils
    >>> import glob
    >>> ncfiles = glob.glob('/*tas*.nc')
    Create ensemble dataset
    >>> ens = utils.create_ensemble(ncfiles)
    Calculate ensemble statistics
    >>> ens_means_std = utils.ensemble_mean_std_max_min(ens)
    >>> print(ens_mean_std['tas_mean'])
    """
    dsOut = ens.drop(ens.data_vars)
    for v in ens.data_vars:

        dsOut[v + '_mean'] = ens[v].mean(dim='realization')
        dsOut[v + '_stdev'] = ens[v].std(dim='realization')
        dsOut[v + '_max'] = ens[v].max(dim='realization')
        dsOut[v + '_min'] = ens[v].min(dim='realization')
        for vv in dsOut.data_vars:
            dsOut[vv].attrs = ens[v].attrs

            if 'description' in dsOut[vv].attrs.keys():
                vv.split()
                dsOut[vv].attrs['description'] = dsOut[vv].attrs['description'] + ' : ' + vv.split('_')[
                    -1] + ' of ensemble'

    return dsOut


def ensemble_percentiles(ens, values=(10, 50, 90), time_block=None):
    """Calculate ensemble statistics between a results from an ensemble of climate simulations

    Returns a dataset containing ensemble statistics for input climate simulations.
    Alternatively calculate ensemble percentiles (default) or ensemble mean and standard deviation

    Parameters
    ----------
    ens : Ensemble dataset (see xclim.utils.create_ensemble)
    values : tuple of integers - percentile values to calculate  : default : (10, 50, 90)
    time_block : integer
      for large ensembles iteratively calculate percentiles in time-step blocks (n==time_block).
       If not defined the function tries to estimate an appropriate value

    Returns
    -------
    xarray dataset with containing data variables of requested ensemble statistics

    Examples
    --------
    >>> from xclim import utils
    >>> import glob
    >>> ncfiles = glob.glob('/*tas*.nc')
    Create ensemble dataset
    >>> ens = utils.create_ensemble(ncfiles)
    Calculate default ensemble percentiles
    >>> ens_percs = utils.ensemble_statistics(ens)
    >>> print(ens_percs['tas_p10'])
    Calculate non-default percentiles (25th and 75th)
    >>> ens_percs = utils.ensemble_statistics(ens, values=(25,75))
    >>> print(ens_percs['tas_p25'])
    Calculate by time blocks (n=10) if ensemble size is too large to load in memory
    >>> ens_percs = utils.ensemble_statistics(ens, time_block=10)
    >>> print(ens_percs['tas_p25'])

    """

    ds_out = ens.drop(ens.data_vars)
    dims = list(ens.dims)
    for v in ens.data_vars:
        # Percentile calculation requires load to memory : automate size for large ensemble objects
        if not time_block:
            time_block = round(2E8 / (ens[v].size / ens[v].shape[dims.index('time')]), -1)  # 2E8

        if time_block > len(ens[v].time):
            out = _calc_percentiles_simple(ens, v, values)

        else:
            # loop through blocks
            Warning('large ensemble size detected : statistics will be calculated in blocks of ', int(time_block),
                    ' time-steps')
            out = _calc_percentiles_blocks(ens, v, values, time_block)
        for vv in out.data_vars:
            ds_out[vv] = out[vv]
    return ds_out


def _calc_percentiles_simple(ens, v, values):
    ds_out = ens.drop(ens.data_vars)
    dims = list(ens[v].dims)
    outdims = [x for x in dims if 'realization' not in x]

    print('loading ensemble data to memory')
    arr = ens[v].load()  # percentile calc requires loading the array
    coords = {}
    for c in outdims:
        coords[c] = arr[c]
    for p in values:
        outvar = v + '_p' + str(p)

        out1 = _calc_perc(arr, p)

        ds_out[outvar] = xr.DataArray(out1, dims=outdims, coords=coords)
        ds_out[outvar].attrs = ens[v].attrs
        if 'description' in ds_out[outvar].attrs.keys():
            ds_out[outvar].attrs['description'] = '{} : {}th percentile of ensemble'.format(
                ds_out[outvar].attrs['description'], str(p))
        else:
            ds_out[outvar].attrs['description'] = '{}th percentile of ensemble'.format(str(p))

    return ds_out


def _calc_percentiles_blocks(ens, v, values, time_block):
    ds_out = ens.drop(ens.data_vars)
    dims = list(ens[v].dims)
    outdims = [x for x in dims if 'realization' not in x]

    blocks = list(range(0, len(ens.time) + 1, int(time_block)))
    if blocks[-1] != len(ens[v].time):
        blocks.append(len(ens[v].time))
    arr_p_all = {}
    for t in range(0, len(blocks) - 1):
        print('Calculating block ', t + 1, ' of ', len(blocks) - 1)
        time_sel = slice(blocks[t], blocks[t + 1])
        arr = ens[v].isel(time=time_sel).load()  # percentile calc requires loading the array
        coords = {}
        for c in outdims:
            coords[c] = arr[c]
        for p in values:

            out1 = _calc_perc(arr, p)

            if t == 0:
                arr_p_all[str(p)] = xr.DataArray(out1, dims=outdims, coords=coords)
            else:
                arr_p_all[str(p)] = xr.concat([arr_p_all[str(p)],
                                               xr.DataArray(out1, dims=outdims, coords=coords)], dim='time')
    for p in values:
        outvar = v + '_p' + str(p)
        ds_out[outvar] = arr_p_all[str(p)]
        ds_out[outvar].attrs = ens[v].attrs
        if 'description' in ds_out[outvar].attrs.keys():
            ds_out[outvar].attrs['description'] = '{} : {}th percentile of ensemble'.format(
                ds_out[outvar].attrs['description'], str(p))
        else:
            ds_out[outvar].attrs['description'] = '{}th percentile of ensemble'.format(str(p))

    return ds_out


def _calc_perc(arr, p):
    dims = arr.dims
    # make sure time is the second dimension
    if dims.index('time') != 1:
        dims1 = [dims[dims.index('realization')], dims[dims.index('time')]]
        for d in dims:
            if d not in dims1:
                dims1.append(d)
        arr = arr.transpose(*dims1)
        dims = dims1

    nan_count = np.isnan(arr).sum(axis=dims.index('realization'))
    out = np.percentile(arr, p, axis=dims.index('realization'))
    if np.any((nan_count > 0) & (nan_count < arr.shape[dims.index('realization')])):
        arr1 = arr.values.reshape(arr.shape[dims.index('realization')],
                                  int(arr.size / arr.shape[dims.index('realization')]))
        # only use nanpercentile where we need it (slow performace compared to standard) :
        nan_index = np.where((nan_count > 0) & (nan_count < arr.shape[dims.index('realization')]))
        t = np.ravel_multi_index(nan_index, nan_count.shape)
        out[np.unravel_index(t, nan_count.shape)] = np.nanpercentile(arr1[:, t], p, axis=dims.index('realization'))

    return out

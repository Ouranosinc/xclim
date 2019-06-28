import xarray as xr
import numpy as np


def delta(src, dst, nbins, group, kind='+'):
    """Compute quantile mapping factors.

    Parameters
    ----------
    src : xr.DataArray
      Source time series.
    dst : xr.DataArray
      Destination time series.
    nbins : int
      Number of quantile bins.
    group : {'time.month', 'time.week', 'time.dayofyear'}
      Grouping criterion.
    kind : {'+', '*'}
      The transfer operation, + for additive and * for multiplicative.

    Returns
    -------
    xr.DataArray
      Delta factor computed over time grouping and quantile bins.
    """
    bins = np.linspace(0., 1., nbins)

    sg = src.groupby(group).quantile(bins)
    dg = dst.groupby(group).quantile(bins)

    if kind == '+':
        out = dg - sg
    elif kind == '*':
        out = dg / sg
    else:
        raise ValueError("kind must be + or *.")

    out.attrs['kind'] = kind
    out.attrs['group'] = group
    out.attrs['bins'] = bins
    return out


def apply(da, qmf, interp=False):
    """Apply quantile mapping delta to an array.

    Parameters
    ----------
    da : xr.DataArray
      Input array to be modified.
    qmf : xr.DataArray
      Quantile mapping factors computed by the `delta` function.
    interp : bool
      Whether to interpolate between the groupings.

    Returns
    -------
    xr.DataArray
      Input array with delta applied.
    """

    if 'time' not in qmf.group:
        raise NotImplementedError

    if 'season' in qmf.group and interp:
        raise NotImplementedError

    # Find the group indexes
    ind, att = qmf.group.split('.')

    time = da.coords['time']
    gc = qmf.coords[att]
    ng = len(gc)

    # Add cyclical values to the scaling factors for interpolation
    if interp:
        qmf = add_cyclic(qmf, att)

    # Compute the percentile time series of the input array
    q = da.groupby(qmf.group).apply(xr.DataArray.rank, pct=True, dim=ind)
    iq = xr.DataArray(q, dims='time', coords={'time': time}, name='quantile index')

    # Create DataArrays for indexing
    # TODO: Adjust for different calendars if necessary.
    if interp:
        it = xr.DataArray((q.indexes['time'].dayofyear-1) / 365. * ng + 0.5,
                          dims='time', coords={'time': time}, name='time group index')
    else:
        it = xr.DataArray(getattr(q.indexes[ind], att),
                          dims='time', coords={'time': time}, name='time group index')


    # Extract the correct quantile for each time step.
    if interp:  # Interpolate both the time group and the quantile.
        factor = qmf.interp({att: it, 'quantile': iq})
    else:  # Find quantile for nearest time group and quantile.
        factor = qmf.sel({att: it, 'quantile': iq}, method='nearest')

    # Apply delta to input time series.
    out = da.copy()
    if qmf.kind == '+':
        out += factor
    elif qmf.kind == '*':
        out *= factor

    out.attrs['bias_corrected'] = True

    # Remove time grouping and quantile coordinates
    return out.drop(['quantile', att])


def add_cyclic(qmf, att):
    """Reindex the scaling factors to include the last grouping
    at the beginning and the first at the end.

    This is done to allow interpolation near the end-points.
    """
    gc = qmf.coords[att]
    i = np.concatenate(([-1, ], range(len(gc)), [0, ]))
    qmf = qmf.reindex({att: gc[i]})
    qmf.coords[att] = range(len(qmf))
    return qmf

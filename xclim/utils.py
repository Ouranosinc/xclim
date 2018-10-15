# -*- coding: utf-8 -*-

"""
xclim xarray.DataArray utilities module
"""

import numpy as np


def daily_downsampler(da, freq='YS'):
    r"""Daily climate data downsampler

    Parameters
    ----------
    da : xarray.DataArray
    freq : string

    Returns
    -------
    xarray.DataArray


    Note
    ----

        Usage Example

            grouper = daily_downsampler(da_std, freq='YS')
            x2 = grouper.mean()

            # add time coords to x2 and change dimension tags to time
            time1 = daily_downsampler(da_std.time, freq=freq).first()
            x2.coords['time'] = ('tags', time1.values)
            x2 = x2.swap_dims({'tags': 'time'})
            x2 = x2.sortby('time')
    """

    # generate tags from da.time and freq
    if isinstance(da.time.values[0], np.datetime64):
        years = ['{:04d}'.format(y) for y in da.time.dt.year.values]
        months = ['{:02d}'.format(m) for m in da.time.dt.month.values]
    else:
        # cannot use year, month, season attributes, not available for all calendars ...
        years = ['{:04d}'.format(v.year) for v in da.time.values]
        months = ['{:02d}'.format(v.month) for v in da.time.values]
    seasons = ['DJF DJF MAM MAM MAM JJA JJA JJA SON SON SON DJF'.split()[int(m) - 1] for m in months]

    n_t = da.time.size
    if freq == 'YS':
        # year start frequency
        l_tags = years
    elif freq == 'MS':
        # month start frequency
        l_tags = [years[i] + months[i] for i in range(n_t)]
    elif freq == 'QS-DEC':
        # DJF, MAM, JJA, SON seasons
        # construct tags from list of season+year, increasing year for December
        ys = []
        for i in range(n_t):
            m = months[i]
            s = seasons[i]
            y = years[i]
            if m == '12':
                y = str(int(y) + 1)
            ys.append(y + s)
        l_tags = ys
    else:
        raise RuntimeError('freqency {:s} not implemented'.format(freq))

    # add tags to buffer DataArray
    buffer = da.copy()
    buffer.coords['tags'] = ('time', l_tags)

    # return groupby according to tags
    return buffer.groupby('tags')


class Indicator(object):
    r"""xclim indicator class"""
    identifier = ''
    units = ''
    required_units = ''
    long_name = ''
    standard_name = ''
    description = ''
    keywords = ''

    def compute(self, *args, **kwds):
        """Index computation method. To be subclassed"""
        raise NotImplementedError

    def convert_units(self, *args):
        """Return DataArray with correct units, defined by `self.required_units`."""
        raise NotImplementedError

    def cfprobe(self, *args):
        """Check input data compliance to expectations.
        Warn of potential issues."""
        raise NotImplementedError

    def validate(self, *args):
        """Validate input data requirements.
        Raise error if conditions are not met."""
        raise NotImplementedError

    def decorate(self, da):
        """Modify output's attributes in place."""
        da.attrs.update(self.attrs)

    def missing(self, *args, **kwds):
        """Return boolean DataArray . To be subclassed"""
        raise NotImplementedError

    def __init__(self):
        # Extract DataArray arguments from compute signature.
        self.attrs = {'long_name': self.long_name,
                      'units': self.units,
                      'standard_name': self.standard_name}

    def __call__(self, *args, **kwds):
        self.validate(*args)
        self.cfprobe(*args)

        cargs = self.convert_units(*args)

        out = self.compute(*cargs, **kwds)
        self.decorate(out)

        return out.where(self.missing(*cargs, freq=kwds['freq']))  # Won't work if keyword is passed as positional arg.

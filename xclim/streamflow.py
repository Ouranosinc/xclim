# -*- coding: utf-8 -*-
import numpy as np
from xclim import checks
from xclim import indices as _ind
from xclim.utils import Indicator, generic_frequency_analysis, generic_max
from xclim.utils import generic_seasonal_stat_return_period
from functools import partial, update_wrapper
from boltons.funcutils import FunctionBuilder


def wrapped_partial(func, *args, **kwargs):
    partial_func = partial(func, *args, **kwargs)
    update_wrapper(partial_func, func)
    return partial_func


docstring_template = \
    """{title}
    
    {description}
    
    Parameters
    ----------
    q : xarray.DataArray
      Input streamflow [m3/s]
    {params}
    
    Returns
    -------
    xarray.DataArray
      {output}
      
    Notes
    -----
    {notes}
    """


class Streamflow(Indicator):
    required_units = 'm^3 s-1'
    units = 'm^3 s-1'
    context = 'hydro'
    standard_name = 'streamflow'

    @staticmethod
    def compute(*args, **kwds):
        pass

    def cfprobe(self, q):
        checks.check_valid(q, 'standard_name', 'streamflow')

    def missing(self, *args, **kwds):
        """Return whether an output is considered missing or not."""
        from functools import reduce

        freq = kwds.get('freq', None) or getattr(self, 'freq')
        # Si on besoin juste d'une saison, cette fonction va quand même checker les données pour toutes les saisons.
        miss = (checks.missing_any(da, freq) for da in args)
        return reduce(np.logical_or, miss)


base_flow_index = Streamflow(identifier='base_flow_index',
                             units='',
                             standard_name='',
                             compute=_ind.base_flow_index)

q_max = Streamflow(identifier='q_max',
                   compute=generic_max,
                   )


def generate(identifier, title, description, notes='', defaults={}, fixed={}):
    """Return a Streamflow instance based on the `generic_frequency_analysis` function.

    Parameters
    ----------
    identifier : str
      Function name
    title : str
      Brief description of function's intent.
    description : str
      One paragraph explanation of function's behavior.
    notes : str
      Additional information, such as the mathematical formula.
    defaults : dict
      Default values for keyword arguments of the created function.
    fixed : dict
      Fixed values passed to the `generic_frequency_analysis` function.
    """

    # TODO: create parameter lines in docstring.
    params = ""
    output = ""
    args = ['q', ] + list(defaults.keys())
    gfa_args = ["da=q", ] + \
               ["{0}={0}".format(k) for k in defaults.keys()] + \
               ["{0}={1}".format(k, v) for (k, v) in fixed.items()]

    body = """from xclim.utils import generic_frequency_analysis as gfa\nreturn gfa({})""".format(', '.join(gfa_args))

    f = FunctionBuilder(name=identifier.lower(),
                        doc=docstring_template.format(**{'identifier': identifier,
                                                         'title': title,
                                                         'description': description,
                                                         'params': params,
                                                         'output': output,
                                                         'notes': notes}),
                        body=body,
                        args=args,
                        defaults=tuple(defaults.values())
                        )

    print(body)
    return Streamflow(identifier=identifier,
                      title=title,
                      description=description,
                      compute=f.get_func())


def generate2(identifier, title, description, notes='', defaults={}, fixed={},
              standard_name='', long_name=''):
    """Return a Streamflow instance based on the `generic_frequency_analysis` function.

    Parameters
    ----------
    identifier : str
      Function name
    title : str
      Brief description of function's intent.
    description : str
      One paragraph explanation of function's behavior.
    notes : str
      Additional information, such as the mathematical formula.
    defaults : dict
      Default values for keyword arguments of the created function.
    fixed : dict
      Fixed values passed to the `generic_frequency_analysis` function.
    """

    # TODO: create parameter lines in docstring.
    params = ""
    output = ""
    args = ['q', ] + list(defaults.keys())
    gfa_args = ["da=q", ] + \
               ["{0}={0}".format(k) for k in defaults.keys()] + \
               ["{0}={1}".format(k, v) for (k, v) in fixed.items()]

    body = """from xclim.utils import generic_seasonal_stat_return_period as gfa2\nreturn gfa2({})""".format(
        ', '.join(gfa_args))

    f = FunctionBuilder(name=identifier.lower(),
                        doc=docstring_template.format(**{'identifier': identifier,
                                                         'title': title,
                                                         'description': description,
                                                         'params': params,
                                                         'output': output,
                                                         'notes': notes}),
                        body=body,
                        args=args,
                        defaults=tuple(defaults.values())
                        )

    print(body)
    s = Streamflow(identifier=identifier,
                   title=title,
                   standard_name=standard_name,
                   long_name=long_name,
                   compute=f.get_func())

    if 'freq' in fixed:
        s.freq = eval(fixed['freq'])

    return s


# What I'm doing here is taking a generic function and assigning new defaults to it (for freq, t, dist, and mode).
# You still can override these defaults when calling the function, which may not be great. I think some defaults
# should be frozen (t, mode) while others could be left free (dist, freq).
q1max2sp_old = generate(identifier='Q1max2Sp',
                        title='2-year return period maximum spring flood peak',
                        description='Annual maximum [max] daily flow [Q1] of 2-year recurrence [2] in spring [Sp]',
                        defaults=dict(freq='QS-MAR', dist='gumbel_r'),
                        fixed=dict(t=2, mode="'high'"))


def QWindowModeTSeasons_generate2_wrapper(window, mode, t, seasons):
    """functiong wrapping generate2 to automatize things for indices of the general form
    QWindowModeTSeasons

    Parameters
    ----------
    window : int
      number of days over which streamflow is averaged
    mode : {'max', 'min'}
      Whether we are looking for a probability of exceedance (max) or a probability of non-exceedance (min).
    t : int
      Return period. The period depends on the resolution of the input data. If the input array's resolution is
      yearly, then the return period is in years.
    seasons : string
      list of the seasons considered among 'sp', 'su', 'a', 'w', 'sua' corresponding to:
      'spring', 'summer', 'autumn', 'winter', 'summer-autumn' respectively

    """
    # check coherence of arguments
    assert (mode in 'min max'.split())
    assert (seasons in 'sp su a w sua'.split())

    long_name_mode = dict(min='minimum', max='maximum')[mode]
    month_seasons = dict(zip('sp su a w sua'.split(),
                             '"MAM" "JJA" "SON" "DJF"'.split() + ['JJA SON'.split()]))[seasons]
    name_seasons = dict(zip('sp su a w sua'.split(),
                            'spring summer automn winter summer-autumn'.split()))[seasons]

    identifier = 'q{:}{:}{:}{:}'.format(window, mode, t, seasons)
    title = '{:}-year return period {:} {:} {:}-day streamflow'.format(t, long_name_mode, name_seasons,
                                                                       window)
    description = 'Annual {:} [{:}] {:}-day flow [Q{:}] of {:}-year recurrence [{:}] in {:} [{:}]'.format(
        long_name_mode, mode, window, window, t, t, name_seasons, seasons)
    standard_name = '{:}-year_return_period_{:}_{:}_{:}-day_flow'.format(t, long_name_mode, name_seasons,
                                                                         window)
    long_name = '{:}-year return period {:} {:} {:}-day flow'.format(t, long_name_mode, name_seasons,
                                                                     window)
    mode_hl = '"{:s}"'.format(dict(max='high', min='low')[mode])

    print(month_seasons)

    gen2 = generate2(identifier=identifier,
                     title=title,
                     description=description,
                     standard_name=standard_name,
                     long_name=long_name,
                     defaults=dict(dist='gumbel_r'),
                     fixed=dict(t=t, mode=mode_hl, seasons=month_seasons, window=1, freq="'AS-Dec'"))

    return gen2


# q1max2sp = generate2(identifier='q1max2sp',
#                      title='2-year return period maximum spring flood peak',
#                      description='Annual maximum [max] daily flow [Q1] of 2-year recurrence [2] in spring [Sp]',
#                      standard_name='2-year_return_period_maximum_spring_daily_flow',
#                      long_name='2-year return period maximum spring daily flow',
#                      defaults=dict(dist='gumbel_r'),
#                      fixed=dict(t=2, mode="'high'", seasons="'MAM'", window=1, freq="'AS-Dec'"))

q1max2sp = QWindowModeTSeasons_generate2_wrapper(1, 'max', 2, 'sp')

q1max20sp = QWindowModeTSeasons_generate2_wrapper(1, 'max', 20, 'sp')

q14max2sp = QWindowModeTSeasons_generate2_wrapper(14, 'max', 2, 'sp')

q14max20sp = QWindowModeTSeasons_generate2_wrapper(14, 'max', 20, 'sp')

q1max2sua = QWindowModeTSeasons_generate2_wrapper(1, 'max', 2, 'sua')

q1max20sua = QWindowModeTSeasons_generate2_wrapper(1, 'max', 20, 'sua')

q7min2su = QWindowModeTSeasons_generate2_wrapper(7, 'min', 2, 'su')

q7min10su = QWindowModeTSeasons_generate2_wrapper(7, 'min', 10, 'su')

q30min5su = QWindowModeTSeasons_generate2_wrapper(30, 'min', 5, 'su')

q7min10w = QWindowModeTSeasons_generate2_wrapper(7, 'min', 10, 'w')

q30min5w = QWindowModeTSeasons_generate2_wrapper(30, 'min', 5, 'w')


def get_mean_doy_of_max(da, seasons=['JJA'], freq='AS'):
    """function computing the multi-year mean of the 'day of year' of the maximum value during given seasons

    Parameters
    ----------
    da : xarray.DataArray
      Input data
    seasons : list of string
      list of the seasons considered among 'DJF', 'MAM', 'JJA', 'SON'
    freq : str
      Resampling frequency used to split the wanted season(s) into different "years"
      defined in http://pandas.pydata.org/pandas-docs/stable/timeseries.html#resampling.

    Examples
    --------
    >>> q = xr.open_dataset('streampflow.nc')
    >>> mdoy_of_max = get_mean_doy_of_max(q, seasons=["SON"])
    """

    das = da.sel(time=da.time.dt.season.isin(seasons)).dropna(dim='time')

    def _get_doy_of_max(da):
        i_time_max = da.argmax(dim='time')
        doy_of_max = da.isel(time=i_time_max).time.dt.dayofyear
        return doy_of_max

    doy_of_max = das.resample(time=freq).apply(_get_doy_of_max)
    doy_of_max_tavg = doy_of_max.mean(dim='time')
    return doy_of_max_tavg


doy_q1maxsp = Streamflow(identifier='doy_q1maxsp',
                         units='day of year',
                         standard_name='average_day_of_year_of_annual_maximum_spring_value',
                         long_name='day of year of annual maximum spring value',
                         description="",
                         cell_methods='time: maximum within season time: mean over years',
                         compute=get_mean_doy_of_max,
                         )

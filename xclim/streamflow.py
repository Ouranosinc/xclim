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
        # TODO
        # Si on besoin juste d'une saison, cette fonction va quand même checker les données pour toutes les saisons.
        miss = (checks.missing_any(da, freq) for da in args)
        return reduce(np.logical_or, miss)


class Streamflow_time_operator(Streamflow):

    def __call__(self, *args, **kwds):
        # necessary imports
        import datetime as dt
        from collections import defaultdict
        from inspect2 import signature

        # Bind call arguments. We need to use the class signature, not the instance, otherwise it removes the first
        # argument.
        if self._partial:
            ba = self._sig.bind_partial(*args, **kwds)
            for key, val in self.compute.keywords.items():
                if key not in ba.arguments:
                    ba.arguments[key] = val
        else:
            ba = self._sig.bind(*args, **kwds)
            ba.apply_defaults()

        # Get history and cell method attributes from source data
        attrs = defaultdict(str)
        for i in range(self._nvar):
            p = self._parameters[i]
            for attr in ['history', 'cell_methods']:
                attrs[attr] += "{}: ".format(p) if self._nvar > 1 else ""
                attrs[attr] += getattr(ba.arguments[p], attr, '')
                if attrs[attr]:
                    attrs[attr] += "\n" if attr == 'history' else " "

        # Update attributes
        out_attrs = self.json(ba.arguments)
        formatted_id = out_attrs.pop('identifier')
        attrs['history'] += '[{:%Y-%m-%d %H:%M:%S}] {}{}'.format(dt.datetime.now(), formatted_id, ba.signature)
        attrs['cell_methods'] += out_attrs.pop('cell_methods')
        attrs.update(out_attrs)

        # Assume the first arguments are always the DataArray.
        das = tuple((ba.arguments.pop(self._parameters[i]) for i in range(self._nvar)))

        # Pre-computation validation checks
        for da in das:
            self.validate(da)
        self.cfprobe(*das)

        # Convert units if necessary
        das = tuple((self.convert_units(da, ru, self.context) for (da, ru) in zip(das, self.required_units)))

        # Compute the indicator values, ignoring NaNs.
        out = self.compute(*das, **ba.arguments)
        out.attrs.update(attrs)

        # Bind call arguments to the `missing` function, whose signature might be different from `compute`.
        mba = signature(self.missing).bind(*das, **ba.arguments)

        # Mask results that do not meet criteria defined by the `missing` method.
        mask = self.missing(*mba.args, **mba.kwargs)
        ma_out = out.where(~mask)

        # apply time operator before returning output
        out = ma_out.rename(formatted_id)

        # TODO
        # give time_operator a default value so we could add this code into utils.Indicator directly and
        # would not need a special Indicator class

        if self.time_operator == 'mean':
            out_op = out.mean(dim='time')
        elif self.time_operator == 'monthly_annual_cycle':
            out_op = out.groupby(out.time.dt.month).mean(dim='time')
        else:
            raise RuntimeError('time_operator:{} not expected'.format(self.time_operator))

        return out_op


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


def generate2(identifier, title, description, gfa_import='from xclim.utils import generic_seasonal_stat_return_period',
              notes='', defaults={}, fixed={},
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
    gfa_import: str
      string to use for the import of gfa
      e.g from xclim.utils import generic_seasonal_stat_return_period
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

    body = """{} as gfa2\nreturn gfa2({})""".format(gfa_import,
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
                     gfa_import='from xclim.utils import generic_seasonal_stat_return_period',
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


def get_doy_of_max(da, seasons=['JJA'], freq='AS'):
    """function returning the 'day of year' of the maximum value of input data
    occuing during given seasons

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
    return doy_of_max


doy_q1maxsp = Streamflow_time_operator(identifier='doy_q1maxsp',
                                       units='day of year',
                                       standard_name='average_day_of_year_of_annual_maximum_spring_value',
                                       long_name='day of year of annual maximum spring value',
                                       description="",
                                       cell_methods='time: maximum within season time: mean over years',
                                       compute=get_doy_of_max,
                                       time_operator='mean'
                                       )


def q_monthly_mean(q):
    """Function computing the monthly averages of streamflow every year

    :param q:
    :return:
    """
    # split data into different months and compute mean
    qrm = q.resample(time='MS').mean(dim='time')

    return qrm


def q_season_mean(q, seasons, freq='AS-DEC'):
    """Function that computes yearly mean of streamflow over wanted seasons

    :param q:
    :param seasons:
    :param freq:
    :return:
    """

    # select streamflow over wanted seasons
    qsr = q.sel(time=q.time.dt.season.isin(seasons)).dropna(dim='time')

    # split the years and make the mean
    qsm = qsr.resample(time=freq).mean(dim='time')

    return qsm


def _qavg(q):
    return q_season_mean(q, seasons='DJF MAM JJA SON'.split(), freq='AS-JAN')


def _qavgwsp(q):
    return q_season_mean(q, seasons='DJF MAM'.split(), freq='AS-DEC')


def _qavgsua(q):
    return q_season_mean(q, seasons='JJA SON'.split(), freq='AS-DEC')


def _qavg_1_12(q):
    qm = q_monthly_mean(q)
    return qm


qavg = Streamflow_time_operator(identifier='qavg',
                                units='m^3 s-1',
                                standard_name='',
                                long_name='',
                                description="",
                                cell_methods='',
                                compute=_qavg,
                                freq='AS-JAN',
                                time_operator='mean'
                                )

qavgwsp = Streamflow_time_operator(identifier='qavgwsp',
                                   units='m^3 s-1',
                                   standard_name='',
                                   long_name='',
                                   description="",
                                   cell_methods='',
                                   compute=_qavgwsp,
                                   freq='AS-DEC',
                                   time_operator='mean'
                                   )

qavgsua = Streamflow_time_operator(identifier='qavgsua',
                                   units='m^3 s-1',
                                   standard_name='',
                                   long_name='',
                                   description="",
                                   cell_methods='',
                                   compute=_qavgsua,
                                   freq='AS-DEC',
                                   time_operator='mean'
                                   )

qavg_1_12 = Streamflow_time_operator(identifier='avg_1_12',
                                     units='m^3 s-1',
                                     standard_name='',
                                     long_name='',
                                     description="",
                                     cell_methods='',
                                     compute=_qavg_1_12,
                                     freq='MS',
                                     time_operator='monthly_annual_cycle'
                                     )

# -*- coding: utf-8 -*-
import numpy as np
from xclim import checks
from xclim import indices as _ind
from xclim.utils import Indicator, wrapped_partial
from boltons.funcutils import FunctionBuilder
from xclim import generic
import calendar


class Streamflow(Indicator):
    required_units = 'm^3 s-1'
    units = 'm^3 s-1'
    context = 'hydro'
    standard_name = 'discharge'

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


base_flow_index = Streamflow(identifier='base_flow_index',
                             units='',
                             long_name="Base flow index",
                             compute=_ind.base_flow_index)



q_mean = Streamflow(identifer='q_mean',
                    long_name="Mean daily streamflow",
                    compute=generic.select_resample_op,
                    op = 'mean',
                    var_name = 'da',
                    doc_template = generic.mean_doc)


class QIndGen:
    """Helper class to create Streamflow instances automatically.

    Example
    -------
    >>> Q = QWindowModeTPeriod()
    >>> ind = Q.fa(1, 'max', 2, season='MAM')
    """

    def __init__(self):
        """
        Parameters
        ----------
        window : int
          number of days over which streamflow is averaged
        mode : {'max', 'min'}
          Whether we are looking for a probability of exceedance (max) or a probability of non-exceedance (min).
        t : int
          Return period. The period depends on the resolution of the input data. If the input array's resolution is
          yearly, then the return period is in years.
        indexer : {period: value}
          Period can either be 'season' or 'month'. Values would be season identifiers ('DJF'), month indices (1-12),
          or a list of season or months.
        """

        self._mode_names = dict(min='minimum', max='maximum')
        self._season_names = dict(sp='spring', su='summer', f='fall', w='winter', suf='summer-fall')
        self._season_months = dict(sp='MAM', su='JJA', f='SON', w='DJF', suf=['JJA', 'SON'])
        self._rseasons = {str(v): k for k, v in self._season_months.items()}

    def sop(self, op, **indexer):
        """Simple operation Streamflow instance generator."""
        self._op = op
        freq = generic.default_freq(**indexer)

        identifier = "q{m.period_id}{m.op}"
        title = "{{freq}} {m.op} over {m.period}"
        long_name = title
        description = ""

        body_template = """from xclim.generic import select_resample_op as func"""
        if not isinstance(op, str):
            body_template += "\nfrom {} import {}".format(op.__module__, op.__name__)

        return self._generate(identifier.format(m=self),
                              title.format(m=self),
                              description.format(m=self),
                              long_name=long_name.format(m=self),
                              body_template=body_template,
                              defaults=dict(freq=freq),
                              fixed=dict(op=self.op))

    def fa(self, window, mode, t, freq=None, **indexer):
        """Frequency analysis Streamflow instance generator."""
        self.window = window
        self.mode = mode
        self.t = t
        indexer = indexer or {None: None}

        self._period, self._p = indexer.popitem()

        self.freq = freq or generic.default_freq(**indexer)

        identifier = 'q{m.window}{m.mode}{m.t}{m.period_id}'
        title = '{m.t}-year return period {m.mode} {m.period} {m.window}-day streamflow'
        description = '{m.freq_name} {m.t} {m.window}-day flow of {m.t}-{m.freq_unit} recurrence during {' \
                       'm.period}'
        long_name = '{m.t}-year return period {m.mode} {m.period} {m.window}-day flow'

        body_template = """from xclim.generic import frequency_analysis as func"""

        return self._generate(identifier.format(m=self),
                              title.format(m=self),
                              description.format(m=self),
                              long_name=long_name.format(m=self),
                              body_template=body_template,
                              defaults=dict(dist='gumbel_r'),
                              fixed=dict(t=t, window=window, mode=mode, freq=freq))

    @property
    def op(self):
        if isinstance(self._op, str):
            return self._op
        else:
            return self._op.__name__

    @property
    def freq_unit(self):
        if self.freq.startswith(('Y', 'A')):
            return 'year'
        elif self.freq.startswith('Q'):
            return 'season'
        elif self.freq.startswith('M'):
            return 'month'

    @property
    def freq_name(self):
        if self.freq.startswith(('Y', 'A')):
            return 'annual'
        elif self.freq.startswith('Q'):
            return 'seasonal'
        elif self.freq.startswith('M'):
            return 'monthly'

    @property
    def mode_name(self):
        return self._mode_names[self.mode]

    @property
    def period_id(self):
        if self._period == 'season':
            return self._rseasons[str(self._p)]
        elif self._period == 'month':
            return ''.join([calendar.month_abbr[m] for m in np.atleast_1d(self._p)])
        elif self._period is None:
            return ''

    @property
    def period(self):
        """Return the period name."""
        if self._period == 'season':
            return self._season_names[self.period_id]
        elif self._period == 'month':
            return '-'.join([calendar.month_name[p] for p in np.atleast_1d(self._p)])
        elif self._period is None:
            return 'year'

    def _generate(self, identifier, title, description, body_template, notes='',defaults={}, fixed={},
                  standard_name='', long_name='', cell_methods=''):
        """Return a Streamflow instance based on the `generic.frequency_analysis` function.

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
        standard_name : str
          Standard name of output variable.
        long_name : str
          Long name of output variable.
        """

        # TODO: create parameter lines in docstring.
        params = ""
        output = ""
        args = ['q', ] + list(defaults.keys())
        gfa_args = ["da=q", ] + \
                   ["{0}={0}".format(k) for k in defaults.keys()] + \
                   ["{0}={1}".format(k, v) for (k, v) in fixed.items()]

        body = body_template + "\nfunc({})".format(', '.join(gfa_args))

        f = FunctionBuilder(name=identifier.lower(),
                            doc=self.docstring_template.format(**{'identifier': identifier,
                                                             'title': title,
                                                             'description': description,
                                                             'params': params,
                                                             'output': output,
                                                             'notes': notes}),
                            body=body,
                            args=args,
                            defaults=tuple(defaults.values())
                            )

        s = Streamflow(identifier=identifier,
                       title=title,
                       standard_name=standard_name,
                       long_name=long_name,
                       cell_methods=cell_methods,
                       compute=f.get_func(),
                       )

        return s

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


Q = QIndGen()

q1max2sp = Q.fa(1, 'max', 2, season='MAM')

q1max20sp = Q.fa(1, 'max', 20, season='MAM')

q14max2sp = Q.fa(14, 'max', 2, season='MAM')

q14max20sp = Q.fa(14, 'max', 20, season='MAM')

q1max2sua = Q.fa(1, 'max', 2, season=['JJA', 'SON'])

q1max20sua = Q.fa(1, 'max', 20, season=['JJA', 'SON'])

q7min2su = Q.fa(7, 'min', 2, season='JJA')

q7min10su = Q.fa(7, 'min', 10, season='JJA')

q30min5su = Q.fa(30, 'min', 5, season='JJA')

q7min10w = Q.fa(7, 'min', 10, season='DJF')

q30min5w = Q.fa(30, 'min', 5, season='DJF')

doy_q1maxsp = Q.sop(generic.doymax, season='MAM')

qmax = Q.sop('max')

qmaxsp = Q.sop('max', season='MAM')

qmean = Q.sop('mean')

qmeansp = Q.sop('mean', season='MAM')


#

# def q_monthly_mean(q):
#     """Function computing the monthly averages of streamflow every year
#
#     :param q:
#     :return:
#     """
#     # split data into different months and compute mean
#     qrm = q.resample(time='MS').mean(dim='time')
#
#     return qrm
#
#
# def q_season_mean(q, seasons, freq='AS-DEC'):
#     """Function that computes yearly mean of streamflow over wanted seasons
#
#     :param q:
#     :param seasons:
#     :param freq:
#     :return:
#     """
#
#     # select streamflow over wanted seasons
#     qsr = q.sel(time=q.time.dt.season.isin(seasons)).dropna(dim='time')
#
#     # split the years and make the mean
#     qsm = qsr.resample(time=freq).mean(dim='time')
#
#     return qsm
#
#
# def _qavg(q):
#     return q_season_mean(q, seasons='DJF MAM JJA SON'.split(), freq='AS-JAN')
#
#
# def _qavgwsp(q):
#     return q_season_mean(q, seasons='DJF MAM'.split(), freq='AS-DEC')
#
#
# def _qavgsua(q):
#     return q_season_mean(q, seasons='JJA SON'.split(), freq='AS-DEC')
#
#
# def _qavg_1_12(q):
#     qm = q_monthly_mean(q)
#     return qm
#
#
# qavg = Streamflow_time_operator(identifier='qavg',
#                                 units='m^3 s-1',
#                                 standard_name='',
#                                 long_name='',
#                                 description="",
#                                 cell_methods='',
#                                 compute=_qavg,
#                                 freq='AS-JAN',
#                                 time_operator='mean'
#                                 )
#
# qavgwsp = Streamflow_time_operator(identifier='qavgwsp',
#                                    units='m^3 s-1',
#                                    standard_name='',
#                                    long_name='',
#                                    description="",
#                                    cell_methods='',
#                                    compute=_qavgwsp,
#                                    freq='AS-DEC',
#                                    time_operator='mean'
#                                    )
#
# qavgsua = Streamflow_time_operator(identifier='qavgsua',
#                                    units='m^3 s-1',
#                                    standard_name='',
#                                    long_name='',
#                                    description="",
#                                    cell_methods='',
#                                    compute=_qavgsua,
#                                    freq='AS-DEC',
#                                    time_operator='mean'
#                                    )
#
# qavg_1_12 = Streamflow_time_operator(identifier='avg_1_12',
#                                      units='m^3 s-1',
#                                      standard_name='',
#                                      long_name='',
#                                      description="",
#                                      cell_methods='',
#                                      compute=_qavg_1_12,
#                                      freq='MS',
#                                      time_operator='monthly_annual_cycle'
#                                      )

# -*- coding: utf-8 -*-
import numpy as np

from xclim.indices import base_flow_index
from xclim import checks
from xclim.utils import Indicator, wrapped_partial
from xclim import generic
# from boltons.funcutils import FunctionBuilder
# import calendar


class Streamflow(Indicator):
    units = 'm^3 s-1'
    context = 'hydro'
    standard_name = 'discharge'

    @staticmethod
    def compute(*args, **kwds):
        pass

    def cfprobe(self, q):
        checks.check_valid(q, 'standard_name', 'streamflow')


class Stats(Streamflow):
    def missing(self, *args, **kwds):
        """Return whether an output is considered missing or not."""
        from functools import reduce

        indexer = kwds['indexer']
        freq = kwds['freq'] or generic.default_freq(**indexer)

        miss = (checks.missing_any(da, freq, **indexer) for da in args)
        return reduce(np.logical_or, miss)


# We need to disable the missing value check because the output here is not a time series.
class FA(Streamflow):
    def missing(self, *args, **kwds):
        """Return whether an output is considered missing or not."""
        return False


base_flow_index = Streamflow(identifier='base_flow_index',
                             units='',
                             long_name="Base flow index",
                             compute=base_flow_index)


freq_analysis = FA(identifier='freq_analysis',
                   var_name='q{window}{mode}{indexer}',
                   long_name='N-year return period {mode} {indexer} {window}-day flow',
                   description="Streamflow frequency analysis for the {mode} {indexer} {window}-day flow "
                               "estimated using the {dist} distribution.",
                   compute=generic.frequency_analysis)


stats = Stats(identifier='stats',
              var_name='q{indexer}{op}',
              long_name='{freq} {op} of {indexer} daily flow ',
              description="{freq} {op} of {indexer} daily flow",
              compute=generic.select_resample_op)


doy_qmax = Streamflow(identifier='doy_qmax',
                      var_name='q{indexer}_doy_qmax',
                      long_name='Day of the year of the maximum over {indexer}',
                      description='Day of the year of the maximum over {indexer}',
                      units='',
                      _partial=True,
                      compute=wrapped_partial(generic.select_resample_op, op=generic.doymax))


doy_qmin = Streamflow(identifier='doy_qmin',
                      var_name='q{indexer}_doy_qmin',
                      long_name='Day of the year of the minimum over {indexer}',
                      description='Day of the year of the minimum over {indexer}',
                      units='',
                      _partial=True,
                      compute=wrapped_partial(generic.select_resample_op, op=generic.doymin))

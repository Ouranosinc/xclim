# -*- coding: utf-8 -*-
import numpy as np
from xclim import checks
from xclim import indices as _ind
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


freq_analysis = Streamflow(identifier='q{window}{mode}{indexer}',
                           long_name='N-year return period {mode} {indexer} {window}-day flow',
                           description="Streamflow frequency analysis for the {mode} {indexer} {window}-day flow "
                                       "estimated using the {dist} distribution.",
                           compute=generic.frequency_analysis)


stats = Streamflow(identifier='q{indexer}{op}',
                   long_name='{freq} {op} of {indexer} daily flow ',
                   description="{freq} {op} of {indexer} daily flow",
                   compute=generic.select_resample_op)


doy_qmax = Streamflow(identifier='q{indexer}_doy_qmax',
                      long_name='Day of the year of the max over {indexer}',
                      description='Day of the year of the max over {indexer}',
                      units='',
                      _partial=True,
                      compute=wrapped_partial(generic.select_resample_op, op=generic.doymax))

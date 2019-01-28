from . import checks
from . import indices as _ind
from .utils import Indicator, generic_frequency_analyis
from functools import partial, update_wrapper

def wrapped_partial(func, *args, **kwargs):
    partial_func = partial(func, *args, **kwargs)
    update_wrapper(partial_func, func)
    return partial_func


class Streamflow(Indicator):
    required_units = 'm^3 s-1'
    units = 'm^3 s-1'
    context = 'hydro'
    standard_name = 'streamflow'
    _partial = True


    def cfprobe(self, q):
        checks.check_valid(q, 'standard_name', 'streamflow')


base_flow_index = Streamflow(identifier='base_flow_index',
                             units='',
                             standard_name='',
                             compute=_ind.base_flow_index)

q_max = Streamflow(identifier='q_max',
                   )


q1max2sp = Streamflow(identifier='Q1max2Sp',
                      title='2-year return period maximum spring flood peak',
                      description='Annual maximum [max] daily flow [Q1] of 2-year recurrence [2] in spring [Sp]',
                      compute=wrapped_partial(generic_frequency_analyis, freq='QS-MAR', t=2, dist='gumbel_r',
                                              mode='high'))

from . import checks
from . import indices as _ind
from .utils import Indicator
from functools import partial
from .stats import fa

class Streamflow(Indicator):
    required_units = 'm3 s-1'
    units = 'm3 s-1',
    context = 'hydro'
    standard_name = 'water_volume_transport_in_river_channel',

    def cfprobe(self, q):
        checks.check_valid(q, 'standard_name', 'water_volume_transport_in_river_channel')


base_flow_index = Streamflow(identifier='base_flow_index',
                             units='',
                             standard_name='',
                             compute=_ind.base_flow_index)

q_max = Streamflow(identifier='q_max',
                   )


q1max2sp = Streamflow(identifier='Q1max2Sp',
                      title='2-year return period maximum spring flood peak',
                      description='Annual maximum [max] daily flow [Q1] of 2-year recurrence [2] in spring [Sp]',
                      compute=partial(fa, T=2, dist='gumbel_r', mode='high'))

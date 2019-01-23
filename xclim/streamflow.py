from . import checks
from . import indices as _ind
from .utils import Indicator


class Streamflow(Indicator):
    required_units = 'm3 s-1'
    context = 'hydro'

    def cfprobe(self, q):
        checks.check_valid(q, 'standard_name', 'water_volume_transport_in_river_channel')


base_flow_index = Streamflow(identifier='base_flow_index',
                             units='',
                             standard_name='',
                             compute=ind.base_flow_index)

q_max = Streamflow(identifier='q_max',
                   units='m3 s-1',
                   standard_name='water_volume_transport_in_river_channel',
                   compute=ind.
                   )




from . import checks
from . import indices as _ind
from .utils import UnivariateIndicator


class BaseFlowIndex(UnivariateIndicator):
    identifier = 'tx_max'
    units = 'm3 s-1'
    required_units = 'K'
    long_name = 'streamflow'  # discharge ?
    standard_name = 'water_volume_transport_in_river_channel'
    description = 'Maximum daily maximum temperature over period.'
    keywords = ''

    def compute(self, q, freq='YS'):
        return _ind.base_flow_index(q, freq)

    def cfprobe(self, q):
        checks.check_valid(q, 'standard_name', 'water_volume_transport_in_river_channel')

    def validate(self, q):
        checks.assert_daily(q)

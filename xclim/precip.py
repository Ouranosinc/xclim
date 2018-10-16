from . import indices as _ind
from .utils import UnivariateIndicator


class R1Max(UnivariateIndicator):
    identifier = 'r1max'
    long_name = ''
    units = 'mm'
    required_units = 'mm'

    def compute(self, da, freq='YS'):
        return _ind.max_1day_precipitation_amount(da, freq)

from . import checks
from . import indices as _ind
from .utils import UnivariateIndicator

# TODO: Should we reference the standard vocabulary we're using ?
# E.g. http://vocab.nerc.ac.uk/collection/P07/current/BHMHISG2/


class TGMean(UnivariateIndicator):
    identifier = 'tg_mean'
    units = 'degK'
    required_units = 'degK'
    long_name = "{freq} mean temperature"
    standard_name = "{freq} mean temperature"
    description = "{freq} of faily mean temperature."
    keywords = ''

    def compute(self, da, freq='YS'):
        return _ind.tg_mean(da, freq)

    def validate(self, da):
        checks.assert_daily(da)

    def cfprobe(self, da):
        checks.check_valid(da, 'cell_methods', 'time: mean within days')
        checks.check_valid(da, 'standard_name', 'air_temperature')


class TxMax(UnivariateIndicator):
    identifier = 'tx_max'
    units = 'degK'
    required_units = 'degK'
    long_name = 'Maximum temperature'
    standard_name = 'tasmax'
    description = 'Maximum daily maximum temperature.'
    keywords = ''

    def compute(self, da, freq='YS'):
        return _ind.tx_max(da, freq)

    def cfprobe(self, da):
        checks.check_valid(da, 'cell_methods', 'time: maximum within days')
        checks.check_valid(da, 'standard_name', 'air_temperature')

    def validate(self, da):
        checks.assert_daily(da)

    def missing(self, da, freq):
        """An aggregated value is missing if any value in the group is missing."""
        g = da.notnull().resample(time=freq)
        return g.sum(dim='time')


class ColdSpellDurationIndex(UnivariateIndicator):
    standard_name = 'cold_spell_duration_index',
    units = 'days'

    def cfprobe(self, da):
        checks.check_valid(da, 'cell_methods', 'time: minimum within days')

    def compute(self, da, tn10, freq='YS'):
        _ind.cold_spell_duration_index(da, tn10, freq)


class TxMin(TxMax):
    identifier = 'tx_min'
    long_name = 'Minimum maximum temperature'
    standard_name = 'tx_min'
    description = 'Minimum daily maximum temperature over the period'
    cell_methods = 'time: minimum within {freq}'

    def compute(self, da, freq='YS'):
        return _ind.tx_min(da, freq)

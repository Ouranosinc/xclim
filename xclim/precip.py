from . import indices as _ind
from .utils import UnivariateIndicator


class R1Max(UnivariateIndicator):
    identifier = 'r1max'
    long_name = ''
    units = 'mm'
    required_units = 'mm'

    def compute(self, da, freq='YS'):
        return _ind.max_1day_precipitation_amount(da, freq)


class WetDays(UnivariateIndicator):
    identifier = 'wet_days'
    standard_name = 'wet_days'
    units = 'days'
    required_units = 'mm/day'
    long_name = 'number of wet days per period'
    description = 'Number of days with daily precipitation over {thresh} mm'

    def compute(self, da, thresh=1, freq='YS'):
        return _ind.wet_days(da, thresh, freq)


class DailyIntensity(UnivariateIndicator):
    identifier = 'daily_pr_intensity'
    standard_name = 'daily_intensity'
    long_name = 'daily precipitation intensity over wet days'
    units = 'mm/day'
    required_units = 'mm/day'
    description = "Average precipitation for days with daily precipitation over {thresh} mm"

    def compute(self, da, thresh=1, freq='YS'):
        return _ind.daily_intensity(da, thresh, freq)

class MaxNDayPrecipitationAmount(UnivariateIndicator):
    identifier = 'max_n_day_precipitation_amount'
    standard_name = 'maximum_{window}_day_total_precipitation'
    long_name = 'maximum {window} day total precipitation',
    units = 'mm'
    required_units = 'mm/day'

    def compute(self, da, window=5, freq='YS'):
        return _ind.max_n_day_precipitation_amount(da, window, freq)




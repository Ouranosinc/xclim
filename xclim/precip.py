from . import indices as _ind
from .utils import UnivariateIndicator


class R1Max(UnivariateIndicator):
    identifier = 'r1max'
    long_name = ''
    units = 'mm'
    required_units = 'mm'

    compute = _ind.max_1day_precipitation_amount


class WetDays(UnivariateIndicator):
    identifier = 'wet_days'
    standard_name = 'wet_days'
    units = 'days'
    required_units = 'mm/day'
    long_name = 'number of wet days per period'
    description = 'Number of days with daily precipitation over {thresh} mm'

    compute =_ind.wet_days


class DailyIntensity(UnivariateIndicator):
    identifier = 'daily_pr_intensity'
    standard_name = 'daily_intensity'
    long_name = 'daily precipitation intensity over wet days'
    units = 'mm/day'
    required_units = 'mm/day'
    description = "Average precipitation for days with daily precipitation over {thresh} mm"

    compute = _ind.daily_intensity


class MaxNDayPrecipitationAmount(UnivariateIndicator):
    identifier = 'max_n_day_precipitation_amount'
    standard_name = 'maximum_{window}_day_total_precipitation'
    long_name = 'maximum {window} day total precipitation',
    units = 'mm'
    required_units = 'mm/day'

    compute = _ind.max_n_day_precipitation_amount

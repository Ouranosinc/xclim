# noqa: D205,D400
"""
Data checks
===========

Utilities designed to check the validity of data inputs.
"""
import xarray as xr

from .options import datacheck
from .utils import ValidationError


@datacheck
def check_freq(var: xr.DataArray, freq: str, strict: bool = True):
    """Raise an error if not series has not the expected temporal frequency or is not monotonically increasing.

    Parameters
    ----------
    var : xr.DataArray
      Input array.
    freq : str
      The temporal frequency defined using the Pandas frequency strings, e.g. 'A', 'M', 'D', 'H', 'T',
      'S'. Note that a 3-hourly time series is declared as '3H'.
    strict : bool
      Whether or not multiples of the frequency are considered invalid. With `strict` set to False, a '3H' series
      will not raise an error if freq is set to 'H'.
    """
    v_freq = xr.infer_freq(var.time)
    if v_freq != freq:
        if (freq in v_freq) and not strict:
            return
        raise ValidationError(
            "Time series has temporal frequency `{v_freq}`, expected `{freq}`."
        )


@datacheck
def check_daily(var: xr.DataArray):
    """Raise an error if not series has a frequency other that daily, or is not monotonically increasing.

    Note that this does not check for gaps in the series.
    """
    if xr.infer_freq(var.time) != "D":
        raise ValidationError("time series is not recognized as daily.")

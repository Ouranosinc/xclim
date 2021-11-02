# noqa: D205,D400
"""
Data checks
===========

Utilities designed to check the validity of data inputs.
"""
from typing import Sequence, Union

import xarray as xr

from .calendar import compare_offsets, parse_offset
from .options import datacheck
from .utils import ValidationError


@datacheck
def check_freq(var: xr.DataArray, freq: Union[str, Sequence[str]], strict: bool = True):
    """Raise an error if not series has not the expected temporal frequency or is not monotonically increasing.

    Parameters
    ----------
    var : xr.DataArray
      Input array.
    freq : str or sequence of str
      The expected temporal frequencies, using Pandas frequency terminology ({'A', 'M', 'D', 'H', 'T', 'S', 'L', 'U'} and multiples thereof).
      To test strictly for 'W', pass '7D' with `strict=True`.
      This ignores the start flag and the anchor (ex: 'AS-JUL' will validate against 'Y').
    strict : bool
      Whether or not multiples of the frequencies are considered invalid. With `strict` set to False, a '3H' series
      will not raise an error if freq is set to 'H'.
    """
    if isinstance(freq, str):
        freq = [freq]
    exp_base = [parse_offset(frq)[1] for frq in freq]
    v_freq = xr.infer_freq(var.time)
    if v_freq is None:
        raise ValidationError("Unable to infer the frequency of the time series.")
    v_base = parse_offset(v_freq)[1]
    if v_base not in exp_base or (
        strict and all(compare_offsets(v_freq, "!=", frq) for frq in freq)
    ):
        raise ValidationError(
            f"Frequency of time series not {'strictly' if strict else ''} in {freq}"
        )


@datacheck
def check_daily(var: xr.DataArray):
    """Raise an error if not series has a frequency other that daily, or is not monotonically increasing.

    Note that this does not check for gaps in the series.
    """
    if xr.infer_freq(var.time) != "D":
        raise ValidationError(
            "time series is not recognized as daily. You can quiet this error by setting `data_validation` to 'warn' or 'log', in `xclim.set_options`."
        )

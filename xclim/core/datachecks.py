"""
Data checks
===========

Utilities designed to check the validity of data inputs.
"""
# import datetime as dt
# import numpy as np
# import pandas as pd
import xarray as xr

from .options import datacheck
from .utils import ValidationError


@datacheck
def check_daily(var):
    r"""Assert that the series is daily and monotonic (no jumps in time index).

    A ValidationError is raised otherwise."""
    if xr.infer_freq(var.time.to_pandas()) != "D":
        raise ValidationError("time series is not recognized as daily.")

    # Check that the series does not go backward in time
    if not var.indexes["time"].is_monotonic_increasing:
        raise ValidationError("time index is not monotonically increasing.")

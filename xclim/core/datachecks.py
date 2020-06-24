"""
Data checks
===========

Utilities designed to check the validity of data inputs.
"""
import datetime as dt

import numpy as np
import pandas as pd

from .options import datacheck
from .utils import ValidationError


@datacheck
def check_daily(var):
    r"""Assert that the series is daily and monotonic (no jumps in time index).

    A ValueError is raised otherwise."""

    t0, t1 = var.time[:2]

    # This won't work for non-standard calendars. Needs to be implemented in xarray. Comment for now
    if isinstance(t0.values, np.datetime64):
        if pd.infer_freq(var.time.to_pandas()) != "D":
            raise ValidationError("time series is not recognized as daily.")

    # Check that the first time step is one day.
    if np.timedelta64(dt.timedelta(days=1)) != (t1 - t0).data:
        raise ValidationError("time series is not daily.")

    # Check that the series does not go backward in time
    if not var.time.to_pandas().is_monotonic_increasing:
        raise ValidationError("time index is not monotonically increasing.")

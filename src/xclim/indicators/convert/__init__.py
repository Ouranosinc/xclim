"""
Conversion Indicators
=====================

This submodule contains indicators that converts CF-compliant variables from one to another.
For example, converting wind speed in cardinal directions to a vector magnitude and direction,
or converting snow depth to snow water equivalent. It also includes indicators that approximate
variables from multiple variables, such as calculating the mean temperature from daily maximum
and minimum temperatures.
"""

from __future__ import annotations

from ._conversion import *

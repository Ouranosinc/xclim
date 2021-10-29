# noqa: D205,D400
"""
Atmospheric indicators
======================

While the `indices` module stores the computing functions, this module defines Indicator classes and instances that
include a number of functionalities, such as input validation, unit conversion, output meta-data handling,
and missing value masking.

The concept followed here is to define Indicator subclasses for each input variable, then create instances
for each indicator.

"""
from ._conversion import *
from ._precip import *
from ._preciptemp import *
from ._temperature import *
from ._wind import *

"""
The Indicator object
====================

The `Indicator` class wraps computations with pre- and post-processing functionality. Prior to computations,
the class runs data and metadata health checks. After computations, the class masks values that should be considered
missing and adds metadata attributes to the output.

There are many ways to construct indicators. A good place to start is
`this notebook <notebooks/extendxclim.ipynb#Defining-new-indicators>`_.
"""

from xclim.core.indicator._indicator import *

"""
==============
SDBA submodule
==============

.. warning::

    The SDBA submodule is in the process of being split from `xclim` in order to facilitate development and effective
    maintenance of the SDBA utilities. The `xclim.sdba` functionality will change in the future.
    For more information, please visit https://xsdba.readthedocs.io/en/latest/.
"""

from __future__ import annotations

import warnings

from . import adjustment, detrending, measures, processing, properties, utils
from .adjustment import *
from .base import Grouper
from .processing import stack_variables, unstack_variables

warnings.warn(
    "The SDBA submodule is in the process of being split from `xclim` in order to facilitate development and effective "
    "maintenance of the SDBA utilities. The `xclim.sdba` functionality will change in the future. "
    "For more information, please visit https://xsdba.readthedocs.io/en/latest/.",
    FutureWarning,
)

# TODO: ISIMIP ? Used for precip freq adjustment in biasCorrection.R
# Hempel, S., Frieler, K., Warszawski, L., Schewe, J., & Piontek, F. (2013). A trend-preserving bias correction &ndash;
# The ISI-MIP approach. Earth System Dynamics, 4(2), 219–236. https://doi.org/10.5194/esd-4-219-2013
# If SBCK is installed, create adjustment classes wrapping SBCK's algorithms.
if hasattr(adjustment, "_generate_SBCK_classes"):
    for cls in adjustment._generate_SBCK_classes():
        adjustment.__dict__[cls.__name__] = cls

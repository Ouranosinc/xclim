# This file is kept for backward compatibility. It will be removed in a later version.
"""Module that doesn't even go here."""
from warnings import warn

from .cfchecks import *
from .datachecks import *
from .missing import *

warn(
    (
        "Submodule 'checks' is deprecated in favor of 'cfchecks', 'datachecks'"
        " and 'missing'. It will be removed in xclim 0.19"
    ),
    FutureWarning,
    stacklevel=2,
)

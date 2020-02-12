import warnings

from .land import base_flow_index
from .land import doy_qmax
from .land import doy_qmin
from .land import fit
from .land import freq_analysis
from .land import stats

# TODO: Remove this file as per DeprecationWarning advises
warnings.warn(
    f"{__name__} will be deprecated in xclim v0.13.x. Please begin using the 'xclim.land' module.",
    DeprecationWarning,
    stacklevel=2,
)

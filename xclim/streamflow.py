import warnings

from .land import base_flow_index
from .land import doy_qmax
from .land import doy_qmin
from .land import fit
from .land import freq_analysis
from .land import stats

warnings.warn(
    "{} will be deprecated in xclim v0.13.x. Please begin using the 'xclim.land' module.".format(
        __name__,
    ),
    DeprecationWarning,
    stacklevel=2,
)

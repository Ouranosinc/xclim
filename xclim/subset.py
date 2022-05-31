"""
Mock subset module for API compatibility.

See Also
--------
clisops.core.subset
"""
from __future__ import annotations

import os
import warnings

try:
    from clisops.core.subset import *  # noqa
    from clisops.core.subset import __all__  # noqa

    __all__ = [x for x in __all__]

    warnings.warn(
        f"{__name__} is deprecated in xclim v0.19.1-beta. "
        f"Please take note that xclim presently exposes the 'clisops' library subsetting API "
        f"via `from clisops.core import subset`. This functionality may eventually change.",
        DeprecationWarning,
        stacklevel=2,
    )

except ImportError as e:
    if not os.getenv("READTHEDOCS"):
        raise ImportError(
            f"{__name__} is deprecated in xclim v0.19.1-beta. "
            f"Subset functions are now dependent on the `clisops` library. This library can be installed via "
            f'`pip install xclim["gis"]`, `pip install clisops` or `conda install clisops`.'
        ) from e
    else:
        pass

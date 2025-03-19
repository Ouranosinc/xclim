"""Statistical downscaling and bias adjustment submodule."""

import warnings

try:
    from xsdba import *

    warnings.warn(
        "The `xclim.sdba` module has been split into its own package `xsdba`. "
        "For the time being, `xclim.sdba` will import `xsdba` to allow for backwards compatibility. "
        "This behaviour may change in the future. "
        "For more information, see: https://xsdba.readthedocs.io/en/stable/xclim_migration_guide.html"
    )
except ImportError:
    error_msg = (
        "The `xclim.sdba` module has been split into its own package: `xsdba`. "
        "Run `pip install xclim[extras]` or install `xsdba` via `pip` or `conda`. "
        "This will allow you to use the `xclim.sdba` module as before, though this behaviour may eventually change. "
        "For more information, see: https://xsdba.readthedocs.io/en/stable/xclim_migration_guide.html"
    )
    raise ImportError(error_msg)

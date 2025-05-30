"""
Statistical downscaling and bias adjustment submodule.

This module is a placeholder for the `xclim.sdba` submodule, which has been split into its own package `xsdba`.
"""

import warnings

try:
    from xsdba import *  # pylint: disable=wildcard-import,unused-wildcard-import

    warnings.warn(
        "The `xclim.sdba` module has been split into its own package `xsdba`. "
        "Users are encouraged to use `xsdba` directly. "
        "For the time being, `xclim.sdba` will import `xsdba` to allow for API compatibility. "
        "This behaviour may change in the future. "
        "For more information, see: https://xsdba.readthedocs.io/en/stable/xclim_migration_guide.html"
    )
except ImportError as err:
    error_msg = (
        "The `xclim.sdba` module has been split into its own package: `xsdba`. "
        "Run `pip install xclim[extras]` or install `xsdba` via `pip` or `conda`. "
        "Users are encouraged to use `xsdba` directly. "
        "For the time being, `xclim.sdba` will import `xsdba` to allow for API compatibility. "
        "This behaviour may change in the future. "
        "For more information, see: https://xsdba.readthedocs.io/en/stable/xclim_migration_guide.html"
    )
    raise ImportError(error_msg) from err

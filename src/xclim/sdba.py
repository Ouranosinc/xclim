"""Statistical downscaling and bias adjustment submodule."""

try:
    from xsdba import *
except ModuleNotFoundError:
    error_msg = (
        "The `xclim.sdba` module has been split into its own package: `xsdba`. "
        "Run `pip install xclim[extras]` or install `xsdba` via `pip` or `conda`. "
        "This will allow you to use the `xclim.sdba` module as before, though this behaviour may eventually change. "
        "For more information, see: https://xsdba.readthedocs.io/en/stable/xclim_migration_guide.html"
    )
    raise ModuleNotFoundError(error_msg)

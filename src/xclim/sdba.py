"""Statistical downscaling and bias adjustment sub-module."""

try:
    from xsdba import *
except ModuleNotFoundError as e:
    error_msg = (
        f"{str(e)}. `sdba` was split from `xclim` in its own submodule `xsdba`. Use conda or pip to install `xsdba`."
    )
    raise ModuleNotFoundError(error_msg)

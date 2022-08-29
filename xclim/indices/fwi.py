# noqa: D100
import warnings

warnings.warn(
    "The `fwi` module is deprecated in xclim v0.37.18-beta and the `fwi` indice and indicator (`fire_weather_indexes`) "
    "has been renamed to `cffwis_indices` to better support international collaboration. The `fwi` submodule alias "
    "will be removed in xclim v0.39.\n"
    "Please take note that xclim now offers a dedicated `fire` submodule under `xclim.indices` that houses all "
    "fire-based indices.",
    DeprecationWarning,
    stacklevel=2,
)

from .fire import fire_weather_indexes  # noqa
from .fire._cffwis import *  # noqa

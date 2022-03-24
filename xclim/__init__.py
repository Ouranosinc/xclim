"""Climate indices computation package based on Xarray."""
import logging
import sys
import warnings
from importlib.resources import contents, path

from loguru import logger

from xclim.core import units
from xclim.core.indicator import build_indicator_module_from_yaml
from xclim.core.locales import load_locale
from xclim.core.options import set_options
from xclim.indicators import atmos, land, seaIce

__author__ = """Travis Logan"""
__email__ = "logan.travis@ouranos.ca"
__version__ = "0.34.2-beta"


# On import, clisops sets a root logger for console output. This is huge annoyance to deal with.
# For upstream fix discussion, see: https://github.com/roocs/clisops/pull/216
if "clisops" in sys.modules:
    root_logger = logging.getLogger()
    root_logger.removeHandler(root_logger.handlers[0])

# Inject warnings from warnings.warn into loguru
showwarning_ = warnings.showwarning


def showwarning(message, *args, **kwargs):
    logger.warning(message)
    showwarning_(message, *args, **kwargs)


warnings.showwarning = showwarning


class PropagateHandler(logging.Handler):
    def emit(self, record):
        logging.getLogger(record.name).handle(record)


# Gather logged events from standard logging
class InterceptHandler(logging.Handler):
    def emit(self, record):
        # Get corresponding Loguru level if it exists
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where originated the logged message
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


# Synchronize logged events between standard logging and loguru, then deactivate their handlers
logging.basicConfig(handlers=[InterceptHandler()], level=logging.NOTSET)
config = dict(
    handlers=[
        dict(
            sink=PropagateHandler(),
            format="{message}",
            filter=lambda record: "emit" in record["extra"],
        )
    ]
)
logger.configure(**config)

# Load official locales
for filename in contents("xclim.data"):
    # Only select <locale>.json and not <module>.<locale>.json
    if filename.endswith(".json") and filename.count(".") == 1:
        locale = filename.split(".")[0]
        with path("xclim.data", filename) as f:
            load_locale(f, locale)


# Virtual modules creation:
with path("xclim.data", "icclim.yml") as f:
    build_indicator_module_from_yaml(f.with_suffix(""), mode="raise")
with path("xclim.data", "anuclim.yml") as f:
    build_indicator_module_from_yaml(f.with_suffix(""), mode="raise")
with path("xclim.data", "cf.yml") as f:
    build_indicator_module_from_yaml(f.with_suffix(""), mode="raise")

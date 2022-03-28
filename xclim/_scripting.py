import logging
import sys
from typing import Union

from loguru import logger


class PropagateHandler(logging.Handler):
    """Propagate loguru logging events into the standard library logging handler."""

    def emit(self, record):
        """Emit message to standard logging."""
        logging.getLogger(record.name).handle(record)


class InterceptHandler(logging.Handler):
    """Gather logged events from standard logging and send them to the loguru logging handler."""

    def emit(self, record):
        """Emit message to loguru logging."""
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


def enable_synced_logger(level: Union[int, str] = logging.WARNING):
    """Synchronize logged events between standard logging and loguru.

    Warnings
    --------
    Loguru is an async-capable logger while standard logging is not. Be warned that the Global Interpreter Lock may
    cause threading problems if logging events occur within running spawned processes.
    """
    logging.basicConfig(handlers=[InterceptHandler()], level=logging.NOTSET)
    config = dict(
        handlers=[
            dict(
                sink=PropagateHandler(),
                filter=lambda record: "emit" in record["extra"],
                enqueue=True,
            ),
            dict(sink=sys.stdout, level=level),
        ]
    )
    logger.configure(**config)

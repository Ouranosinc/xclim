"""Global or contextual options for xclim, similar to xarray.set_options"""
import logging
from inspect import signature
from pathlib import Path
from warnings import warn

from boltons.funcutils import wraps

from xclim.core.utils import ValidationError

logging.captureWarnings(True)


METADATA_LOCALES = "metadata_locales"
VALIDATE_INPUTS = "validate_inputs"
CHECK_MISSING = "check_missing"
MISSING_OPTIONS = "missing_options"

MISSING_METHODS = {}

OPTIONS = {
    METADATA_LOCALES: [],
    VALIDATE_INPUTS: "warn",
    CHECK_MISSING: "any",
    MISSING_OPTIONS: MISSING_METHODS,
}

_VALIDATION_OPTIONS = frozenset(["log", "warn", "raise"])


def _valid_locales(locales):
    from xclim.locales import get_best_locale

    return all(
        [
            (isinstance(locale, str) and get_best_locale(locale) is not None)
            or (
                not isinstance(locale, str)
                and isinstance(locale[0], str)
                and (Path(locale[1]).is_file() or isinstance(locale[1], dict))
            )
            for locale in locales
        ]
    )


def _valid_missing_options(mopts):
    for meth, opts in mopts.items():
        if (
            meth not in MISSING_METHODS  # Method must be registered
            # All options must exist
            or any([opt not in MISSING_METHODS[meth] for opt in opts.keys()])
            # Method option validator, if it exists, must pass
            or (
                MISSING_METHODS[meth]["_validator"] is not None
                and not MISSING_METHODS[meth]["_validator"](
                    **{
                        k: v
                        for k, v in opts.items()
                        if k not in ["_func", "_validator"]
                    }
                )
            )
            # Must not overwrite validator and func
            or "_validator" in opts
            or "_func" in opts
        ):
            return False
    return True


_VALIDATORS = {
    METADATA_LOCALES: _valid_locales,
    VALIDATE_INPUTS: _VALIDATION_OPTIONS.__contains__,
    CHECK_MISSING: MISSING_METHODS.__contains__,
    MISSING_OPTIONS: _valid_missing_options,
}


def _set_missing_options(mopts):
    for meth, opts in mopts.items():
        MISSING_METHODS[meth].update(opts)


_SETTERS = {MISSING_OPTIONS: _set_missing_options}


def register_missing_method(name, validator=None):
    def _register_missing_method(func):
        sig = signature(func)
        opts = {
            name: param.default
            for name, param in sig.parameters.items()
            if name not in ["da", "freq", "indexer"]
        }
        MISSING_METHODS[name] = {"_func": func, "_validator": validator, **opts}
        return func

    return _register_missing_method


def check(func):
    @wraps(func)
    def _run_check(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except ValidationError as err:
            if OPTIONS[VALIDATE_INPUTS] == "log":
                logging.info(err.msg)
            elif OPTIONS[VALIDATE_INPUTS] == "warn":
                warn(err.msg, UserWarning, stacklevel=3)
            else:
                raise err

    return _run_check


class set_options:
    """Set options for xclim in a controlled context.

    Currently supported options:

    - ``metadata_locales"``:  List of IETF language tags or
        tuples of language tags and a translation dict, or
        tuples of language tags and a path to a json file defining translation
        of attributes.
      Default: ``[]``.
    - ``validate_inputs``: Whether to do nothing ('no'),  'raise' an error or
        'warn' the user on inputs that fail the checks in `xclim.core.checks`.
      Default: ``'warn'``.
    - ``check_missing``: How to check for missing data and flag computed indicators.
        Default available methods are "any", "wmo" and "pct". or None to skip missing checks.
        Missing method can be registered through the `xclim.core.checks.register_missing_method` decorator.
      Default: ``'any'``
    - ``missing_options``: Dictionary of options to pass to the missing method. Keys must the name of
        missing method and values must be mappings from option names to values.

    You can use ``set_options`` either as a context manager:

    >>> with xclim.set_options(metadata_locales=['fr']):
    ...     ...

    Or to set global options:

    >>> xclim.set_options(missing_options={'pct': {'tolerance': 0.04}})
    """

    def __init__(self, **kwargs):
        self.old = {}
        for k, v in kwargs.items():
            if k not in OPTIONS:
                raise ValueError(
                    "argument name %r is not in the set of valid options %r"
                    % (k, set(OPTIONS))
                )
            if k in _VALIDATORS and not _VALIDATORS[k](v):
                raise ValueError(f"option {k!r} given an invalid value: {v!r}")

            self.old[k] = OPTIONS[k]

        self.update(kwargs)

    def __enter__(self):
        return

    def update(self, kwargs):
        for k, v in kwargs.items():
            if k in _SETTERS:
                _SETTERS[k](v)
            else:
                OPTIONS[k] = v

    def __exit__(self, type, value, traceback):
        self.update(self.old)

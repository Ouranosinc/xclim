"""Global or contextual options for xclim, similar to xarray.set_options"""
import logging
from inspect import signature
from pathlib import Path
from warnings import warn

from boltons.funcutils import wraps

from xclim.core.utils import ValidationError

logging.captureWarnings(True)


METADATA_LOCALES = "metadata_locales"
DATA_VALIDATION = "data_validation"
CF_COMPLIANCE = "cf_compliance"
CHECK_MISSING = "check_missing"
MISSING_OPTIONS = "missing_options"

MISSING_METHODS = {}

OPTIONS = {
    METADATA_LOCALES: [],
    DATA_VALIDATION: "raise",
    CF_COMPLIANCE: "warn",
    CHECK_MISSING: "any",
    MISSING_OPTIONS: MISSING_METHODS,
}

_LOUDNESS_OPTIONS = frozenset(["log", "warn", "raise"])


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
                hasattr(MISSING_METHODS[meth]["_cls"], "validate")
                and not MISSING_METHODS[meth]["_cls"].validate(
                    **{k: v for k, v in opts.items() if k != "_cls"}
                )
            )
            # Must not overwrite cls
            or "_cls" in opts
        ):
            return False
    return True


_VALIDATORS = {
    METADATA_LOCALES: _valid_locales,
    DATA_VALIDATION: _LOUDNESS_OPTIONS.__contains__,
    CF_COMPLIANCE: _LOUDNESS_OPTIONS.__contains__,
    CHECK_MISSING: MISSING_METHODS.__contains__,
    MISSING_OPTIONS: _valid_missing_options,
}


def _set_missing_options(mopts):
    for meth, opts in mopts.items():
        MISSING_METHODS[meth].update(opts)


_SETTERS = {MISSING_OPTIONS: _set_missing_options}


def register_missing_method(name, validator=None):
    def _register_missing_method(cls):
        sig = signature(cls.is_missing)
        opts = {
            name: param.default
            for name, param in sig.parameters.items()
            if name not in ["self", "null", "count"]
        }
        MISSING_METHODS[name] = {"_cls": cls, **opts}
        return cls

    return _register_missing_method


def datacheck(func):
    @wraps(func)
    def _run_check(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except ValidationError as err:
            if OPTIONS[DATA_VALIDATION] == "log":
                logging.info(err.msg)
            elif OPTIONS[DATA_VALIDATION] == "warn":
                warn(err.msg, UserWarning, stacklevel=3)
            else:
                raise err

    return _run_check


def cfcheck(func):
    @wraps(func)
    def _run_check(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except ValidationError as err:
            if OPTIONS[CF_COMPLIANCE] == "log":
                logging.info(err.msg)
            elif OPTIONS[CF_COMPLIANCE] == "warn":
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
    - ``data_validation``: Whether to 'log',  'raise' an error or
        'warn' the user on inputs that fail the data checks in `xclim.core.checks`.
      Default: ``'raise'``.
    - ``cf_compliance``: Whether to 'log',  'raise' an error or
        'warn' the user on inputs that fail the CF compliance checks in `xclim.core.checks`.
      Default: ``'warn'``.
    - ``check_missing``: How to check for missing data and flag computed indicators.
        Default available methods are "any", "wmo", "pct" and "at_least_n".
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

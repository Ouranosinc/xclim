"""Global or contextual options for xclim, similar to xarray.set_options"""
from pathlib import Path

METADATA_LOCALES = "metadata_locales"
VALIDATE_INPUTS = "validate_inputs"
CHECK_MISSING = "check_missing"
MISSING_WMO = "missing_wmo"
MISSING_PCT = "missing_pct"


OPTIONS = {
    METADATA_LOCALES: [],
    VALIDATE_INPUTS: "warn",
    CHECK_MISSING: "any",
    MISSING_WMO: {"nm": 11, "nc": 5},
    MISSING_PCT: 0.05,
}

_VALIDATION_OPTIONS = frozenset(["log", "warn", "raise"])
MISSING_METHODS = {None: lambda da, freq: True}


def _valid_locales(locales):
    from xclim.locales import get_best_locale

    return all(
        [
            (isinstance(locale, str) and get_best_locale(locale) is None)
            or (
                not isinstance(locale, str)
                and isinstance(locale[1], str)
                and not Path(locale[1]).is_file()
            )
            for locale in locales
        ]
    )


def _valid_missing_wmo(opts):
    return opts.get("nm", 50) < 31 and opts.get("nc", 50) < 31


_VALIDATORS = {
    METADATA_LOCALES: _valid_locales,
    VALIDATE_INPUTS: _VALIDATION_OPTIONS.__contains__,
    CHECK_MISSING: MISSING_METHODS.__contains__,
    MISSING_WMO: _valid_missing_wmo,
    MISSING_PCT: lambda opt: (opt >= 0) and (opt <= 1),
}


def register_missing_method(name):
    def _register_missing_method(func):
        MISSING_METHODS[name] = func
        return func

    return _register_missing_method


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
      Default: ``'any'``
    - ``missing_wmo``: Dictionary of the 'nm' and 'nc' options for the "wmo" missing check.
        See `xclim.core.checks.missing_wmo`.
      Default: ``{'nm': 11, 'nc': 5}``
    - ``missing_pct``: Fraction of missing value that is tolerated by the "pct" missing check.
        See `xclim.core.checks.missing_pct`.
      Default: ``0.05``

    You can use ``set_options`` either as a context manager:

    >>> with xclim.set_options(metadata_locales=['fr']):
    ...     ...

    Or to set global options:

    >>> xclim.set_options(metadata_locales=['fr'])
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
        OPTIONS.update(kwargs)

    def __enter__(self):
        return

    def __exit__(self, type, value, traceback):
        OPTIONS.update(self.old)

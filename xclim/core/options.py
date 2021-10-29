"""
Options submodule
=================

Global or contextual options for xclim, similar to xarray.set_options.
"""
from inspect import signature
from typing import Callable, Dict

from boltons.funcutils import wraps

from .locales import _valid_locales
from .utils import ValidationError, raise_warn_or_log

METADATA_LOCALES = "metadata_locales"
DATA_VALIDATION = "data_validation"
CF_COMPLIANCE = "cf_compliance"
CHECK_MISSING = "check_missing"
MISSING_OPTIONS = "missing_options"
RUN_LENGTH_UFUNC = "run_length_ufunc"
SDBA_EXTRA_OUTPUT = "sdba_extra_output"
SDBA_ENCODE_CF = "sdba_encode_cf"

MISSING_METHODS: Dict[str, Callable] = dict()

OPTIONS = {
    METADATA_LOCALES: list(),
    DATA_VALIDATION: "raise",
    CF_COMPLIANCE: "warn",
    CHECK_MISSING: "any",
    MISSING_OPTIONS: dict(),
    RUN_LENGTH_UFUNC: "auto",
    SDBA_EXTRA_OUTPUT: False,
    SDBA_ENCODE_CF: False,
}

_LOUDNESS_OPTIONS = frozenset(["log", "warn", "raise"])
_RUN_LENGTH_UFUNC_OPTIONS = frozenset(["auto", True, False])


def _valid_missing_options(mopts):
    for meth, opts in mopts.items():
        cls = MISSING_METHODS.get(meth, None)
        if (
            cls is None  # Method must be registered
            # All options must exist
            or any([opt not in OPTIONS[MISSING_OPTIONS][meth] for opt in opts.keys()])
            # Method option validator must pass, default validator is always True.
            or not cls.validate(**opts)  # noqa
        ):
            return False
    return True


_VALIDATORS = {
    METADATA_LOCALES: _valid_locales,
    DATA_VALIDATION: _LOUDNESS_OPTIONS.__contains__,
    CF_COMPLIANCE: _LOUDNESS_OPTIONS.__contains__,
    CHECK_MISSING: lambda meth: meth != "from_context" and meth in MISSING_METHODS,
    MISSING_OPTIONS: _valid_missing_options,
    RUN_LENGTH_UFUNC: _RUN_LENGTH_UFUNC_OPTIONS.__contains__,
    SDBA_EXTRA_OUTPUT: lambda opt: isinstance(opt, bool),
    SDBA_ENCODE_CF: lambda opt: isinstance(opt, bool),
}


def _set_missing_options(mopts):
    for meth, opts in mopts.items():
        OPTIONS[MISSING_OPTIONS][meth].update(opts)


def _set_metadata_locales(locales):
    if isinstance(locales, str):
        OPTIONS[METADATA_LOCALES] = [locales]
    else:
        OPTIONS[METADATA_LOCALES] = locales


_SETTERS = {
    MISSING_OPTIONS: _set_missing_options,
    METADATA_LOCALES: _set_metadata_locales,
}


def register_missing_method(name: str) -> Callable:
    """Register missing method."""

    def _register_missing_method(cls):
        sig = signature(cls.is_missing)
        opts = {
            key: param.default if param.default != param.empty else None
            for key, param in sig.parameters.items()
            if key not in ["self", "null", "count"]
        }

        MISSING_METHODS[name] = cls
        OPTIONS[MISSING_OPTIONS][name] = opts
        return cls

    return _register_missing_method


def _run_check(func, option, *args, **kwargs):
    """Run function and customize exception handling based on option."""
    try:
        func(*args, **kwargs)
    except ValidationError as err:
        raise_warn_or_log(err, OPTIONS[option], stacklevel=4)


def datacheck(func: Callable) -> Callable:
    """Decorate functions checking data inputs validity."""

    @wraps(func)
    def run_check(*args, **kwargs):
        return _run_check(func, DATA_VALIDATION, *args, **kwargs)

    return run_check


def cfcheck(func: Callable) -> Callable:
    """Decorate functions checking CF-compliance of DataArray attributes.

    Functions should raise ValidationError exceptions whenever attributes are non-conformant.
    """

    @wraps(func)
    def run_check(*args, **kwargs):
        return _run_check(func, CF_COMPLIANCE, *args, **kwargs)

    return run_check


class set_options:
    """Set options for xclim in a controlled context.

    Currently supported options:

    - ``metadata_locales``:
      List of IETF language tags or tuples of language tags and a translation dict, or
      tuples of language tags and a path to a json file defining translation of attributes.
      Default: ``[]``.
    - ``data_validation``:
      Whether to 'log',  'raise' an error or 'warn' the user on inputs that fail the data checks in `xclim.core.datachecks`.
      Default: ``'raise'``.
    - ``cf_compliance``:
      Whether to 'log',  'raise' an error or 'warn' the user on inputs that fail the CF compliance checks in `xclim.core.cfchecks`.
      Default: ``'warn'``.
    - ``check_missing``:
      How to check for missing data and flag computed indicators.
      Default available methods are "any", "wmo", "pct", "at_least_n" and "skip".
      Missing method can be registered through the `xclim.core.options.register_missing_method` decorator.
      Default: ``'any'``
    - ``missing_options``:
      Dictionary of options to pass to the missing method. Keys must the name of
      missing method and values must be mappings from option names to values.
    - ``run_length_ufunc``:
      Whether to use the 1D ufunc version of run length algorithms or the dask-ready broadcasting version.
      Default is ``'auto'`` which means the latter is used for dask-backed and large arrays.
    - ``sdba_extra_output``:
      Whether to add diagnostic variables to outputs of sdba's `train`, `adjust`
      and `processing` operations.  Details about these additional variables are given in the object's
      docstring. When activated,  `adjust` will return a Dataset with `scen`  and those extra diagnostics
      For `processing` functions, see the doc, the output type might change, or not depending on the
      algorithm. Default: ``False``.
    - ``sdba_encode_cf``:
      Whether to encode cf coordinates in the ``map_blocks`` optimization most of adjustment methods are based on.
      This should have no impact on the results, but should run much faster in the graph creation phase.

    Examples
    --------
    You can use ``set_options`` either as a context manager:

    >>> import xclim
    >>> ds  = xr.open_dataset(path_to_tas_file).tas
    >>> with xclim.set_options(metadata_locales=['fr']):
    ...     out = xclim.atmos.tg_mean(ds)

    Or to set global options:

    >>> xclim.set_options(missing_options={'pct': {'tolerance': 0.04}})  # doctest: +SKIP
    <xclim.core.options.set_options object at ...>
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

        self._update(kwargs)

    def __enter__(self):
        """Context management."""
        return

    def _update(self, kwargs):
        """Update values."""
        for k, v in kwargs.items():
            if k in _SETTERS:
                _SETTERS[k](v)
            else:
                OPTIONS[k] = v

    def __exit__(self, type, value, traceback):
        """Context management."""
        self._update(self.old)

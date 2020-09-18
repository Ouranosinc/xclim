# -*- coding: utf-8 -*-
# noqa: D205,D400
"""
Indicator base classes
======================

The `Indicator` class wraps indices computations with pre- and post-processing functionality. Prior to computations,
the class runs data and metadata health checks. After computations, the class masks values that should be considered
missing and adds metadata attributes to the output object.

For more info on how to define new indicators see `here <notebooks/customize.ipynb#Defining-new-indicators>`_.
"""
import re
import warnings
import weakref
from collections import OrderedDict, defaultdict
from copy import deepcopy
from inspect import _empty, signature
from typing import Sequence, Union

import numpy as np
from boltons.funcutils import wraps
from xarray import DataArray

from xclim.indices.generic import default_freq

from . import datachecks
from .formatting import (
    AttrFormatter,
    default_formatter,
    merge_attributes,
    parse_doc,
    update_history,
)
from .locales import TRANSLATABLE_ATTRS, get_local_attrs, get_local_formatter
from .options import MISSING_METHODS, MISSING_OPTIONS, OPTIONS
from .units import convert_units_to, units

# Indicators registry
registry = {}  # Main class registry
_indicators_registry = defaultdict(list)  # Private instance registry


class IndicatorRegistrar:
    """Climate Indicator registering object."""

    def __new__(cls):
        """Add subclass to registry."""
        name = cls.__name__
        if name in registry:
            warnings.warn(f"Class {name} already exists and will be overwritten.")
        registry[name] = cls
        return super().__new__(cls)

    def __init__(self):
        _indicators_registry[self.__class__].append(weakref.ref(self))

    @classmethod
    def get_instance(cls):
        """Return first found instance.

        Raises `ValueError` if no instance exists.
        """
        for inst_ref in _indicators_registry[cls]:
            inst = inst_ref()
            if inst is not None:
                return inst
        raise ValueError(
            f"There is no existing instance of {cls.__name__}. Either none were created or they were all garbage-collected."
        )


class Indicator(IndicatorRegistrar):
    r"""Climate indicator base class.

    Climate indicator object that, when called, computes an indicator and assigns its output a number of
    CF-compliant attributes. Some of these attributes can be *templated*, allowing metadata to reflect
    the value of call arguments.

    Instantiating a new indicator returns an instance but also creates and registers a custom subclass.

    Parameters in `Indicator._cf_names` will be added to the output variable(s). When creating new `Indicators` subclasses,
    if the compute function returns multiple variables, attributes may be given as lists of strings or strings.
    In the latter case, the same value is used on all variables.

    Parameters
    ----------
    identifier: str
      Unique ID for class registry, should be a valid slug.
    realm : {'atmos', 'seaIce', 'land', 'ocean'}
      General domain of validity of the indicator. Indicators created outside xclim.indicators must set this attribute.
    compute: func
      The function computing the indicators. It should return one or more DataArray.
    var_name: str or Sequence[str]
      Output variable(s) name(s). May use tags {<tag>}. If the indicator outputs multiple variables,
      var_name *must* be a list of the same length.
    standard_name: str or Sequence[str]
      Variable name (CF).
    long_name: str or Sequence[str]
      Descriptive variable name. Parsed from `compute` docstring if not given.
    units: str or Sequence[str]
      Representative units of the physical quantity (CF).
    cell_methods: str or Sequence[str]
      List of blank-separated words of the form "name: method" (CF).
    description: str or Sequence[str]
      Sentence meant to clarify the qualifiers of the fundamental quantities, such as which
      surface a quantity is defined on or what the flux sign conventions are.
    comment: str or Sequence[str]
      Miscellaneous information about the data or methods used to produce it.
    title: str
      A succinct description of what is in the computed outputs. Parsed from `compute` docstring if None.
    abstract: str
      A long description of what is in the computed outputs. Parsed from `compute` docstring if None.
    keywords: str
      Comma separated list of keywords. Parsed from `compute` docstring if None.
    references: str
      Published or web-based references that describe the data or methods used to produce it. Parsed from
      `compute` docstring if None.
    notes: str
      Notes regarding computing function, for example the mathematical formulation. Parsed from `compute`
      docstring if None.
    missing: {any, wmo, pct, at_least_n, skip, from_context}
      The name of the missing value method. See `xclim.core.checks.MissingBase` to create new custom methods. If
      None, this will be determined by the global configuration (see `xclim.set_options`). Defaults to "from_context".
    freq: {"D", "H", None}
      The expected frequency of the input data. Use None if irrelevant.
    missing_options : dict, None
      Arguments to pass to the `missing` function. If None, this will be determined by the global configuration.
    context: str
      The `pint` unit context, for example use 'hydro' to allow conversion from kg m-2 s-1 to mm/day.

    Notes
    -----
    All subclasses created are available in the `registry` attribute and can be used to defined custom subclasses.

    """

    # Number of input DataArray variables. Should be updated by subclasses if needed.
    _nvar = 1

    # Allowed metadata attributes on the output variables
    _cf_names = [
        "var_name",
        "standard_name",
        "long_name",
        "units",
        "cell_methods",
        "description",
        "comment",
    ]

    # metadata fields that are formatted as free text.
    _text_fields = ["long_name", "description", "comment"]

    _funcs = ["compute", "cfcheck", "datacheck"]

    # Will become the class's name
    identifier = None

    missing = "from_context"
    missing_options = None
    context = "none"
    freq = None

    # Variable metadata (_cf_names, those that can be lists or strings)
    # A developper should access those through cf_attrs on instances
    var_name = None
    standard_name = ""
    long_name = ""
    units = ""
    cell_methods = ""
    description = ""
    comment = ""

    # Global metadata (must be strings, not attributed to the output)
    realm = None
    title = ""
    abstract = ""
    keywords = ""
    references = ""
    notes = ""
    parameters = None

    def __new__(cls, **kwds):
        """Create subclass from arguments."""
        identifier = kwds.get("identifier", cls.identifier)
        if identifier is None:
            raise AttributeError("`identifier` has not been set.")

        kwds["var_name"] = kwds.get("var_name", cls.var_name) or identifier

        # Parse docstring of the compute function, its signature and its parameters
        kwds = cls._parse_docstring(kwds)

        # Parse kwds to organize cf_attrs
        # Must be done after parsing var_name
        # And before converting callables to staticmethods
        kwds["cf_attrs"] = cls._parse_cf_attrs(kwds)

        # Convert function objects to static methods.
        for key in cls._funcs + cls._cf_names:
            if key in kwds and callable(kwds[key]):
                kwds[key] = staticmethod(kwds[key])

        # Infer realm for built-in xclim instances
        if cls.__module__.startswith(__package__.split(".")[0]):
            xclim_realm = cls.__module__.split(".")[2]
        else:
            xclim_realm = None
        # Priority given to passed realm -> parent's realm -> location of the class declaration (official inds only)
        kwds.setdefault("realm", cls.realm or xclim_realm)
        if kwds["realm"] not in ["atmos", "seaIce", "land", "ocean"]:
            raise AttributeError(
                "Indicator's realm must be given as one of 'atmos', 'seaIce', 'land' or 'ocean'"
            )

        # Create new class object
        new = type(identifier.upper(), (cls,), kwds)

        # Set the module to the base class' module. Otherwise all indicators will have module `xclim.core.indicator`.
        new.__module__ = cls.__module__

        #  Add the created class to the registry
        # This will create an instance from the new class and call __init__.
        return super().__new__(new)

    @classmethod
    def _parse_docstring(cls, kwds):
        """Parse `compute` docstring to extract missing attributes and parameters' doc."""
        # Priority: explicit arguments > super class attributes > `compute` docstring info
        func = kwds.get("compute", None) or cls.compute
        parsed = parse_doc(func.__doc__)

        for name, value in parsed.copy().items():
            if not getattr(cls, name):
                # Set if neither the class attr is set nor the kwds attr
                kwds.setdefault(name, value)
        # The `compute` signature
        kwds["_sig"] = signature(func)
        # The input parameters' name
        kwds["_parameters"] = tuple(kwds["_sig"].parameters.keys())
        # Fill default values and annotation in parameter doc
        # params is a multilayer dict, we want to use a brand new one so deepcopy
        params = deepcopy(kwds.get("parameters", cls.parameters or {}))
        for name, param in kwds["_sig"].parameters.items():
            param_doc = params.setdefault(name, {"type": "", "description": ""})
            param_doc["default"] = param.default
            param_doc["annotation"] = param.annotation
        for name in list(params.keys()):
            if name not in kwds["_parameters"]:
                params.pop(name)
        kwds["parameters"] = params
        return kwds

    @classmethod
    def _parse_cf_attrs(cls, kwds):
        """CF-compliant metadata attributes for all output variables."""
        # Get number of outputs
        n_outs = (
            len(kwds["var_name"]) if isinstance(kwds["var_name"], (list, tuple)) else 1
        )

        # Populate cf_attrs from attribute set during class creation and __new__
        cf_attrs = [{} for i in range(n_outs)]
        for name in cls._cf_names:
            values = kwds.get(name, getattr(cls, name))
            if not isinstance(values, (list, tuple)):
                values = [values] * n_outs
            elif len(values) != n_outs:
                raise ValueError(
                    f"Attribute {name} has {len(values)} elements but should have {n_outs} according to passed var_name."
                )
            for attrs, value in zip(cf_attrs, values):
                if value:
                    attrs[name] = value
        return cf_attrs

    def __init__(self, **kwds):
        """Run checks and organizes the metadata."""
        # keywords of kwds that are class attributes have already been set in __new__
        self.check_identifier(self.identifier)
        if self.missing == "from_context" and self.missing_options is not None:
            raise ValueError(
                "Cannot set `missing_options` with `missing` method being from context."
            )

        # Validate hard-coded missing options
        kls = MISSING_METHODS[self.missing]
        self._missing = kls.execute
        if self.missing_options:
            kls.validate(**self.missing_options)

        # Validation is done : register the instance.
        super().__init__()

        # The `compute` signature
        self._sig = signature(self.compute)

        # The input parameters' name
        self._parameters = tuple(self._sig.parameters.keys())

        # Copy the docstring and signature
        self.__call__ = wraps(self.compute)(self.__call__)

    def __call__(self, *args, **kwds):
        """Call function of Indicator class."""
        # For convenience
        n_outs = len(self.cf_attrs)

        # Bind call arguments to `compute` arguments and set defaults.
        ba = self._sig.bind(*args, **kwds)
        ba.apply_defaults()

        # Assume the first arguments are always the DataArrays.
        das = OrderedDict()
        for i in range(self._nvar):
            das[self._parameters[i]] = ba.arguments.pop(self._parameters[i])

        # Metadata attributes from templates
        var_id = None
        var_attrs = []
        for attrs in self.cf_attrs:
            if n_outs > 1:
                var_id = f"{self.identifier}.{attrs['var_name']}"
            var_attrs.append(
                self.update_attrs(ba, das, attrs, names=self._cf_names, var_id=var_id)
            )

        # Pre-computation validation checks on DataArray arguments
        self.bind_call(self.datacheck, **das)
        self.bind_call(self.cfcheck, **das)

        # Compute the indicator values, ignoring NaNs and missing values.
        outs = self.compute(**das, **ba.kwargs)
        if isinstance(outs, DataArray):
            outs = [outs]
        if len(outs) != n_outs:
            raise ValueError(
                f"Indicator {self.identifier} was wrongly defined. Expected {n_outs} outputs, got {len(outs)}."
            )

        # Convert to output units
        outs = [
            convert_units_to(out, attrs.get("units", ""), self.context)
            for out, attrs in zip(outs, var_attrs)
        ]

        # Update variable attributes
        for out, attrs in zip(outs, var_attrs):
            var_name = attrs.pop("var_name")
            out.attrs.update(attrs)
            out.name = var_name

        # Mask results that do not meet criteria defined by the `missing` method.
        # This means all variables must have the same dimensions...
        mask = self.mask(*das.values(), **ba.arguments)
        outs = [out.where(~mask) for out in outs]

        # Return a single DataArray in case of single output, otherwise a tuple
        if n_outs == 1:
            return outs[0]
        return tuple(outs)

    def bind_call(self, func, **das):
        """Call function using `__call__` `DataArray` arguments.

        This will try to bind keyword arguments to `func` arguments. If this fails, `func` is called with positional
        arguments only.

        Notes
        -----
        This method is used to support two main use cases.

        In use case #1, we have two compute functions with arguments in a different order:
            `func1(tasmin, tasmax)` and `func2(tasmax, tasmin)`

        In use case #2, we have two compute functions with arguments that have different names:
            `generic_func(da)` and `custom_func(tas)`

        For each case, we want to define a single `cfcheck` and `datacheck` methods that will work with both compute
        functions.

        Passing a dictionary of arguments will solve #1, but not #2.
        """
        # First try to bind arguments to function.
        try:
            ba = signature(func).bind(**das)
        except TypeError:
            # If this fails, simply call the function using positional arguments
            return func(*das.values())
        else:
            # Call the func using bound arguments
            return func(*ba.args, **ba.kwargs)

    @classmethod
    def update_attrs(cls, ba, das, attrs, var_id=None, names=None):
        """Format attributes with the run-time values of `compute` call parameters.

        Cell methods and xclim_history attributes are updated, adding to existing values. The language of the string is
        taken from the `OPTIONS` configuration dictionary.

        Parameters
        ----------
        das: tuple
          Input arrays.
        ba: bound argument object
          Keyword arguments of the `compute` call.
        attrs : Mapping[str, str]
          The attributes to format and update.
        var_id : str
          The identifier to use when requesting the attributes translations.
          Defaults to the class name (for the translations) or the `identifier` field of the class (for the xclim_history attribute).
          If given, the identifier will be converted to uppercase to get the translation attributes.
          This is meant for multi-outputs indicators.
        names : Sequence[str]
          List of attribute names for which to get a translation.

        Returns
        -------
        dict
          Attributes with {} expressions replaced by call argument values. With updated `cell_methods` and `xclim_history`.
          `cell_methods` is not added is `names` is given and those not contain `cell_methods`.
        """
        args = ba.arguments

        out = cls.format(attrs, args)
        for locale in OPTIONS["metadata_locales"]:
            out.update(
                cls.format(
                    get_local_attrs(
                        (var_id or cls.__name__).upper(),
                        locale,
                        names=names or list(attrs.keys()),
                        append_locale_name=True,
                    ),
                    args=args,
                    formatter=get_local_formatter(locale),
                )
            )

        # Update the signature with the values of the actual call.
        cp = OrderedDict()
        for (k, v) in ba.signature.parameters.items():
            if v.default is not None and isinstance(v.default, (float, int, str)):
                cp[k] = v.replace(default=ba.arguments[k])
            else:
                cp[k] = v

        # Get history and cell method attributes from source data
        attrs = defaultdict(str)
        if names is None or "cell_methods" in names:
            attrs["cell_methods"] = merge_attributes(
                "cell_methods", new_line=" ", missing_str=None, **das
            )
            if "cell_methods" in out:
                attrs["cell_methods"] += " " + out.pop("cell_methods")

        attrs["xclim_history"] = update_history(
            f"{var_id or cls.identifier}{ba.signature.replace(parameters=cp.values())}",
            new_name=out.get("var_name"),
            **das,
        )

        attrs.update(out)
        return attrs

    @staticmethod
    def check_identifier(identifier):
        """Verify that the identifier is a proper slug."""
        if not re.match(r"^[-\w]+$", identifier):
            warnings.warn(
                "The identifier contains non-alphanumeric characters. It could make life "
                "difficult for downstream software reusing this class.",
                UserWarning,
            )

    def translate_attrs(
        self, locale: Union[str, Sequence[str]], fill_missing: bool = True
    ):
        """Return a dictionary of unformated translated translatable attributes.

        Translatable attributes are defined in xclim.core.locales.TRANSLATABLE_ATTRS

        Parameters
        ----------
        locale : Union[str, Sequence[str]]
            The POSIX name of the locale or a tuple of a locale name and a path to a
            json file defining the translations. See `xclim.locale` for details.
        fill_missing : bool
            If True (default fill the missing attributes by their english values.
        """

        def _translate(var_id, var_attrs, names):
            attrs = get_local_attrs(
                var_id,
                locale,
                names=names,
                append_locale_name=False,
            )
            if fill_missing:
                for name in names:
                    if name not in attrs and var_attrs.get(name):
                        attrs[name] = var_attrs.get(name)
            return attrs

        # Translate global attrs
        attrid = self.identifier.upper()
        attrs = _translate(
            attrid,
            self.__dict__,
            # Translate only translatable attrs that are not variable attrs
            set(TRANSLATABLE_ATTRS).difference(set(self._cf_names)),
        )
        # Translate variable attrs
        attrs["outputs"] = []
        for var_attrs in self.cf_attrs:  # Translate for each variable
            if len(self.cf_attrs) > 1:
                attrid = f"{self.identifier.upper()}.{var_attrs['var_name']}"
            attrs["outputs"].append(_translate(attrid, var_attrs, TRANSLATABLE_ATTRS))
        return attrs

    def json(self, args=None):
        """Return a dictionary representation of the class.

        Notes
        -----
        This is meant to be used by a third-party library wanting to wrap this class into another interface.

        """
        names = ["identifier", "title", "abstract", "keywords"]
        out = {key: getattr(self, key) for key in names}
        out = self.format(out, args)
        out["outputs"] = [self.format(attrs, args) for attrs in self.cf_attrs]

        out["notes"] = self.notes
        # We need to deepcopy, otherwise empty defaults get overwritten!
        out["parameters"] = deepcopy(self.parameters)
        for param in out["parameters"].values():
            if param["default"] is _empty:
                param["default"] = "none"
        return out

    @classmethod
    def format(
        cls,
        attrs: dict,
        args: dict = None,
        formatter: AttrFormatter = default_formatter,
    ):
        """Format attributes including {} tags with arguments.

        Parameters
        ----------
        attrs: dict
          Attributes containing tags to replace with arguments' values.
        args : dict
          Function call arguments.
        formatter : AttrFormatter
        """
        if args is None:
            return attrs

        out = {}
        for key, val in attrs.items():
            mba = {"indexer": "annual"}
            # Add formatting {} around values to be able to replace them with _attrs_mapping using format.
            for k, v in args.items():
                if isinstance(v, dict):
                    if v:
                        dk, dv = v.copy().popitem()
                        if dk == "month":
                            dv = "m{}".format(dv)
                        mba[k] = dv
                elif isinstance(v, units.Quantity):
                    mba[k] = "{:g~P}".format(v)
                elif isinstance(v, (int, float)):
                    mba[k] = "{:g}".format(v)
                else:
                    mba[k] = v

            if callable(val):
                val = val(**mba)

            out[key] = formatter.format(val, **mba)

            if key in cls._text_fields:
                out[key] = out[key].strip().capitalize()

        return out

    def mask(self, *args, **kwds):
        """Return whether mask for output values, based on the output of the `missing` method."""
        from functools import reduce

        indexer = kwds.get("indexer") or {}
        freq = kwds.get("freq") if "freq" in kwds else default_freq(**indexer)

        options = self.missing_options or OPTIONS[MISSING_OPTIONS].get(self.missing, {})

        # We flag periods according to the missing method.
        miss = (self._missing(da, freq, self.freq, options, indexer) for da in args)

        return reduce(np.logical_or, miss)

    # The following static methods are meant to be replaced to define custom indicators.
    @staticmethod
    def compute(*args, **kwds):
        """Compute the indicator.

        This would typically be a function from `xclim.indices`.
        """
        raise NotImplementedError

    @staticmethod
    def cfcheck(**das):
        """Compare metadata attributes to CF-Convention standards.

        When subclassing this method, use functions decorated using `xclim.core.options.cfcheck`.
        """
        return True

    @staticmethod
    def datacheck(**das):
        """Verify that input data is valid.

        When subclassing this method, use functions decorated using `xclim.core.options.datacheck`.

        For example, checks could include:
         - assert temporal frequency is daily
         - assert no precipitation is negative
         - assert no temperature has the same value 5 days in a row
        """
        return True


class Indicator2D(Indicator):
    """Indicator using two dimensions."""

    _nvar = 2


class Daily(Indicator):
    """Indicator defined for inputs at daily frequency."""

    freq = "D"

    @staticmethod
    def datacheck(**das):  # noqa
        for key, da in das.items():
            datachecks.check_daily(da)


class Daily2D(Daily):
    """Indicator using two dimensions at daily frequency."""

    _nvar = 2


class Hourly(Indicator):
    """Indicator defined for inputs at strict hourly frequency, meaning 3-hourly inputs would raise an error."""

    freq = "H"

    @staticmethod
    def datacheck(**das):  # noqa
        for key, da in das.items():
            datachecks.check_freq(da, "H")

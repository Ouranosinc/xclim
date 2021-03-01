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
from inspect import Parameter, _empty, signature
from typing import Any, Callable, Dict, List, Mapping, Sequence, Union

import numpy as np
from boltons.funcutils import copy_function, wraps
from xarray import DataArray, Dataset

from xclim.indices.generic import default_freq

from . import datachecks
from .formatting import (
    AttrFormatter,
    default_formatter,
    generate_indicator_docstring,
    merge_attributes,
    parse_doc,
    update_history,
)
from .locales import TRANSLATABLE_ATTRS, get_local_attrs, get_local_formatter
from .options import MISSING_METHODS, MISSING_OPTIONS, OPTIONS
from .units import convert_units_to, units
from .utils import MissingVariableError, infer_kind_from_parameter

# Indicators registry
registry = {}  # Main class registry
_indicators_registry = defaultdict(list)  # Private instance registry


def _parse_indice(indice: Callable, **new_kwargs):
    """Parse an indice function and return all elements needed for constructing an indicator.

    Parameters
    ----------
    indice : Callable
      A indice function, written according to xclim's guidelines.
    new_kwargs :
      Mapping from name to dicts containing the necessary info for injecting new keyword-only
      arguments into the indice_wrapper function. The meta dict can include (all optional):
      `default`, `description`, `annotation`.

    Returns
    -------
    indice_wrapper : callable
      A function with a new signature including the injected args in new_kwargs.
    docmeta : Mapping[str, str]
      A dictionary of the metadata attributes parsed in the docstring.
    params : Mapping[str, Mapping[str, Any]]
      A dictionary of metadata for each input parameter of the indice. The metadata dictionaries
      include the following entries: "default", "description", "kind" and, optionally, "choices" and "units".
      "kind" is one of the constants in :py:class:`xclim.core.utils.InputKind`.
    """
    # Base signature
    sig = signature(indice)

    # Update
    def _upd_param(param):
        # Required DataArray arguments receive their own name as new default
        #         + the Union[str, DataArray] annotation
        # Parameters with no default receive None
        if param.kind in [param.VAR_KEYWORD, param.VAR_POSITIONAL]:
            return param

        if param.annotation is DataArray:
            annot = Union[str, DataArray]
        else:
            annot = param.annotation

        if param.default is _empty:
            if param.annotation is DataArray:
                default = param.name
            else:
                default = None
        else:
            default = param.default

        return Parameter(
            param.name,
            # We keep the kind, except we replace POSITIONAL_ONLY by POSITONAL_OR_KEYWORD
            max(param.kind, 1),
            default=default,
            annotation=annot,
        )

    # Parse all parameters, replacing annotations and default where needed and possible.
    new_params = list(map(_upd_param, sig.parameters.values()))

    # Injection
    for name, meta in new_kwargs.items():
        # ds argunent
        param = Parameter(
            name,
            Parameter.KEYWORD_ONLY,
            default=meta.get("default"),
            annotation=meta.get("annotation"),
        )

        if new_params[-1].kind == Parameter.VAR_KEYWORD:
            new_params.insert(-1, param)
        else:
            new_params.append(param)

    # Create new compute function to be wrapped in __call__
    indice_wrapper = copy_function(indice)
    indice_wrapper.__signature__ = new_sig = sig.replace(parameters=new_params)
    indice_wrapper.__doc__ = indice.__doc__

    # Docstring parsing
    parsed = parse_doc(indice.__doc__)

    # Extract params and pop those not in the signature.
    params = parsed.pop("parameters", {})
    for dropped in set(params.keys()) - set(new_sig.parameters.keys()):
        params.pop(dropped)

    if hasattr(indice, "in_units"):
        # Try to put units
        for var, ustr in indice.in_units.items():
            if var in params:
                params[var]["units"] = ustr

    # Fill default values and annotation in parameter doc
    for name, param in new_sig.parameters.items():
        if name in new_kwargs and "description" in new_kwargs[name]:
            params[name] = {"description": new_kwargs[name]["description"]}
        param_doc = params.setdefault(name, {"description": ""})
        param_doc["default"] = param.default
        param_doc["kind"] = infer_kind_from_parameter(param, "units" in param_doc)

    return indice_wrapper, parsed, params


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

    Compared to their base `compute` function, indicators add the possibility of using dataset as input,
    with the injected argument `ds` in the call signature. All arguments that were indicated by the compute function
    to be DataArrays through annotations will be promoted to also accept strings that correspond to variable names
    in the `ds` dataset.

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
      The name of the missing value method. See `xclim.core.missing.MissingBase` to create new custom methods. If
      None, this will be determined by the global configuration (see `xclim.set_options`). Defaults to "from_context".
    freq: {"D", "H", None}
      The expected frequency of the input data. Use None if irrelevant.
    missing_options : dict, None
      Arguments to pass to the `missing` function. If None, this will be determined by the global configuration.
    context: str
      The `pint` unit context, for example use 'hydro' to allow conversion from kg m-2 s-1 to mm/day.

    Notes
    -----
    All subclasses created are available in the `registry` attribute and can be used to define custom subclasses
    or parse all available instances.

    """

    #: Number of input DataArray variables. Should be updated by subclasses if needed.
    #: This number sets which inputs are passed to the tests.
    nvar = 1

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

    parameters: Mapping[str, Any]
    """A dictionary mapping metadata about the input parameters to the indicator.

       Contains : "default", "description", "kind" and, sometimes, "units" and "choices".
       "kind" refers to the constants of :py:class:`xclim.core.utils.InputKind`.
    """

    cf_attrs: Sequence[Mapping[str, Any]]
    """A list of metadata information for each output of the indicator.

       It minimally contains a "var_name" entry, and may contain : "standard_name", "long_name",
       "units", "cell_methods", "description" and "comment".
    """

    def __new__(cls, **kwds):
        """Create subclass from arguments."""
        identifier = kwds.get("identifier", cls.identifier)
        if identifier is None:
            raise AttributeError("`identifier` has not been set.")

        kwds["var_name"] = kwds.get("var_name", cls.var_name) or identifier

        # Parse and update compute's signature.
        kwds["compute"] = kwds.get("compute", None) or cls.compute
        # Updated to allow string variable names and the ds arg.
        # Parse docstring of the compute function, its signature and its parameters
        kwds["_indcompute"], docmeta, params = _parse_indice(
            kwds["compute"],
            ds={
                "annotation": Dataset,
                "description": "A dataset with the variables given by name.",
            },
        )

        # The update signature
        kwds["_sig"] = kwds["_indcompute"].__signature__
        # The input parameters' name
        kwds["_parameters"] = tuple(kwds["_sig"].parameters.keys())

        # All fields parsed by parse_doc except "parameters"
        # i.e. : title, abstract, notes, references, long_name
        for name, value in docmeta.items():
            if not getattr(cls, name):
                # Set if neither the class attr is set nor the kwds attr
                kwds.setdefault(name, value)

        # The input parameters' metadata
        # We dump whatever the base class had and take what was parsed from the current compute function.
        kwds["parameters"] = params

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

        kwds["_indcompute"].__doc__ = kwds["__doc__"] = generate_indicator_docstring(
            kwds
        )

        # Create new class object
        new = type(identifier.upper(), (cls,), kwds)

        # Set the module to the base class' module. Otherwise all indicators will have module `xclim.core.indicator`.
        new.__module__ = cls.__module__

        #  Add the created class to the registry
        # This will create an instance from the new class and call __init__.
        return super().__new__(new)

    @classmethod
    def _parse_cf_attrs(
        cls, kwds: Dict[str, Any]
    ) -> Union[List[Dict[str, str]], List[Dict[str, Union[str, Callable]]]]:
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
        self._check_identifier(self.identifier)
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

        # Update call signature
        self.__call__ = wraps(self._indcompute)(self.__call__)

    def __call__(self, *args, **kwds):
        """Call function of Indicator class."""
        # For convenience
        n_outs = len(self.cf_attrs)

        # Bind call arguments to `compute` arguments and set defaults.
        ba = self._sig.bind(*args, **kwds)
        ba.apply_defaults()

        # Assign inputs passed as strings from ds.
        self._assign_named_args(ba)

        # Assume the first arguments are always the DataArrays.
        # Only the first nvar inputs are checked (data + cf checks)
        das = OrderedDict()
        for name in self._parameters[: self.nvar]:
            das[name] = ba.arguments.pop(name)

        # Metadata attributes from templates
        var_id = None
        var_attrs = []
        for attrs in self.cf_attrs:
            if n_outs > 1:
                var_id = f"{self.identifier}.{attrs['var_name']}"
            var_attrs.append(
                self._update_attrs(ba, das, attrs, names=self._cf_names, var_id=var_id)
            )

        # Pre-computation validation checks on DataArray arguments
        self._bind_call(self.datacheck, **das)
        self._bind_call(self.cfcheck, **das)

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
        mask = self._mask(*das.values(), **ba.arguments)
        outs = [out.where(~mask) for out in outs]

        # Return a single DataArray in case of single output, otherwise a tuple
        if n_outs == 1:
            return outs[0]
        return tuple(outs)

    def _assign_named_args(self, ba):
        """Assign inputs passed as strings from ds."""
        ds = ba.arguments.pop("ds")
        for name, param in self._sig.parameters.items():
            if param.annotation is Union[str, DataArray] and isinstance(
                ba.arguments[name], str
            ):
                if ds is not None:
                    try:
                        ba.arguments[name] = ds[ba.arguments[name]]
                    except KeyError:
                        raise MissingVariableError(
                            f"For input '{name}', variable '{ba.arguments[name]}' was not found in the input dataset."
                        )
                else:
                    raise ValueError(
                        f"Passing variable names as string requires giving the `ds` dataset (got {name}='{ba.arguments[name]}')"
                    )

    def _bind_call(self, func, **das):
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
    def _update_attrs(cls, ba, das, attrs, var_id=None, names=None):
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

        out = cls._format(attrs, args)
        for locale in OPTIONS["metadata_locales"]:
            out.update(
                cls._format(
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

        # Generate a signature string for the history attribute
        # We remove annotations, replace default float/int/str by values
        # and replace others by type
        callstr = []
        for (k, v) in das.items():
            callstr.append(f"{k}=<array>")
        for (k, v) in ba.arguments.items():
            if isinstance(v, (float, int, str)):
                callstr.append(f"{k}={v!r}")  # repr so strings have ' '
            else:
                callstr.append(
                    f"{k}={type(v)}"
                )  # don't take chance of having unprintable values

        # Get history and cell method attributes from source data
        attrs = defaultdict(str)
        if names is None or "cell_methods" in names:
            attrs["cell_methods"] = merge_attributes(
                "cell_methods", new_line=" ", missing_str=None, **das
            )
            if "cell_methods" in out:
                attrs["cell_methods"] += " " + out.pop("cell_methods")

        attrs["xclim_history"] = update_history(
            f"{var_id or cls.identifier}({', '.join(callstr)})",
            new_name=out.get("var_name"),
            **das,
        )

        attrs.update(out)
        return attrs

    @staticmethod
    def _check_identifier(identifier: str) -> None:
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

        Translatable attributes are defined in :py:const:`xclim.core.locales.TRANSLATABLE_ATTRS`.

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
        attrid = str(self.identifier).upper()
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
                attrid = f"{str(self.identifier).upper()}.{var_attrs['var_name']}"
            attrs["outputs"].append(_translate(attrid, var_attrs, TRANSLATABLE_ATTRS))
        return attrs

    def json(self, args=None):
        """Return a serializable dictionary representation of the class.

        Parameters
        ----------
        args : mapping, optional
            Arguments as passed to the call method of the indicator.
            If not given, the default arguments will be used when formatting the attributes.

        Notes
        -----
        This is meant to be used by a third-party library wanting to wrap this class into another interface.

        """
        names = ["identifier", "title", "abstract", "keywords"]
        out = {key: getattr(self, key) for key in names}
        out = self._format(out, args)

        # Format attributes
        out["outputs"] = [self._format(attrs, args) for attrs in self.cf_attrs]
        out["notes"] = self.notes

        # We need to deepcopy, otherwise empty defaults get overwritten!
        # All those tweaks are to ensure proper serialization of the returned dictionary.
        out["parameters"] = deepcopy(self.parameters)
        for param in out["parameters"].values():
            if param["default"] is _empty:
                param.pop("default")
            param["kind"] = param["kind"].value  # Get the int.
            if "choices" in param:  # A set is stored, convert to list
                param["choices"] = list(param["choices"])
        return out

    @classmethod
    def _format(
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
        args : dict, optional
          Function call arguments. If not given, the default arguments will be used when formatting the attributes.
        formatter : AttrFormatter
        """
        # Use defaults
        if args is None:
            args = {k: v["default"] for k, v in cls.parameters.items()}

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

    def _default_freq(self, **indexer):
        """Return default frequency."""
        if self.freq in ["D", "H"]:
            return default_freq(**indexer)
        return None

    def _mask(self, *args, **kwds):
        """Return whether mask for output values, based on the output of the `missing` method."""
        from functools import reduce

        indexer = kwds.get("indexer") or {}
        freq = kwds.get("freq") if "freq" in kwds else self._default_freq(**indexer)

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

    nvar = 2


class Daily(Indicator):
    """Indicator defined for inputs at daily frequency."""

    freq = "D"

    @staticmethod
    def datacheck(**das):  # noqa
        for key, da in das.items():
            datachecks.check_daily(da)


class Daily2D(Daily):
    """Indicator using two dimensions at daily frequency."""

    nvar = 2


class Hourly(Indicator):
    """Indicator defined for inputs at strict hourly frequency, meaning 3-hourly inputs would raise an error."""

    freq = "H"

    @staticmethod
    def datacheck(**das):  # noqa
        for key, da in das.items():
            datachecks.check_freq(da, "H")

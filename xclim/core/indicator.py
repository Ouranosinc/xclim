# -*- coding: utf-8 -*-
# noqa: D205,D400
"""
Indicators utilities
====================

The `Indicator` class wraps indices computations with pre- and post-processing functionality. Prior to computations,
the class runs data and metadata health checks. After computations, the class masks values that should be considered
missing and adds metadata attributes to the  object.

There are many ways to construct indicators. A good place to start is `this notebook <notebooks/extendxclim.ipynb#Defining-new-indicators>`_.

Dictionary and YAML parser
--------------------------

To construct indicators dynamically, xclim can also use dictionaries and parse them from YAML files.
This is especially useful for generating whole indicator "submodules" from files.
This functionality is inspired by the work of [clix-meta](https://github.com/clix-meta/clix-meta/).

YAML file structure
~~~~~~~~~~~~~~~~~~~

Indicator-defining yaml files are structured in the following way.
Most entries of the `indicators` section are mirroring attributes of
the :py:class:`Indicator`, please refer to its documentation for more
details on each.

.. code-block:: yaml

    module: <module name>  # Defaults to the file name
    realm: <realm>  # If given here, applies to all indicators that do no give it.
    keywords: <keywords> # Merged with indicator-specific keywords (joined with a space)
    references: <references> # Merged with indicator-specific referencess (joined with a new line)
    base: <base indicator class>  # Defaults to "Daily" and applies to all indicators that do not give it.
    doc: <module docstring>  # Defaults to a minimal header, only valid if the module doesn't already exists.
    indicators:
      <identifier>:
        # From which Indicator to inherit
        base: <base indicator class>  # Defaults to module-wide base class
                                      # If the name startswith a '.', the base class is taken from the current module (thus an indicator declared _above_)
                                      # Available classes are listed in `xclim.core.indicator.registry` and `xclim.core.indicator.base_registry`.

        # General metadata, usually parsed from the compute's docstring when possible.
        realm: <realm>  # defaults to module-wide realm. One of "atmos", "land", "seaIce", "ocean".
        title: <title>
        abstract: <abstract>
        keywords: <keywords>  # Space-separated, merged to module-wide keywords.
        references: <references>  # newline-seperated, merged to module-wide references.
        notes: <notes>

        # Other options
        missing: <missing method name>
        missing_options:
            # missing options mapping
        allowed_periods: [<list>, <of>, <allowed>, <periods>]

        # Compute function
        compute: <function name>  # Refering to a function in the passed indices module, xclim.indices.generic or xclim.indices
        input:  # When "compute" is a generic function this is a mapping from argument
                # name to what CMIP6/xclim variable is expected. This will allow for
                # declaring expected input units and have a CF metadata check on the inputs.
          <var1> : <variable official name 1>
          ...
        parameters:
         <param name>: <param data>  # Simplest case, to inject parameters in the compute function.
         <param name>:  # To change parameters metadata or to declare units when "compute" is a generic function.
            units: <param units>  # Only valid if "compute" points to a generic function
            default : <param default>
            description: <param description>
        ...
      ...  # and so on.

All fields are optional. Other fields found in the yaml file will trigger errors in xclim.
In the following, the section under `<identifier>` is refered to as `data`. When creating indicators from
a dictionary, with :py:meth:`Indicator.from_dict`, the input dict must follow the same structure of `data`.

The resulting yaml file can be validated using the provided schema (in xclim/data/schema.yml) and the [yamale](https://github.com/23andMe/Yamale) tool.
See the "Extending xclim" notebook for more info.

Inputs
~~~~~~
As xclim has strict definitions of possible input variables (see :py:data:`xclim.core.utils.variables`),
the mapping of `data.input` simply links an argument name from the function given in "compute"
to one of those official variables.

"""
import re
import warnings
import weakref
from collections import OrderedDict, defaultdict
from copy import deepcopy
from inspect import Parameter, _empty, signature  # noqa
from os import PathLike
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Union

import numpy as np
from boltons.funcutils import copy_function, wraps
from xarray import DataArray, Dataset
from yaml import safe_load

from .. import indices
from . import datachecks
from .calendar import parse_offset
from .cfchecks import cfcheck_from_name
from .formatting import (
    AttrFormatter,
    default_formatter,
    gen_call_string,
    generate_indicator_docstring,
    merge_attributes,
    parse_doc,
    update_history,
)
from .locales import (
    TRANSLATABLE_ATTRS,
    get_local_attrs,
    get_local_formatter,
    load_locale,
    read_locale_file,
)
from .options import METADATA_LOCALES, MISSING_METHODS, MISSING_OPTIONS, OPTIONS
from .units import convert_units_to, declare_units, units
from .utils import (
    VARIABLES,
    InputKind,
    MissingVariableError,
    infer_kind_from_parameter,
    load_module,
    raise_warn_or_log,
    wrapped_partial,
)

# Indicators registry
registry = dict()  # Main class registry
base_registry = dict()
_indicators_registry = defaultdict(list)  # Private instance registry


class IndicatorRegistrar:
    """Climate Indicator registering object."""

    def __new__(cls):
        """Add subclass to registry."""
        name = cls.__name__.upper()
        module = cls.__module__
        # If the module is not one of xclim's default, prepend the submodule name.
        if module.startswith("xclim.indicators"):
            submodule = module.split(".")[2]
            if submodule not in ["atmos", "land", "ocean", "seaIce"]:
                name = f"{submodule}.{name}"
        else:
            name = f"{module}.{name}"
        if name in registry:
            warnings.warn(
                f"Class {name} already exists and will be overwritten.", stacklevel=1
            )
        registry[name] = cls
        cls._registry_id = name
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
            f"There is no existing instance of {cls.__name__}. "
            "Either none were created or they were all garbage-collected."
        )


class Indicator(IndicatorRegistrar):
    r"""Climate indicator base class.

    Climate indicator object that, when called, computes an indicator and assigns its output a number of
    CF-compliant attributes. Some of these attributes can be *templated*, allowing metadata to reflect
    the value of call arguments.

    Instantiating a new indicator returns an instance but also creates and registers a custom subclass.

    Parameters in `Indicator._cf_names` will be added to the output variable(s). When creating new `Indicators`
    subclasses, if the compute function returns multiple variables, attributes may be given as lists of strings or
    strings. In the latter case, the same value is used on all variables.

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
    freq: str, sequence of strings, optional
      The expected frequency of the input data. Can be a list for multiple frequencies, or None if irrelevant.
    missing_options : dict, None
      Arguments to pass to the `missing` function. If None, this will be determined by the global configuration.
    context: str
      The `pint` unit context, for example use 'hydro' to allow conversion from kg m-2 s-1 to mm/day.
    allowed_periods : Sequence[str], optional
      A list of allowed periods, i.e. base parts of the `freq` parameter. For example, indicators meant to be
      computed annually only will have `allowed_periods=["A"]`. `None` means "any period" or that the
      indicator doesn't take a `freq` argument.

    Notes
    -----
    All subclasses created are available in the `registry` attribute and can be used to define custom subclasses
    or parse all available instances.

    """
    # Officially-supported metadata attributes on the output variables
    _cf_names = [
        "var_name",
        "standard_name",
        "long_name",
        "units",
        "cell_methods",
        "description",
        "comment",
    ]

    # metadata fields that are formatted as free text (first letter capitalized)
    _text_fields = ["long_name", "description", "comment"]

    _funcs = ["compute", "cfcheck", "datacheck"]

    # Will become the class's name
    identifier = None

    missing = "from_context"
    missing_options = None
    context = "none"
    freq = None
    allowed_periods = None

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

    cf_attrs: Sequence[Mapping[str, Any]] = None
    """A list of metadata information for each output of the indicator.

       It minimally contains a "var_name" entry, and may contain : "standard_name", "long_name",
       "units", "cell_methods", "description" and "comment" on official xclim indicators. Other
       fields could also be present if the indicator was created from outside xclim.
    """

    def __new__(cls, **kwds):
        """Create subclass from arguments."""
        identifier = kwds.get("identifier", cls.identifier)
        if identifier is None:
            raise AttributeError("`identifier` has not been set.")

        # Parse and update compute's signature.
        kwds["compute"] = kwds.get("compute", None) or cls.compute

        # Updated to allow string variable names and the ds arg.
        # Parse docstring of the compute function, its signature and its parameters
        kwds["_indcompute"], docmeta, params = _parse_indice(
            kwds["compute"],
            passed=kwds.get("parameters"),
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
            if not getattr(cls, name, None):
                # Set if neither the class attr is set nor the kwds attr
                kwds.setdefault(name, value)

        # The input parameters' metadata
        # We dump whatever the base class had and take what was parsed from the current compute function.
        kwds["parameters"] = params

        # By default skip missing values handling if there is no resampling.
        # Dont only check if freq is in current parameters but also if it was injected earlier.
        if "freq" not in params and "freq" not in getattr(
            kwds["compute"], "_injected", {}
        ):
            kwds["missing"] = "skip"

        # Parse kwds to organize `cf_attrs`
        # And before converting callables to staticmethods
        kwds["cf_attrs"] = cls._parse_output_attrs(kwds, identifier)

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

        # Forcing the module is there so YAML-generated submodules are correctly seen by IndicatorRegistrar.
        if kwds.get("module") is not None:
            new.__module__ = f"xclim.indicators.{kwds['module']}"
        else:
            # If the module was not forced, set the module to the base class' module.
            # Otherwise all indicators will have module `xclim.core.indicator`.
            new.__module__ = cls.__module__

        # Generate docstring
        new._indcompute.__doc__ = new.__doc__ = generate_indicator_docstring(new)

        #  Add the created class to the registry
        # This will create an instance from the new class and call __init__.
        return super().__new__(new)

    @classmethod
    def _parse_output_attrs(
        cls, kwds: Dict[str, Any], identifier: str
    ) -> List[Dict[str, Union[str, Callable]]]:
        """CF-compliant metadata attributes for all output variables."""
        parent_cf_attrs = cls.cf_attrs
        cf_attrs = kwds.get("cf_attrs")
        if isinstance(cf_attrs, dict):
            # Single output indicator, but we store as a list anyway.
            cf_attrs = [cf_attrs]
        elif cf_attrs is None and parent_cf_attrs:
            cf_attrs = deepcopy(parent_cf_attrs)
        elif cf_attrs is None:
            # Attributes were passed the "old" way, with lists or strings directly (only _cf_names)
            # We need to get the number of outputs first, defaulting to the length of parent's cf_attrs or 1
            n_outs = len(parent_cf_attrs) if parent_cf_attrs is not None else 1
            for name in cls._cf_names:
                arg = kwds.get(name)
                if isinstance(arg, (tuple, list)):
                    n_outs = len(arg)

            # Populate new cf_attrs from parsing cf_names passed directly.
            cf_attrs = [{} for i in range(n_outs)]
            for name in cls._cf_names:
                values = kwds.pop(name, None)
                if values is None:  # None passed, skip
                    continue
                elif not isinstance(values, (tuple, list)):
                    # a single string or callable, same for all outputs
                    values = [values] * n_outs
                elif len(values) != n_outs:  # A sequence of the wrong length.
                    raise ValueError(
                        f"Attribute {name} has {len(values)} elements but should xclim expected {n_outs}."
                    )
                for attrs, value in zip(cf_attrs, values):
                    if value:  # Skip the empty ones (None or '')
                        attrs[name] = value
        # else we assume a list of dicts

        # For single output, var_name defauls to identifer.
        if len(cf_attrs) == 1 and "var_name" not in cf_attrs[0]:
            cf_attrs[0]["var_name"] = identifier

        # update from parent, if they have the same length.
        if parent_cf_attrs is not None and len(parent_cf_attrs) == len(cf_attrs):
            for old, new in zip(parent_cf_attrs, cf_attrs):
                for attr, value in old.items():
                    new.setdefault(attr, value)

        # check if we have var_names for everybody
        for i, var in enumerate(cf_attrs, start=1):
            if "var_name" not in var:
                raise ValueError(f"Output #{i} is missing a var_name! Got: {var}.")

        return cf_attrs

    @classmethod
    def from_dict(
        cls,
        data: dict,
        identifier: str,
        module: Optional[str] = None,
    ):
        """Create an indicator subclass and instance from a dictionary of parameters.

        Most parameters are passed directly as keyword arguments to the class constructor, except:

        - "compute" : A string function name translates to a
          :py:mod:`xclim.indices.generic` or :py:mod:`xclim.indices` function.
        - "input" : This mapping is used when "compute" is a generic function, to map
           from function argument names to CMIP6/xclim variable names (see :py:data:`xclim.core.utils.VARIABLES`).
        - "parameters" : This mapping can be used to either inject a parameters,
           by passing a value directly, or to update the metadata of a parameters,
           by passing a mapping with one or more keys in "default", "description" or "units".
           The later is only valid is "compute" is a generic function.

        Parameters
        ----------
        data: dict
          The exact structure of this dictionary is detailed in the submodule documentation.
        identifier : str
          The name of the subclass and internal indicator name.
        module : str
          The module name of the indicator. This is meant to be used only if the indicator
          is part of a dynamically generated submodule, to override the module of the base class.
        """
        data = data.copy()

        if "base" in data:
            if isinstance(data["base"], str):
                cls = registry.get(
                    data["base"].upper(), base_registry.get(data["base"])
                )
                if cls is None:
                    raise ValueError(
                        f"Requested base class {data['base']} is neither in the indicators registry nor in base classes registry."
                    )
            else:
                cls = data["base"]

        params = {}
        input_units = {}

        inputs = data.pop("input", None)
        if inputs is not None:
            # Override input metadata
            for varname, name in inputs.items():
                # Indicator's new will put the name of the variable as its default, we override this with the real variable name.
                params[varname] = {
                    "default": name,
                    "description": VARIABLES[name]["description"],
                }
                input_units[varname] = VARIABLES[name]["canonical_units"]

        compute = data.pop("compute", None)
        # data.compute refers to a function in xclim.indices.generic or xclim.indices (in this order of priority).
        # It can also directly be a function (like if a module was passed to build_indicator_module_from_yaml)
        if isinstance(compute, str):
            compute_func = getattr(
                indices.generic, compute, getattr(indices, compute, None)
            )
            if compute_func is None:
                raise ImportError(
                    f"Indice function {compute} not found in xclim.indices or xclim.indices.generic."
                )
            compute = compute_func

        injected_params = {}
        for name, param in data.pop("parameters", {}).items():
            if not isinstance(param, dict):
                # Injecting by passing a value directly, catch all YAML-supported types
                injected_params[name] = param
            else:
                # Changing the metadata (only "description", "default", "choices" and "units")
                params[name] = param
                if "units" in param:
                    input_units[name] = param["units"]

        if input_units:
            if hasattr(compute, "in_units"):
                raise ValueError(
                    f"Passing inputs or changing parameters' units is only valid if the compute function is not aleady wrapped by `xclim.core.units.declare_units`. Got function {compute} that has input units {compute.in_units}."
                )
            compute = declare_units(**input_units)(compute)

        if injected_params:
            # It's possible to inject params without passing a compute, in that case we use the one from the base class.
            compute = wrapped_partial(compute or cls.compute, **injected_params)

        # Dont pass things it they were not updated
        if compute is not None:
            data["compute"] = compute
        if params:  # non-empty dict
            data["parameters"] = params

        return cls(identifier=identifier, module=module, **data)

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

        # Put the variables in `das`, parse them according to the annotations
        # das : OrderedDict of variables (required + non-None optionals)
        # params : OrderedDict of parameters INCLUDING unpacked kwargs
        # all_params: OrderedDict of parameters with PACKED kwargs <- this is needed by _update_attrs and _mask because of `indexer`.
        #                  AND includes injected arguments <- this is needed by update_attrs and missing (when "freq" is injected)
        das, params, all_params = self._parse_variables_from_call(args, kwds)

        # Metadata attributes from templates
        var_id = None
        cf_attrs = []
        for attrs in self.cf_attrs:
            if n_outs > 1:
                var_id = attrs["var_name"]
            cf_attrs.append(
                self._update_attrs(
                    all_params.copy(), das, attrs, names=self._cf_names, var_id=var_id
                )
            )

        # Pre-computation validation checks on DataArray arguments
        self._bind_call(self.datacheck, **das)
        self._bind_call(self.cfcheck, **das)

        # Check if the period is allowed:
        if (
            self.allowed_periods is not None
            and "freq" in all_params
            and parse_offset(all_params["freq"])[1] not in self.allowed_periods
        ):
            raise ValueError(
                f"Resampling frequency {all_params['freq']} is not allowed for indicator "
                f"{self.identifier} (needs something equivalent to one of {self.allowed_periods})."
            )

        # Compute the indicator values, ignoring NaNs and missing values.
        outs = self.compute(**das, **params)

        if isinstance(outs, DataArray):
            outs = [outs]

        if len(outs) != n_outs:
            raise ValueError(
                f"Indicator {self.identifier} was wrongly defined. Expected {n_outs} outputs, got {len(outs)}."
            )

        # Convert to output units
        outs = [
            convert_units_to(out, attrs.get("units", ""), self.context)
            for out, attrs in zip(outs, cf_attrs)
        ]

        # Update variable attributes
        for out, attrs in zip(outs, cf_attrs):
            var_name = attrs.pop("var_name")
            out.attrs.update(attrs)
            out.name = var_name

        if self.missing != "skip":
            # Mask results that do not meet criteria defined by the `missing` method.
            # This means all outputs must have the same dimensions as the broadcasted inputs (excluding time)
            mask = self._mask(*das.values(), **all_params)
            outs = [out.where(~mask) for out in outs]

        # Return a single DataArray in case of single output, otherwise a tuple
        if n_outs == 1:
            return outs[0]
        return tuple(outs)

    def _assign_named_args(self, ba):
        """Assign inputs passed as strings from ds."""
        ds = ba.arguments.pop("ds")
        for name, param in self._sig.parameters.items():
            if (
                self.parameters[name]["kind"]
                in (
                    InputKind.VARIABLE,
                    InputKind.OPTIONAL_VARIABLE,
                )
                and isinstance(ba.arguments[name], str)
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

    def _parse_variables_from_call(self, args, kwds):
        """Extract variable and optional variables from call arguments."""
        # Bind call arguments to `compute` arguments and set defaults.
        ba = self._sig.bind(*args, **kwds)
        ba.apply_defaults()

        # Assign inputs passed as strings from ds.
        self._assign_named_args(ba)

        das = OrderedDict()
        for name, param in self.parameters.items():
            kind = param["kind"]
            # If a variable pop the arg
            if kind in (InputKind.VARIABLE, InputKind.OPTIONAL_VARIABLE):
                data = ba.arguments.pop(name)
                # If a non-optional variable OR None, store the arg
                if kind == InputKind.VARIABLE or data is not None:
                    das[name] = data

        # Remove **kwargs from bind object and put all those params in "kwargs" to be passed to compute.
        params = ba.arguments.copy()
        for param in self._sig.parameters.values():
            if param.kind == param.VAR_KEYWORD:
                kwargs = params.pop(param.name)
                params.update(**kwargs)

        # Add injected kwargs to the all_params
        all_params = ba.arguments
        all_params.update(getattr(self._indcompute, "_injected", {}))
        return das, params, all_params

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
    def _get_translated_metadata(
        cls, locale, var_id=None, names=None, append_locale_name=True
    ):
        """Get raw translated metadata for the curent indicator and a given locale.

        All available translated metadata from the current indicator and those it is based on are merged,
        with highest priority to the current one.
        """
        var_id = var_id or ""
        if var_id:
            var_id = "." + var_id

        family_tree = []
        cl = cls
        while hasattr(cl, "_registry_id"):
            family_tree.append(cl._registry_id + var_id)
            # The indicator mechanism always has single inheritance.
            cl = cl.__bases__[0]

        return get_local_attrs(
            family_tree,
            locale,
            names=names,
            append_locale_name=append_locale_name,
        )

    @classmethod
    def _update_attrs(cls, args, das, attrs, var_id=None, names=None):
        """Format attributes with the run-time values of `compute` call parameters.

        Cell methods and history attributes are updated, adding to existing values. The language of the string is
        taken from the `OPTIONS` configuration dictionary.

        Parameters
        ----------
        args: Mapping[str, Any]
          Keyword arguments of the `compute` call.
        das: Mapping[str, DataArray]
          Input arrays.
        attrs : Mapping[str, str]
          The attributes to format and update.
        var_id : str
          The identifier to use when requesting the attributes translations.
          Defaults to the class name (for the translations) or the `identifier` field of the class (for the history attribute).
          If given, the identifier will be converted to uppercase to get the translation attributes.
          This is meant for multi-outputs indicators.
        names : Sequence[str]
          List of attribute names for which to get a translation.

        Returns
        -------
        dict
          Attributes with {} expressions replaced by call argument values. With updated `cell_methods` and `history`.
          `cell_methods` is not added is `names` is given and those not contain `cell_methods`.
        """
        out = cls._format(attrs, args)
        for locale in OPTIONS[METADATA_LOCALES]:
            out.update(
                cls._format(
                    cls._get_translated_metadata(
                        locale, var_id=var_id, names=names or list(attrs.keys())
                    ),
                    args=args,
                    formatter=get_local_formatter(locale),
                )
            )

        # Get history and cell method attributes from source data
        attrs = defaultdict(str)
        if names is None or "cell_methods" in names:
            attrs["cell_methods"] = merge_attributes(
                "cell_methods", new_line=" ", missing_str=None, **das
            )
            if "cell_methods" in out:
                attrs["cell_methods"] += " " + out.pop("cell_methods")

        # Use of OrderedDict to ensure inputs (das) get listed before parameters (args).
        # In the history attr, call signature will be all keywords
        # and might be in a different order than the real function (but order doesn't really matter with keywords).
        kwargs = OrderedDict(**das)
        kwargs.update(**args)
        attrs["history"] = update_history(
            gen_call_string(cls._registry_id, **kwargs),
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

    @classmethod
    def translate_attrs(
        cls, locale: Union[str, Sequence[str]], fill_missing: bool = True
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

        def _translate(cf_attrs, names, var_id=None):
            attrs = cls._get_translated_metadata(
                locale,
                var_id=var_id,
                names=names,
                append_locale_name=False,
            )
            if fill_missing:
                for name in names:
                    if name not in attrs and cf_attrs.get(name):
                        attrs[name] = cf_attrs.get(name)
            return attrs

        # Translate global attrs
        attrs = _translate(
            cls.__dict__,
            # Translate only translatable attrs that are not variable attrs
            set(TRANSLATABLE_ATTRS).difference(set(cls._cf_names)),
        )
        # Translate variable attrs
        attrs["cf_attrs"] = []
        var_id = None
        for cf_attrs in cls.cf_attrs:  # Translate for each variable
            if len(cls.cf_attrs) > 1:
                var_id = cf_attrs["var_name"]
            attrs["cf_attrs"].append(
                _translate(
                    cf_attrs,
                    set(TRANSLATABLE_ATTRS).intersection(cls._cf_names),
                    var_id=var_id,
                )
            )
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
            args.update(getattr(cls._indcompute, "_injected", {}))

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
            return indices.generic.default_freq(**indexer)
        return None

    def _mask(self, *args, **kwds):
        """Return whether mask for output values, based on the output of the `missing` method."""
        from functools import reduce

        indexer = kwds.get("indexer") or {}
        freq = kwds.get("freq") if "freq" in kwds else self._default_freq(**indexer)

        options = self.missing_options or OPTIONS[MISSING_OPTIONS].get(self.missing, {})

        # We flag periods according to the missing method. skip variables without a time coordinate.
        src_freq = self.freq if isinstance(self.freq, str) else None
        miss = (
            self._missing(da, freq, src_freq, options, indexer)
            for da in args
            if "time" in da.coords
        )
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

        Default cfchecks use the specifications in `xclim.core.utils.VARIABLES`,
        assuming the indicator's inputs are using the CMIP6/xclim variable names correctly.
        Variables absent from these default specs are silently ignored.

        When subclassing this method, use functions decorated using `xclim.core.options.cfcheck`.
        """
        for varname, vardata in das.items():
            try:
                cfcheck_from_name(varname, vardata)
            except KeyError:
                # Silently ignore unknown variables.
                pass

    @staticmethod
    def datacheck(**das):
        """Verify that input data is valid.

        When subclassing this method, use functions decorated using `xclim.core.options.datacheck`.

        For example, checks could include:
         - assert temporal frequency is daily
         - assert no precipitation is negative
         - assert no temperature has the same value 5 days in a row
        """
        pass

    def __getattr__(self, attr):
        if attr in self._cf_names:
            out = [meta.get(attr, "") for meta in self.cf_attrs]
            if len(out) == 1:
                return out[0]
            return out
        raise AttributeError(attr)


class Daily(Indicator):
    """Indicator defined for inputs at daily frequency."""

    freq = "D"

    @staticmethod
    def datacheck(**das):  # noqa
        for key, da in das.items():
            if "time" in da.coords and da.time.ndim == 1 and len(da.time) > 3:
                datachecks.check_daily(da)


class Hourly(Indicator):
    """Indicator defined for inputs at strict hourly frequency, meaning 3-hourly inputs would raise an error."""

    freq = "H"

    @staticmethod
    def datacheck(**das):  # noqa
        for key, da in das.items():
            datachecks.check_freq(da, "H")


class DailyWeeklyMonthly(Indicator):
    """Indicator defined for inputs at daily, weekly or monthly frequencies.

    Required by ANUCLIM indicators.
    """

    freq = ["D", "7D", "M"]

    @staticmethod
    def datacheck(**das):  # noqa
        for key, da in das.items():
            if "time" in da.coords and da.time.ndim == 1 and len(da.time) > 3:
                datachecks.check_freq(da, ["D", "7D", "M"], strict=True)


base_registry["Hourly"] = Hourly
base_registry["Daily"] = Daily
base_registry["DailyWeeklyMonthly"] = DailyWeeklyMonthly


def _parse_indice(indice: Callable, passed=None, **new_kwargs):
    """Parse an indice function and return corresponding elements needed for constructing an indicator.

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
    passed = passed or {}

    # Update
    def _upd_param(param):
        # Required DataArray arguments receive their own name as new default
        #         + the Union[str, DataArray] annotation
        if param.kind in [param.VAR_KEYWORD, param.VAR_POSITIONAL]:
            return param

        xckind = infer_kind_from_parameter(param)

        default = passed.get(param.name, {}).get("default", param.default)
        if xckind == InputKind.OPTIONAL_VARIABLE and (
            default is _empty or isinstance(default, str)
        ):
            # Was wrapped with suggested={param: _empty} OR somehow a variable name was injected (ex: through yaml)
            # It becomes a non-optional variable
            xckind = InputKind.VARIABLE
        if default is _empty:
            if xckind == InputKind.VARIABLE:
                default = param.name
            else:
                # Parameters with no default receive None
                # Because we can't have no-default args _after_ default args and we just set the default on the variables (which are the first args)
                default = None

        # Python dont need no switch case
        annots = {
            InputKind.VARIABLE: Union[str, DataArray],
            InputKind.OPTIONAL_VARIABLE: Optional[Union[str, DataArray]],
        }
        annot = annots.get(xckind, param.annotation)

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
        param_doc.update(passed.get(name, {}))

    return indice_wrapper, parsed, params


def add_iter_indicators(module):
    if not hasattr(module, "iter_indicators"):

        def iter_indicators():
            for indname, ind in module.__dict__.items():
                if isinstance(ind, Indicator):
                    yield indname, ind

        iter_indicators.__doc__ = f"Iterate over the (name, indicator) pairs in the {module.__name__} indicator module."

        module.__dict__["iter_indicators"] = iter_indicators


def build_indicator_module(
    name: str,
    objs: Mapping[str, Indicator],
    doc: Optional[str] = None,
) -> ModuleType:
    """Create or update a module from imported objects.

    The module is inserted as a submodule of `xclim.indicators`.

    Parameters
    ----------
    name : str
      New module name. If it already exists, the module is extended with the passed objects,
      overwriting those with same names.
    objs : dict
      Mapping of the indicators to put in the new module. Keyed by the name they will take in that module.
    doc : str
      Docstring of the new module. Defaults to a simple header. Invalid if the module already exists.

    Returns
    -------
    ModuleType
      A indicator module built from a mapping of Indicators.
    """
    from xclim import indicators

    if hasattr(indicators, name):
        if doc is not None:
            warnings.warn(
                "Passed docstring ignored when extending existing module.", stacklevel=1
            )
        out = getattr(indicators, name)
    else:
        doc = doc or f"{name.capitalize()} indicators\n" + "=" * (len(name) + 11)
        try:
            out = ModuleType(name, doc)
        except TypeError as err:
            raise TypeError(f"Module '{name}' is not properly formatted") from err
        indicators.__dict__[name] = out

    out.__dict__.update(objs)
    add_iter_indicators(out)
    return out


def build_indicator_module_from_yaml(
    filename: PathLike,
    name: Optional[str] = None,
    indices: Optional[Union[Mapping[str, Callable], ModuleType]] = None,
    translations: Optional[Mapping[str, dict]] = None,
    mode: str = "raise",
    encoding: str = "UTF8",
) -> ModuleType:
    """Build or extend an indicator module from a YAML file.

    The module is inserted as a submodule of `xclim.indicators`. When given only a base filename (no 'yml' extesion), this
    tries to find custom indices in a module of the same name (*.py) and translations in json files (*.<lang>.json), see Notes.

    Parameters
    ----------
    filename: PathLike
      Path to a YAML file or to the stem of all module files. See Notes for behaviour when passing a basename only.
    name: str, optional
      The name of the new or existing module, defaults to the basename of the file.
      (e.g: `atmos.yml` -> `atmos`)
    indices : Mapping of callables or module, optional
      A mapping or module of indice functions. When creating the indicator, the name in the `index_function` field is
      first sought here, then in xclim.indices.generic and finally in xclim.indices.
    translations  : Mapping of dicts, optional
      Translated metadata for the new indicators. Keys of the mapping must be 2-char language tags.
      See Notes and :ref:`Internationalization` for more details.
    mode: {'raise', 'warn', 'ignore'}
      How to deal with broken indice definitions.
    encoding: str
      The encoding used to open the `.yaml` and `.json` files.
      It defaults to UTF-8, overriding python's mechanism which is machine dependent.

    Returns
    -------
    ModuleType
      A submodule of `xclim.indicators`.

    Notes
    -----
    When the given `filename` has no suffix (usually '.yaml' or '.yml'), the function will try to load
    custom indice definitions from a file with the same name but with a `.py` extension. Similarly,
    it will try to load translations in `*.<lang>.json` files, where `<lang>` is the IETF language tag.

    For example. a set of custom indicators could be fully described by the following files:

        - `example.yml` : defining the indicator's metadata.
        - `example.py` : defining a few indice functions.
        - `example.fr.json` : French translations
        - `example.tlh.json` : Klingon translations.

    See also
    --------
    The doc of :py:mod:`xclim.core.indicator` and of :py:func:`build_module`.
    """
    filepath = Path(filename)

    if not filepath.suffix:
        # A stem was passed, try to load files
        ymlpath = filepath.with_suffix(".yml")
    else:
        ymlpath = filepath

    # Read YAML file
    with ymlpath.open(encoding=encoding) as f:
        yml = safe_load(f)

    # Load values from top-level in yml.
    # Priority of arguments differ.
    module_name = name or yml.get("module", filepath.stem)
    default_base = registry.get(
        yml.get("base"), base_registry.get(yml.get("base"), Daily)
    )
    doc = yml.get("doc")

    # When given as a stem, we try to load indices and translations
    if not filepath.suffix:
        if indices is None:
            try:
                indices = load_module(filepath.with_suffix(".py"))
            except ModuleNotFoundError:
                pass

        if translations is None:
            translations = {}
            for locfile in filepath.parent.glob(filepath.stem + ".*.json"):
                locale = locfile.suffixes[0][1:]
                translations[locale] = read_locale_file(
                    locfile, module=module_name, encoding=encoding
                )

    # Module-wide default values for some attributes
    defkwargs = {
        # Only used in case the indicator definition does not give them.
        "realm": yml.get("realm", "atmos"),
        # Merged with a space
        "keywords": yml.get("keywords"),
        # Merged with a new line
        "references": yml.get("references"),
    }

    def _merge_attrs(dbase, dextra, attr, sep):
        """Merge or replace attribute in dbase from dextra."""
        a = dbase.get(attr)
        b = dextra.get(attr)
        # If both are not None and sep is a string, join.
        if a and b and sep is not None:
            dbase[attr] = sep.join([a, b])
        # If both are not None but sep is, this overrides with b
        # also fills when a is simply missing
        elif b:
            dbase[attr] = b

    # Parse the indicators:
    mapping = {}
    for identifier, data in yml["indicators"].items():
        try:
            # Get base class if it was relative to this module
            if "base" in data:
                if data["base"].startswith("."):
                    # A point means the base has been declared above.
                    data["base"] = registry[module_name + data["base"].upper()]
            else:
                # If no base is specified, pass the default one.
                data["base"] = default_base

            # Get the compute function if it is from the passed mapping
            if indices is not None and "compute" in data:
                indice_name = data["compute"]
                indice_func = getattr(indices, indice_name, None)
                if indice_func is None and hasattr(indices, "__getitem__"):
                    try:
                        indice_func = indices[indice_name]
                    except KeyError:
                        pass

                if indice_func is not None:
                    data["compute"] = indice_func

            _merge_attrs(data, defkwargs, "references", "\n")
            _merge_attrs(data, defkwargs, "keywords", " ")
            _merge_attrs(data, defkwargs, "realm", None)

            mapping[identifier] = Indicator.from_dict(
                data, identifier=identifier, module=module_name
            )

        except Exception as err:
            raise_warn_or_log(
                err, mode, msg=f"Constructing {identifier} failed with {err!r}"
            )

    # Construct module
    mod = build_indicator_module(module_name, objs=mapping, doc=doc)

    # If there are translations, load them
    if translations:
        for locale, locdict in translations.items():
            load_locale(locdict, locale)

    return mod

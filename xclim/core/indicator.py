# -*- coding: utf-8 -*-
# noqa: D205,D400
"""
Indicator base classes
======================

The `Indicator` class wraps indices computations with pre- and post-processing functionality. Prior to computations,
the class runs data and metadata health checks. After computations, the class masks values that should be considered
missing and adds metadata attributes to the output object.

Defining new indicators
=======================

The key ingredients to create a new indicator are the `identifier`, the `compute` function, the name of the missing
value algorithm, and the `datacheck` and `cfcheck` functions, which respectively assess the validity of data and
metadata. The `indicators` module contains over 50 examples of indicators to draw inspiration from.

New indicators can be created using standard Python subclasses::

    class NewIndicator(xclim.core.indicator.Indicator):
        identifier = "new_indicator"
        missing = "any"

        @staticmethod
        def compute(tas):
            return tas.mean(dim="time")

        @staticmethod
        def cfcheck(tas):
            xclim.core.cfchecks.check_valid(tas, "standard_name", "air_temperature")

        @staticmethod
        def datacheck(tas):
            xclim.core.datachecks.check_daily(tas)

Another mechanism to create subclasses is to call Indicator with all the attributes passed as arguments::

    Indicator(identifier="new_indicator", compute=xclim.core.indices.tg_mean, var_attrs=[{'var_name': 'tmean', 'units':"K"}])

Behind the scene, this will create a `NEW_INDICATOR` subclass and return an instance. Note that variable metadata
attributes should passed with the "var_attrs" keyword argument. This limits the possibility of using
class inheritance to fill/override those attributes. However, for the common case where the indicator
only has one output, metadata can be given directly as keyword arguments or as class attributes, altough
the latter is not recommended if not necessary. Thus, the two following example will have the exact same
behavior as the line above:

.. code-block:: python

    # Ex 1
    Indicator(identifier="new_indicator", compute=xclim.core.indices.tg_mean, var_name='tmean', units="K")

    # Ex 2 - Use this method of setting variable metadata attributes only when necessary.
    class NewIndicator(xclim.core.indicator.Indicator):
        identifier = "new_indicator"
        missing = "any"
        var_name = 'tmean'
        units = 'K'
    NewIndicator()


One pattern to create multiple indicators is to write a standard subclass that declares all the attributes that
are common to indicators, then call this subclass with the custom attributes. See for example in
`xclim.indicators.atmos` how indicators based on daily mean temperatures are created from the :class:`Tas` subclass
of the :class:`Daily` subclass.

Subclass registries
-------------------
All subclasses that are created from :class:`Indicator` are stored in a *registry*. So for
example::

>>> from xclim.core.indicator import Daily, registry  # doctest: +SKIP
>>> my_indicator = Daily(identifier="my_indicator", compute=lambda x: x.mean())  # doctest: +SKIP
>>> assert "MY_INDICATOR" in registry  # doctest: +SKIP

This registry is meant to facilitate user customization of existing indicators. So for example, it you'd like
a `tg_mean` indicator returning values in Celsius instead of Kelvins, you could simply do::

>>> from xclim.core.indicator import registry
>>> tg_mean_c = registry["TG_MEAN"](identifier="tg_mean_c", units="C")  # doctest: +SKIP

"""
import re
import warnings
from collections import OrderedDict, defaultdict
from inspect import signature
from typing import Sequence, Union

import numpy as np
from boltons.funcutils import wraps
from xarray import Dataset

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
from .options import MISSING_METHODS, MISSING_OPTIONS, OPTIONS, OUTPUT_DATASET
from .units import convert_units_to, units

# Indicators registry
registry = {}


class Indicator:
    r"""Climate indicator base class.

    Climate indicator object that, when called, computes an indicator and assigns its output a number of
    CF-compliant attributes. Some of these attributes can be *templated*, allowing metadata to reflect
    the value of call arguments.

    Instantiating a new indicator returns an instance but also creates and registers a custom subclass.

    The output type is controlled by global option "output_datasets". If True, calling an indicator always
    returns a Dataset. If False, Indicators returning only one variable will return a single DataArray, whereas
    indicators returning multiple variables will still return datasets.

    Parameters whose names are in `Indicator._cf_global` will be added to the output dataset (if applicable).
    Parameters in `Indicator._cf_var` will be added to the output variables. In the case that global option
    "output_dataset" is False, that the indicator outputs a single DataArray and that a parameter is missing
    from `var_attrs`, then the global version of that parameter will be used instead.

    Parameters
    ----------
    identifier: str
      Unique ID for class registry, should be a valid slug.
    compute: func
      The function computing the indicators. It should return a sequence of more than one DataArray.
    var_attrs : Sequence[Mapping[str, str]]
      List of dictionaries containing the attributes of the individual output arrays. Its length and order should match the output of the compute function.
      Each dictionary should at least define the `var_name` parameter.
      In the case of a single output, the parameters may be given directly as kwargs of the init.
      Valid parameters are (all strings):

      var_name : Variable name in the dataset. If not given in the single-output case, `identifier` is used.

      standard_name : Variable name (CF).

      long_name : Descriptive variable name.

      units : Representative units of the physical quantity (CF).

      cell_methods : List of blank-separated words of the form "name: method" (CF).

      description : Sentence meant to clarify the qualifiers of the fundamental quantities

      keywords : Comma separated list of keywords.

      comment : Miscellaneous information about the data or methods used to produce it.

    description: str
      Sentence meant to clarify the qualifiers of the fundamental quantities, such as which
      surface a quantity is defined on or what the flux sign conventions are.
    title: str, None
      A succinct description of what is in the computed outputs. Parsed from `compute` docstring if None.
    abstract: str
      A long description of what is in the computed outputs. Parsed from `compute` docstring if None.
    keywords: str
      Comma separated list of keywords. Parsed from `compute` docstring if None.
    references: str
      Published or web-based references that describe the data or methods used to produce it. Parsed from
      `compute` docstring if None.
    comment: str
      Miscellaneous information about the data or methods used to produce it.
    notes: str
      Notes regarding computing function, for example the mathematical formulation. Parsed from `compute`
      docstring if None.
    missing: {any, wmo, pct, at_least_n, skip, from_context}
      The name of the missing value method. See `xclim.core.checks.MissingBase` to create new custom methods. If
      None, this will be determined by the global configuration (see `xclim.set_options`). Defaults to "from_context".
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

    # Allowed metadata attributes for the variables
    _var_names = [
        "var_name",
        "standard_name",
        "long_name",
        "units",
        "cell_methods",
        "description",
        "keywords",
        "comment",
    ]
    # Allowed metadata attributes on the output variables
    _cf_var = [
        "var_name",
        "standard_name",
        "long_name",
        "units",
        "cell_methods",
        "description",
        "comment",
    ]
    # Allowed metadata attributes on the output dataset
    _cf_global = ["description", "references"]

    # metadata fields that are formatted as free text.
    _text_fields = ["long_name", "description", "comment"]

    _funcs = ["compute", "cfcheck", "datacheck"]

    # Default attribute values
    identifier = None
    # Default to 1 empty dict,  __new__ will populate it with kwargs-passed attrs.
    var_attrs = [{}]
    missing = "from_context"
    missing_options = None
    context = "none"

    # Global metadata
    description = ""
    title = ""
    abstract = ""
    keywords = ""
    references = ""
    comment = ""
    notes = ""

    def __new__(cls, **kwds):
        """Create subclass from arguments."""
        identifier = kwds.get("identifier", getattr(cls, "identifier"))
        if identifier is None:
            raise AttributeError("`identifier` has not been set.")

        if "var_attrs" not in kwds:
            kwds["var_attrs"] = getattr(cls, "var_attrs")

        # Parse `compute` docstring to extract missing attributes
        # Priority: explicit arguments > super class attributes > `compute` docstring info
        func = kwds.get("compute", None) or cls.compute
        parsed = parse_doc(func.__doc__)

        attrs = {}
        for name, value in parsed.copy().items():
            if not getattr(cls, name, True):
                # Set if it is a class attr but not set by the class
                attrs[name] = value
            elif name in cls._var_names:
                kwds["var_attrs"][0].setdefault(name, value)

        # Single output case : accept var_attrs in kwargs
        if len(kwds["var_attrs"]) == 1:
            for name in cls._var_names:
                if name in kwds or hasattr(cls, name):
                    # Attributes existing in the class are also global metadata fields
                    #   do not remove them from kwargs.
                    kwds["var_attrs"][0].setdefault(
                        name,
                        kwds.get(name, getattr(cls, name))
                        if hasattr(cls, name)
                        else kwds.pop(name),
                    )
            kwds["var_attrs"][0].setdefault("var_name", identifier)
        else:  # Multi output : Refuse var_attrs in kwargs
            for name in cls._var_names:
                if name in kwds and not hasattr(cls, name):
                    raise AttributeError(
                        f"Variable attributes must be set through 'var_attrs' if there are more than one output variables. ({name} was passed as kwarg)"
                    )

        # Convert function objects to static methods.
        for key in cls._funcs + cls._cf_global:
            if key in kwds and callable(kwds[key]):
                kwds[key] = staticmethod(kwds[key])

        attrs.update(kwds)
        # Create new class object
        new = type(identifier.upper(), (cls,), attrs)

        # Set the module to the base class' module. Otherwise all indicators will have module `xclim.core.indicator`.
        new.__module__ = cls.__module__

        #  Add the created class to the registry
        cls.register(new)

        # This will create an instance from the new class and call __init__.
        return super().__new__(new)

    @classmethod
    def register(cls, obj):
        """Add subclass to registry."""
        name = obj.__name__
        if name in registry:
            warnings.warn(f"Class {name} already exists and will be overwritten.")
        registry[name] = obj

    def __init__(self, **kwds):
        """Run checks and assign default values."""
        # Check identifier is well formed - no funny characters
        self.identifier = kwds.pop("identifier", self.identifier)
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

        # The `compute` signature
        self._sig = signature(self.compute)

        # The input parameters' name
        self._parameters = tuple(self._sig.parameters.keys())

        # Copy the docstring and signature
        self.__call__ = wraps(self.compute)(self.__call__)

    def __call__(self, *args, **kwds):
        """Call function of Indicator class."""
        # For convenience
        n_outs = len(self.var_attrs)

        # Bind call arguments to `compute` arguments and set defaults.
        ba = self._sig.bind(*args, **kwds)
        ba.apply_defaults()

        # Assume the first arguments are always the DataArrays.
        das = OrderedDict()
        for i in range(self._nvar):
            das[self._parameters[i]] = ba.arguments.pop(self._parameters[i])

        # Metadata attributes from templates
        cf_attrs = self.cf_attrs
        var_id = None
        var_attrs = []
        for attrs in cf_attrs.pop("vars"):
            if n_outs > 1:
                var_id = f"{self.identifier}.{attrs['var_name']}"
            var_attrs.append(
                self.update_attrs(ba, das, attrs, names=self._cf_var, var_id=var_id)
            )

        ds_attrs = self.update_attrs(ba, das, cf_attrs, names=self._cf_global)

        # Pre-computation validation checks on DataArray arguments
        self.bind_call(self.datacheck, **das)
        self.bind_call(self.cfcheck, **das)

        # Compute the indicator values, ignoring NaNs and missing values.
        outs = self.compute(**das, **ba.kwargs)
        if n_outs == 1:
            outs = [outs]

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

        if OPTIONS[OUTPUT_DATASET] or n_outs > 1:
            out = Dataset(data_vars={out.name: out for out in outs}, attrs=ds_attrs)
        else:
            out = outs[0]
            out.attrs.update(ds_attrs)

        # Mask results that do not meet criteria defined by the `missing` method.
        # This means all variables must have the same dimensions...
        mask = self.mask(*das.values(), **ba.arguments)

        return out.where(~mask)

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

        Cell methods and history attributes are updated, adding to existing values. The language of the string is
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

        attrs["history"] = update_history(
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

        def _translate(var_id, var_attrs):
            attrs = get_local_attrs(
                var_id, locale, names=TRANSLATABLE_ATTRS, append_locale_name=False,
            )
            if fill_missing:
                for name in TRANSLATABLE_ATTRS:
                    if name not in attrs and var_attrs.get(name):
                        attrs[name] = var_attrs.get(name)
            return attrs

        # Translate global attrs
        attrs = _translate(self.identifier.upper(), self.__dict__)
        attrs["vars"] = []
        for var_attrs in self.var_attrs:  # Translate for each variable
            attrs["vars"].append(
                _translate(
                    f"{self.identifier.upper()}.{var_attrs['var_name']}", var_attrs
                )
            )
        return attrs

    @property
    def cf_attrs(self):
        """CF-Convention attributes of the output value."""
        attrs = {k: getattr(self, k) for k in self._cf_global if getattr(self, k)}
        # Get attributes of each variables
        # If the attribute is not given, but exists globally, use the global one.
        attrs["vars"] = [
            {k: attrs.get(k) for k in self._cf_var if attrs.get(k)}
            for attrs in self.var_attrs
        ]
        return attrs

    def json(self, args=None):
        """Return a dictionary representation of the class.

        Notes
        -----
        This is meant to be used by a third-party library wanting to wrap this class into another interface.

        """
        names = ["identifier", "title", "abstract", "keywords"]
        out = {key: getattr(self, key) for key in names}
        out.update(self.cf_attrs)
        vars_attrs = out.pop("vars")
        out = self.format(out, args)
        out["vars"] = []
        for var_attrs in vars_attrs:
            out["vars"].append(self.format(var_attrs, args))

        out["notes"] = self.notes

        out["parameters"] = str(
            {
                key: {
                    "default": p.default if p.default != p.empty else None,
                    "desc": "",
                }
                for (key, p) in self._sig.parameters.items()
            }
        )
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
        miss = (self._missing(da, freq, options, indexer) for da in args)

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
    """Indicator at Daily frequency."""

    @staticmethod
    def datacheck(**das):  # noqa
        for key, da in das.items():
            datachecks.check_daily(da)


class Daily2D(Daily):
    """Indicator using two dimensions at Daily frequency."""

    _nvar = 2

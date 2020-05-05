# -*- coding: utf-8 -*-
"""
Indicator base submodule
========================
"""
import datetime as dt
import re
import warnings
from collections import defaultdict
from collections import OrderedDict
from inspect import signature
from typing import Sequence
from typing import Union

import numpy as np
from boltons.funcutils import wraps

from xclim.core import checks
from xclim.core.formatting import AttrFormatter
from xclim.core.formatting import default_formatter
from xclim.core.formatting import merge_attributes
from xclim.core.formatting import parse_doc
from xclim.core.formatting import update_history
from xclim.core.units import convert_units_to
from xclim.core.units import units
from xclim.core.utils import wrapped_partial
from xclim.locales import get_local_attrs
from xclim.locales import get_local_formatter
from xclim.locales import LOCALES


class Indicator:
    r"""Climate indicator base class.

    Climate indicator object that, when called, computes an indicator and assigns its output a number of
    CF-compliant attributes. Some of these attributes can be *templated*, allowing metadata to reflect
    the value of call arguments.

    Instantiating a new indicator returns an instance but also creates and registers a custom subclass.

    Parameters
    ----------
    identifier: str
      Unique ID for class registry, should be a valid slug.
    var_name: str
      Output variable name. May use tags {<tag>}.
    standard_name: str
      Variable name (CF).
    long_name: str
      Descriptive variable name.
    units: str
      Representative units of the physical quantity (CF).
    compute: func
      The function computing the indicator.
    cell_methods: str
      List of blank-separated words of the form "name: method" (CF).
    description: str
      Sentence meant to clarify the qualifiers of the fundamental quantities, such as which
      surface a quantity is defined on or what the flux sign conventions are.
    context: str
      The `pint` unit context, for example use 'hydro' to allow conversion from kg m-2 s-1 to mm/day.
    title: str, None
      A succinct description of what is in the computed output. Parsed from `compute` docstring if None.
    abstract: str
      A long description of what is in the computed output. Parsed from `compute` docstring if None.
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

    Notes
    -----
    All subclasses created are available in the `registry` attribute and can be used to defined custom subclasses.

    """
    # Number of DataArray variables
    _nvar = 1

    # Allowed metadata attributes on the output
    _cf_names = [
        "standard_name",
        "long_name",
        "units",
        "cell_methods",
        "description",
        "comment",
        "references",
    ]

    # metadata fields that are formatted as free text.
    _text_fields = ["long_name", "description", "comment"]

    _funcs = ["compute", "missing_func", "cfprobe", "validate"]

    # Default attribute values
    identifier = None
    var_name = None
    standard_name = ""
    long_name = ""
    units = ""
    missing_func = staticmethod(checks.missing_any)
    cell_methods = ""
    description = ""
    context = "none"
    title = ""
    abstract = ""
    keywords = ""
    references = ""
    comment = ""
    notes = ""

    # Subclass registry
    registry = {}

    def __new__(cls, **kwds):
        """Create subclass from arguments."""
        identifier = kwds.get("identifier", getattr(cls, "identifier"))

        # Handle function objects.
        for key in cls._funcs:
            if key in kwds:
                kwds[key] = staticmethod(kwds[key])

        func = kwds.get("compute", None) or cls.compute

        # Parse `compute` docstring to extract attributes
        attrs = {
            k: v for (k, v) in parse_doc(func.__doc__).items() if not getattr(cls, k)
        }
        attrs.update(kwds)

        # Create new class object and add it to registry
        new = type(identifier.upper(), (cls,), attrs)
        cls.register(new)

        return super(Indicator, cls).__new__(new)

    @classmethod
    def register(cls, obj):
        """Add subclass to registry."""
        name = obj.__name__
        if name in cls.registry:
            warnings.warn(f"Class {name} already exists and will be overwritten.")
        cls.registry[name] = obj

    def __init__(self, **kwds):
        """Run checks and assign default values.
        """

        self.identifier = kwds.pop("identifier", self.identifier)
        self.check_identifier(self.identifier)

        if self.var_name is None:
            self.var_name = self.identifier

        # The signature
        self._sig = signature(self.compute)

        # The input parameter names
        self._parameters = tuple(self._sig.parameters.keys())

        # Copy the docstring and signature
        self.__call__ = wraps(self.compute)(self.__call__)

    def __call__(self, *args, **kwds):
        # Bind call arguments.
        ba = self._sig.bind(*args, **kwds)
        ba.apply_defaults()

        # Assume the first arguments are always the DataArray.
        das = OrderedDict()
        for i in range(self._nvar):
            das[self._parameters[i]] = ba.arguments.pop(self._parameters[i])

        # Metadata attributes from templates
        attrs = self.update_attrs(ba, das)
        vname = attrs.pop("var_name")

        # Pre-computation validation checks
        for da in das.values():
            self.validate(da)
        self.cfprobe(*das.values())

        # Compute the indicator values, ignoring NaNs.
        out = self.compute(**das, **ba.kwargs)

        # Convert to output units
        out = convert_units_to(out, self.units, self.context)

        # Update netCDF attributes
        out.attrs.update(attrs)

        # Bind call arguments to the `missing` function, whose signature might be different from `compute`.
        mba = signature(self.missing).bind(*das.values(), **ba.arguments)

        # Mask results that do not meet criteria defined by the `missing` method.
        mask = self.missing(*mba.args, **mba.kwargs)
        ma_out = out.where(~mask)

        return ma_out.rename(vname)

    def update_attrs(self, ba, das):
        """Update attributes.

        Parameters
        ----------
        ba: bound argument object
          ...
        das: tuple
          Input arrays.
        """
        args = ba.arguments
        out = self.format(self.cf_attrs, args)
        for locale in LOCALES:
            out.update(
                self.format(
                    get_local_attrs(
                        self,
                        locale,
                        names=self._cf_names,
                        fill_missing=False,
                        append_locale_name=True,
                    ),
                    args=args,
                    formatter=get_local_formatter(locale),
                )
            )

        out["var_name"] = vname = self.format({"var_name": self.var_name}, args)[
            "var_name"
        ]

        # Update the signature with the values of the actual call.
        cp = OrderedDict()
        for (k, v) in ba.signature.parameters.items():
            if v.default is not None and isinstance(v.default, (float, int, str)):
                cp[k] = v.replace(default=ba.arguments[k])
            else:
                cp[k] = v

        # Get history and cell method attributes from source data
        attrs = defaultdict(str)
        attrs["cell_methods"] = merge_attributes(
            "cell_methods", new_line=" ", missing_str=None, **das
        )
        if "cell_methods" in out:
            attrs["cell_methods"] += " " + out.pop("cell_methods")

        attrs["history"] = update_history(
            f"{self.identifier}{ba.signature.replace(parameters=cp.values())}",
            new_name=vname,
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

        Translatable attributes are defined in xclim.locales.TRANSLATABLE_ATTRS

        Parameters
        ----------
        locale : Union[str, Sequence[str]]
            The POSIX name of the locale or a tuple of a locale name and a path to a
            json file defining the translations. See `xclim.locale` for details.
        fill_missing : bool
            If True (default fill the missing attributes by their english values.
        """
        return get_local_attrs(
            self, locale, fill_missing=fill_missing, append_locale_name=False
        )

    @property
    def cf_attrs(self):
        """CF-Convention attributes of the output value."""
        attrs = {k: getattr(self, k) for k in self._cf_names if getattr(self, k)}
        return attrs

    def json(self, args=None):
        """Return a dictionary representation of the class.

        Notes
        -----
        This is meant to be used by a third-party library wanting to wrap this class into another interface.

        """
        names = ["identifier", "var_name", "abstract", "keywords"]
        out = {key: getattr(self, key) for key in names}
        out.update(self.cf_attrs)
        out = self.format(out, args)

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

        # if six.PY2:
        #     out = walk_map(out, lambda x: x.decode('utf8') if isinstance(x, six.string_types) else x)

        return out

    def format(
        self,
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

            if key in self._text_fields:
                out[key] = out[key].strip().capitalize()

        return out

    @staticmethod
    def cfprobe(self, *das):
        """Check input data compliance to expectations.
        Warn of potential issues."""
        return True

    @staticmethod
    def compute(self, *args, **kwds):
        """The function computing the indicator."""
        raise NotImplementedError

    def missing(self, *args, **kwds):
        """Return whether an output is considered missing or not."""
        from functools import reduce

        freq = kwds.get("freq")
        if freq is not None:
            # We flag any period with missing data
            miss = (self.missing_func(da, freq) for da in args)
        else:
            # There is no resampling, we flag where one of the input is missing
            miss = (da.isnull() for da in args)
        return reduce(np.logical_or, miss)

    @staticmethod
    def validate(da):
        """Validate input data requirements.
        Raise error if conditions are not met."""
        checks.assert_daily(da)


class Indicator2D(Indicator):
    _nvar = 2

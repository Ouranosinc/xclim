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
from xclim.locales import get_local_attrs
from xclim.locales import get_local_formatter
from xclim.locales import LOCALES


# This class needs to be subclassed by individual indicator classes defining metadata information, compute and
# missing functions. It can handle indicators with any number of forcing fields.
class Indicator:
    r"""Climate indicator based on xarray
    """
    # Unique ID for function registry.
    identifier = ""

    # Output variable name. May use tags {<tag>} that will be formatted at runtime.
    var_name = ""

    _nvar = 1

    # CF-Convention metadata to be attributed to the output variable. May use tags {<tag>} formatted at runtime.
    # The set of permissible standard names is contained in the standard name table.
    standard_name = ""
    long_name = ""  # Parsed.
    units = ""  # Representative units of the physical quantity.
    cell_methods = ""  # List of blank-separated words of the form "name: method"
    description = ""  # The description is meant to clarify the qualifiers of the fundamental quantities, such as which
    #   surface a quantity is defined on or what the flux sign conventions are.

    # The `pint` unit context. Use 'hydro' to allow conversion from kg m-2 s-1 to mm/day.
    context = "none"

    # Additional information that can be used by third party libraries or to describe the file content.
    title = ""  # A succinct description of what is in the dataset. Default parsed from compute.__doc__
    abstract = ""  # Parsed
    keywords = ""  # Comma separated list of keywords
    # Published or web-based references that describe the data or methods used to produce it. Parsed.
    references = ""
    comment = (
        ""  # Miscellaneous information about the data or methods used to produce it.
    )
    notes = ""  # Mathematical formulation. Parsed.

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

    # Can be used to override the compute docstring.
    doc_template = None

    def __init__(self, **kwds):

        # Set instance attributes.
        for key, val in kwds.items():
            setattr(self, key, val)

        # Verify that the identifier is a proper slug
        if not re.match(r"^[-\w]+$", self.identifier):
            warnings.warn(
                "The identifier contains non-alphanumeric characters. It could make life "
                "difficult for downstream software reusing this class.",
                UserWarning,
            )

        # Default value for `var_name` is the `identifier`.
        if self.var_name == "":
            self.var_name = self.identifier

        # Extract information from the `compute` function.
        # The signature
        self._sig = signature(self.compute)

        # The input parameter names
        self._parameters = tuple(self._sig.parameters.keys())
        #        self._input_params = [p for p in self._sig.parameters.values() if p.default is p.empty]
        #        self._nvar = len(self._input_params)

        # Copy the docstring and signature
        self.__call__ = wraps(self.compute)(self.__call__.__func__)
        if self.doc_template is not None:
            self.__call__.__doc__ = self.doc_template.format(i=self)

        # Fill in missing metadata from the doc
        meta = parse_doc(self.compute.__doc__)
        for key in ["abstract", "title", "notes", "references"]:
            setattr(self, key, getattr(self, key) or meta.get(key, ""))

    def __call__(self, *args, **kwds):
        # Bind call arguments. We need to use the class signature, not the instance, otherwise it removes the first
        # argument.
        ba = self._sig.bind(*args, **kwds)
        ba.apply_defaults()

        # Update attributes
        out_attrs = self.format(self.cf_attrs, ba.arguments)
        for locale in LOCALES:
            out_attrs.update(
                self.format(
                    get_local_attrs(
                        self,
                        locale,
                        names=self._cf_names,
                        fill_missing=False,
                        append_locale_name=True,
                    ),
                    args=ba.arguments,
                    formatter=get_local_formatter(locale),
                )
            )
        vname = self.format({"var_name": self.var_name}, ba.arguments)["var_name"]

        # Update the signature with the values of the actual call.
        cp = OrderedDict()
        for (k, v) in ba.signature.parameters.items():
            if v.default is not None and isinstance(v.default, (float, int, str)):
                cp[k] = v.replace(default=ba.arguments[k])
            else:
                cp[k] = v

        # Assume the first arguments are always the DataArray.
        das = OrderedDict()
        for i in range(self._nvar):
            das[self._parameters[i]] = ba.arguments.pop(self._parameters[i])

        # Get history and cell method attributes from source data
        attrs = defaultdict(str)
        attrs["cell_methods"] = merge_attributes(
            "cell_methods", new_line=" ", missing_str=None, **das
        )
        if "cell_methods" in out_attrs:
            attrs["cell_methods"] += " " + out_attrs.pop("cell_methods")
        attrs["history"] = update_history(
            f"{self.identifier}{ba.signature.replace(parameters=cp.values())}",
            new_name=vname,
            **das,
        )
        attrs.update(out_attrs)

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

    def cfprobe(self, *das):
        """Check input data compliance to expectations.
        Warn of potential issues."""
        return True

    def compute(*args, **kwds):
        """The function computing the indicator."""
        raise NotImplementedError

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
    def missing(*args, **kwds):
        """Return whether an output is considered missing or not."""
        from functools import reduce

        freq = kwds.get("freq")
        if freq is not None:
            # We flag any period with missing data
            miss = (checks.missing_any(da, freq) for da in args)
        else:
            # There is no resampling, we flag where one of the input is missing
            miss = (da.isnull() for da in args)
        return reduce(np.logical_or, miss)

    def validate(self, da):
        """Validate input data requirements.
        Raise error if conditions are not met."""
        checks.assert_daily(da)


class Indicator2D(Indicator):
    _nvar = 2

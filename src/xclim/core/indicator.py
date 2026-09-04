"""
Indicator Utilities
===================

The `Indicator` class wraps computations with pre- and post-processing functionality. Prior to computations,
the class runs data and metadata health checks. After computations, the class masks values that should be considered
missing and adds metadata attributes to the output.

There are many ways to construct indicators. A good place to start is
`this notebook <notebooks/extendxclim.ipynb#Defining-new-indicators>`_.
"""  # numpydoc ignore=GL07

from __future__ import annotations

import inspect
import logging
import re
import warnings
from ast import literal_eval
from collections import namedtuple
from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from functools import reduce
from inspect import _empty as _empty_default
from itertools import zip_longest
from typing import Any

import numpy as np
import xarray
from docstring_parser import parse_from_object
from xarray import DataArray, Dataset

import xclim.core.locales as xloc
from xclim.core import (
    KIND_ANNOTATION,
    VARIABLES,
    InputKind,
    MissingVariableError,
    ValidationError,
    datachecks,
    infer_kind_from_parameter,
    is_percentile_dataarray,
)
from xclim.core.calendar import parse_offset, select_time
from xclim.core.cfchecks import cfcheck_from_name
from xclim.core.formatting import (
    capitalize_free_text,
    default_formatter,
    gen_call_string,
    get_percentile_metadata,
    update_history,
)
from xclim.core.options import (
    AS_DATASET,
    CHECK_MISSING,
    METADATA_LOCALES,
    MISSING_METHODS,
    MISSING_OPTIONS,
    OPTIONS,
    set_options,
)
from xclim.core.units import check_units, convert_units_to, declare_units, units
from xclim.core.utils import CaseInsensitiveDict, split_auxiliary_coordinates

# Indicators registry
registry = CaseInsensitiveDict()  # Main indicator registry
base_registry = {}  # Base classes registry


# Sentinel class for unset properties of Indicator's parameters."""
class _empty:  # pylint: disable=too-few-public-methods
    pass


@dataclass
class Parameter:
    """
    Object representing an indicator's parameter.

    For convenience, this class implements a special "contains".

    Examples
    --------
    >>> p = Parameter(InputKind.NUMBER, default=2, description="A simple number")
    >>> p.units is Parameter._empty  # has not been set
    True
    >>> "units" in p  # Easier/retrocompatible way to test if units are set
    False
    >>> p.description
    'A simple number'
    """

    _empty = _empty

    kind: InputKind
    default: Any = _empty_default
    # Name of the compute function's argument corresponding to this parameter.
    compute_name: str = _empty
    description: str = ""
    units: str = _empty
    choices: set = _empty
    value: Any = _empty
    annotation: str = _empty

    def update(self, other: dict) -> None:
        """
        Update a parameter's values from a dict.

        Parameters
        ----------
        other : dict
            A dictionary of parameters to update the current.
        """
        for k, v in other.items():
            if hasattr(self, k):
                setattr(self, k, v)
            else:
                raise AttributeError(f"Unexpected parameter field '{k}'.")

    @classmethod
    def is_parameter_dict(cls, other: dict) -> bool:
        """
        Return whether `other` can update a parameter dictionary.

        Parameters
        ----------
        other : dict
            A dictionary of parameters.

        Returns
        -------
        bool
            Whether `other` can update a parameter dictionary.
        """
        # Passing compute_name is forbidden.
        # name is valid, but is handled by the indicator
        return set(other.keys()).issubset({"kind", "default", "description", "units", "choices", "value", "name"})

    def __contains__(self, key) -> bool:
        """Imitate previous behaviour where "units" and "choices" were missing, instead of being "_empty"."""
        return getattr(self, key, _empty) is not _empty

    @property
    def injected(self) -> bool:
        """
        Indicate whether values are injected.

        Returns
        -------
        bool
            Whether values are injected.
        """
        return self.value is not _empty

    def json(self) -> dict:
        """
        Return a json-serializable dictionary of the Parameter.

        Returns
        -------
        dict
            Dictionary representation of the object, ready for serialization into json.
        """
        if self.injected:
            return deepcopy(self.value)
        out = {
            "kind": self.kind.value,  # Get the int.
            "description": self.description,
        }
        if "choices" in self:  # A set is stored, convert to list
            out["choices"] = list(self.choices)
        if self.default is not _empty_default:
            out["default"] = self.default
        for field in ["annotation", "compute_name", "units"]:
            if field in self:
                out[field] = str(getattr(self, field))
        return out

    def _gen_doc(self, name: str | None = None) -> str:
        """
        Generate an item of a numpydoc style Parameters section for this parameter.

        Parameters
        ----------
        name : str, optional
            A new name for this parameter if it was overridden from the "compute_name".

        Returns
        -------
        str
            A two lines string to be added in a Parameters section of a numpydoc style docstring.
        """
        annot = KIND_ANNOTATION[self.kind]
        if self.choices is not _empty:
            annot = str(self.choices)
        default = " Required."
        if self.default is not _empty_default:
            default = f" Default: {self.default!r}."
        elif self.kind == InputKind.KWARGS:
            default = ""
        unit = ""
        if self.units is not _empty:
            unit = f" [Required units : {self.units}]"

        return f"{name or self.compute_name} : {annot}\n {self.description}{default}{unit}"

    def _gen_signature(self, name: str | None = None) -> inspect.Parameter:
        """Generate a :py:class:`inspect.Parameter` object from this Parameter."""
        name = name or self.compute_name
        if self.kind == InputKind.KWARGS:
            return inspect.Parameter(name, kind=inspect.Parameter.VAR_KEYWORD)
        if self.kind in [InputKind.VARIABLE, InputKind.OPTIONAL_VARIABLE]:
            kind = inspect.Parameter.POSITIONAL_OR_KEYWORD
        else:
            kind = inspect.Parameter.KEYWORD_ONLY
        annot = self.annotation if self.annotation is not _empty else KIND_ANNOTATION[self.kind]
        return inspect.Parameter(name, kind=kind, default=self.default, annotation=annot)


class Output(dict):  # numpydoc ignore=PR01
    """Dictionary metadata for the output of an indicator."""

    var_name: str | None
    """Output variable name."""

    dimensionality: str | None
    """Dimensionality specification, similar but not necessarily compatible with pint."""

    units: str | None
    """Units of the output."""

    units_metadata: str | None
    """Additional CF metadata for the units."""

    def __init__(
        self,
        var_name: str | None = None,
        dimensionality: str | None = None,
        units: str | None = None,
        units_metadata: str | None = None,
        **kwargs,
    ):
        """
        Create an output attributes dictionary.

        Parameters
        ----------
        var_name : str, optional
            The name of the output variable.
        dimensionality : str, optional
            A pint-like dimensionality string. This is not added to the attribute dictionary
            This field is a human-readable indication for documenting the output.
        units : str, optional
            Units of the output. When set, the indicator computation will explicitly convert the output.
        units_metadata : str, optional
            Additional CF metadata for the units.
        **kwargs
            Any other attribute describing the metadata, which will be added as attributes.
            Usually, output variable will use `standard_name`, `long_name` and `description`.
        """
        self.var_name = var_name
        self.dimensionality = dimensionality
        self.units = units
        self.units_metadata = units_metadata
        super().__init__(**kwargs)

    @property
    def meta(self) -> dict:
        """
        A dictionary of the non-attribute metadata for this output.

        Returns
        -------
        dict
            The non-attribute metadata of this Output.
        """
        return {
            "var_name": self.var_name,
            "dimensionality": self.dimensionality,
            "units": self.units,
            "units_metadata": self.units_metadata,
        }

    def __or__(self, other):
        """Dict-style merging via the OR operator."""
        meta = self.meta
        if isinstance(other, Output):
            other_meta = other.meta
            other_attrs = dict(other)
        else:
            other_meta = {k: v for k, v in other.items() if k in meta}
            other_attrs = {k: v for k, v in other.items() if k not in meta}
        merged = (meta | other_meta) | (dict(self) | other_attrs)
        return self.__class__(**merged)

    def __repr__(self):
        """Readable representation."""
        meta = ", ".join(f"{k}='{v}'" for k, v in self.meta.items() if k != "var_name" and v is not None)
        attrs = ", ".join(f"{k}='{v}'" for k, v in self.items())
        return f"<{self.__class__.__name__} {self.var_name or '[unnamed]'} ({meta}) : {attrs}>"

    def _gen_doc(self, multiple_returns: bool = True) -> str:
        """
        Generate an item of a numpydoc style Returns section for this output.

        Parameters
        ----------
        multiple_returns: bool
            If True and the `var_name` is defined, it is added at the beginning of the string.
            The numpydoc style forbids adding an output name if there's only one output.

        Returns
        -------
        str
            A two lines string to be added in a Returns section of a numpydoc style docstring.
        """
        name = ""
        if multiple_returns and self.var_name is not None:
            name = f"{self.var_name} : "
        dim = ""
        if self.units:
            dim = f", [{self.units}]"
        elif self.dimensionality:
            dim = f", {self.dimensionality}"
        sname = "" if self.get("standard_name") is None else f"{self['standard_name']}, "
        add = "."
        if other := (set(self.keys()) - {"standard_name", "long_name"}):
            add = f". With additional attributes: {', '.join([f'**{k}**: ``{self[k]}``' for k in other])}"
        return f"{name}xarray.DataArray{dim}\n  {sname}{self.get('long_name', '')}{add}"


class IndexWrapper:  # numpydoc ignore=PR01
    """
    Template object wrapping an index-like compute function by parsing its signature, docstring and declared units.

    This class is not instantiable, but used as a base for :py:class:`IndicatorBase`,
    itself the base of :py:class:`Indicator`.
    """

    title: str  # First line of the docstring
    """Short description of the indicator."""
    abstract: str  # Second paragraph of the docstring
    """Description of the indicator."""

    # Other sections of the docstring
    notes: str
    """Additional information about the indicator."""
    references: str
    """
    rst cite directives for literature about this indicator.
    Child classes append their references as a new line when inheriting.
    """

    _all_parameters: Mapping[str, Parameter]  # Parameter section
    """A dictionary mapping metadata about the input parameters to the indicator.

    Keys are the arguments of the "compute" function. All parameters are listed, even
    those "injected", absent from the indicator's call signature. All are instances of
    :py:class:`~xclim.core.indicator.Parameter`.
    """

    attrs: list[Output]  # Returns section
    """List of output metadata."""

    def __new__(cls, compute):
        """
        Create an IndexWrapper from a compute function.

        Parameters
        ----------
        compute : callable
          A function, annotated, documentated and wrapped by :py:func:`xclim.core.units.declare_units`
          as explained in :ref:`notebooks/extendxclim:Defining new index-like compute functions`.

        Returns
        -------
        dict
          Metadata extracted from the function.
        """
        doc = parse_from_object(compute)
        sig = inspect.signature(compute)
        declared_units = getattr(compute, "in_units", {})

        if doc.description is None:
            title = ""
            abstract = ""
        elif "\n\n" in doc.description:
            title, abstract, *_ = doc.description.split("\n\n")
            title = title.replace("\n", " ")
            abstract = abstract.replace("\n", " ")
        else:
            title = doc.description.replace("\n", " ")
            abstract = ""

        doc_params = {p.arg_name.replace("*", ""): p for p in doc.params}

        # Check that the `Parameters` section of the docstring does not include parameters
        # that are not in the `compute` function signature.
        missing = set(doc_params.keys()) - set(sig.parameters.keys())
        if missing:
            raise ValueError(
                f"Malformed docstring on {compute} : the parameters {missing} are absent from the signature."
            )

        # Parse inputs.
        inputs = {}
        for name, sigparam in sig.parameters.items():
            kind = infer_kind_from_parameter(sigparam)
            default = sigparam.default
            compute_name = name

            units = declared_units.get(name, _empty)

            description = ""
            choices = _empty
            if name in doc_params:
                description = doc_params[name].description.replace("\n", " ")

                choices_raw = None
                docannot = doc_params[name].type_name.strip()
                # To string to cover both cases where it is a Literal type or a string already
                sigannot = str(sigparam.annotation)
                if "dim: indexer" not in docannot and (match := re.match(r"(\{.*\})", docannot)):
                    choices_raw = match.groups()[0]
                elif match := re.match(r"Literal\[(.*)\]", sigannot):
                    choices_raw = "{" + match.groups()[0] + "}"
                if choices_raw:
                    try:
                        choices = literal_eval(choices_raw)
                        if doc_params[name].is_optional or default is None:
                            choices.add(None)
                    except ValueError:
                        # If the literal_eval fails, we just ignore the choices.
                        msg = (
                            f"Choices defined in the description of parameter {name}"
                            f" of function {compute} could not be parsed. "
                            f"Got: {choices_raw}."
                        )
                        logging.error(msg)

            annotation = _empty if sigparam.annotation == _empty_default else sigparam.annotation
            inputs[name] = Parameter(
                kind=kind,
                default=default,
                compute_name=compute_name,
                units=units,
                description=description,
                choices=choices,
                annotation=annotation,
            )

        # Parse outputs
        outputs = []
        for r in doc.many_returns:
            dimensionality = None
            if ", " in r.type_name:
                _, dimensionality = r.type_name.split(", ")
            outputs.append(Output(var_name=r.return_name, dimensionality=dimensionality, long_name=r.description or ""))

        notes = ""
        references = ""
        for section in doc.meta:
            if section.args == ["notes"]:
                notes = section.description
            elif section.args == ["references"]:
                references = section.description

        new = type(
            cls.__name__,
            (cls,),
            {
                "title": title,
                "abstract": abstract,
                "_all_parameters": inputs,
                "attrs": outputs,
                "notes": notes,
                "references": references,
                "compute": staticmethod(compute),
            },
        )
        return super().__new__(new)

    @staticmethod
    def compute(*args, **kwargs):  # numpydoc ignore=PR01
        """The index-like compute function."""
        raise NotImplementedError()

    @property
    def n_outs(self) -> int:
        """
        The number of outputs of this indicator.

        Returns
        -------
        int
            The number of outputs.
        """
        return len(self.attrs)

    @property
    def parameters(self) -> Mapping[str, Parameter]:
        """
        Dictionary of controllable (non-injected) parameters.

        Similar to :py:attr:`IndexWrapper._all_parameters`, but doesn't include injected parameters.

        Returns
        -------
        dict
            A dictionary of controllable parameters.
        """
        return {name: param for name, param in self._all_parameters.items() if not param.injected}

    @property
    def injected_parameters(self) -> Mapping[str, Any]:
        """
        Dictionary of all injected parameters (values).

        Returns
        -------
        dict
            A dictionary of all injected parameters' values.
        """
        return {name: param.value for name, param in self._all_parameters.items() if param.injected}

    @property
    def is_generic(self) -> bool:
        """
        If the indicator is "generic" returns True, meaning that it can accept variables with any units.

        Returns
        -------
        bool
            True if the indicator is generic.
        """
        return not hasattr(self.compute.__wrapped__, "in_units")

    def _extra_doc(self) -> list[str]:
        """
        Method returning extra lines to the docstring, under the abstract.

        Children classes can implemented this, appending to anything returned by super()._extra_doc().
        """
        return []

    def _gen_docstring(self):
        """Generate a docstring for this indicator."""
        extra = self._extra_doc()

        source = []
        if hasattr(self.compute, "__module__"):
            source.append(f"Based on function :py:func:`~{self.compute.__module__}.{self.compute.__name__}`.")
        else:
            source.append(f"Based on function {self.compute.__name__}.")
        if self.injected_parameters:
            source.append(
                f"With injected parameters: {', '.join([f'{k}={v}' for k, v in self.injected_parameters.items()])}."
            )
        extra = "\n".join(extra + source)

        paramstext = "\n".join([p._gen_doc(name=n) for n, p in self.parameters.items()])
        parameters = f"Parameters\n----------\n{paramstext}"
        returnstext = "\n".join([o._gen_doc(multiple_returns=len(self.attrs) > 1) for o in self.attrs])
        returns = f"Returns\n-------\n{returnstext}"

        extra_sections = []
        if self.notes:
            extra_sections.append(f"Notes\n-----\n{self.notes}\n")
        if self.references:
            extra_sections.append(f"References\n----------\n{self.references}\n")

        doc = "\n\n".join([self.title, self.abstract, extra, parameters, returns, *extra_sections])
        return doc

    def _gen_signature(self):
        """
        Generate a signature for the indicator, skipping injected parameters.

        The signature might be invalid if parameters are not properly ordered with
        [optional] variables first and kwargs last.
        """
        parameters = []
        for name, param in self.parameters.items():
            parameters.append(param._gen_signature(name=name))
        ret_ann = DataArray if self.n_outs == 1 else tuple[(DataArray,) * self.n_outs]  # ty: ignore[invalid-type-form]
        return inspect.Signature(parameters, return_annotation=ret_ann)

    def __init__(self, **kwds):
        self.__signature__ = self._gen_signature()
        self.__doc__ = self._gen_docstring()


class IndicatorBase(IndexWrapper):
    """Extends IndexWrapper by allowing metadata overrides, adding non-parsable fields and orchestrating computation."""

    identifier: str = None
    """Unique ID identifying this indicator. Mostly for registry purposes."""

    realm: str = None
    """General domain of validity of the indicator. Should use the same vocabulary as CMIP."""

    keywords: tuple[str] = ()
    """
    Keywords describing the indicator and its domains of application.
    Child classes append to the list when inheriting.
    """
    context: str = "none"
    """Name of `xclim.units.units` context which will be enabled during computation."""

    def __new__(cls, **kwds):
        """Create a new indicator but also a new class."""
        identifier = kwds.get("identifier", cls.identifier)
        if identifier is None:
            raise TypeError(f"Missing argument 'identifier' to constructor of {cls.__name__}.")

        # Need to get this before the IndexWrapper twist
        module = kwds.pop("module", None)

        if "compute" in kwds:
            if cls.compute is not IndexWrapper.compute:
                raise TypeError(
                    "Can't change the compute function of an indicator, create from a base class instead. "
                    f"Indicator {identifier} got {kwds['compute']} but already has {cls.compute}."
                )
            # Create a new subclass, using IndexWrapper's parsing
            # Python doesn't have a hook purely for class creation, so we cheat by creating a transitory instance
            cls = IndexWrapper.__new__(cls, kwds["compute"]).__class__

            # Subclasses can override or extend this through the classmethod _added_parameters
            # We add them to the indicator at the same time as parsing the compute.
            for name, param in cls._added_parameters().items():
                if name in cls._all_parameters:
                    raise ValueError(
                        f"Class {cls.__name__} can't wrap compute functions that have a `{name}`"
                        " argument as it conflicts with an argument it adds."
                    )
                cls._all_parameters[name] = param
        elif cls.compute is IndexWrapper.compute:
            raise TypeError(f"Missing argument 'compute' to constructor of {cls.__name__}.")

        parameters, new_units = cls._update_parameters(
            deepcopy(cls._all_parameters), kwds.pop("parameters", {}), kwds.pop("input", {})
        )
        if new_units and not hasattr(cls.compute, "in_units"):  # Update units in the compute function
            compute = declare_units(**new_units)(cls.compute)
            # Update non-variable parameter metadata, assuming previous compute was decorated with
            # `declare_relative_units`, otherwise this does nothing
            for name, units in compute.in_units.items():
                if name not in new_units:
                    p = [p for p in parameters.values() if p.compute_name == name][0]
                    p.units = units
        else:
            compute = cls.compute
        # Without this, compute becomes a bound method
        kwds["compute"] = staticmethod(compute)

        parameters = cls._ensure_correct_parameters(parameters)
        kwds["_all_parameters"] = parameters

        # Output Attributes
        attrs = cls._update_attrs(cls.attrs, kwds.pop("attrs", None))
        attrs = cls._ensure_correct_attrs(attrs, identifier)
        kwds["attrs"] = attrs

        # Other metadata
        # If these fields were not given, set them from the parsed docstring
        for field in ["title", "abstract", "notes"]:
            kwds.setdefault(field, getattr(cls, field))
        # Special inheritance rules for keywords and references
        if "references" in kwds and cls.references:
            kwds["references"] = f"{cls.references}\n{kwds['references']}"
        kwds["keywords"] = tuple((*cls.keywords, *kwds.get("keywords", [])))

        # Create new class object
        new = type(identifier.upper(), (cls,), kwds)

        # Module is normally set to the file in which the class is defined
        # We are creating classes dynamically, so we allow patching the module to get meaningful metadata
        if module is not None:
            new.__module__ = module
        return object.__new__(new)

    @classmethod
    def _added_parameters(cls):
        return {}

    @classmethod
    def _update_parameters(cls, parameters, new_params, var_mapping):
        """
        Merge parent input parameters with passed specifications, rename variables.

        Parameters
        ----------
        parameters : dict of Parameters
            Dict of :py:class:`~xclim.core.indicator.Parameter` objects.
        new_params : dict
            Dict of parameters overrides passed to the indicator constructor (as `parameters`).
        var_mapping : dict
            Mapping from variable name in the parent indicator or compute function to new name.
            The new name must be known by xclim, it must be in :py:data:`xclim.core.VARIABLES`.

        Returns
        -------
        dict
            The merged dict of Parameter objects.
        dict
            For renamed variables, this maps the name in the compute function to the new units.
        """
        new_units = {}
        for old_name, new_name in var_mapping.items():
            meta = parameters[new_name] = parameters.pop(old_name)
            try:
                var_meta = VARIABLES[new_name]
            except KeyError as err:
                raise ValueError(
                    f"Compute argument {old_name} was mapped to variable "
                    f"{new_name} which is not understood by xclim or CMIP6. Please"
                    " use names listed in `xclim.core.VARIABLES`."
                ) from err
            if meta.units is not _empty:
                try:
                    check_units(var_meta["canonical_units"], meta.units)
                except ValidationError as err:
                    raise ValueError(
                        "When changing the name of a variable by passing `input`, "
                        "the units dimensionality must stay the same. Got: old = "
                        f"{meta.units}, new = {var_meta['canonical_units']}"
                    ) from err
            meta.units = var_meta.get("dimensions", var_meta["canonical_units"])
            new_units[meta.compute_name] = meta.units
            meta.description = var_meta["description"]

        # Parse passed "parameters", allowing overriding their metadata and behaviour
        for key, val in new_params.items():
            if key not in parameters:
                raise ValueError(
                    f"Parameter {key} was passed but it does not exist on the "
                    f"compute function (not one of {parameters.keys()})"
                )
            if isinstance(val, dict) and Parameter.is_parameter_dict(val):
                if "units" in val:
                    raise ValueError(
                        "Can only change expected dimensions/units of a parameter through the `input` argument."
                        f" Got overrides {val} for parameter {key}"
                    )
                if "name" in val:
                    new_key = val.pop("name")
                    if new_key in new_params:
                        raise ValueError(
                            "Cannot rename a parameter or variable with the same name as another parameter. "
                            f"'{new_key}' is already a parameter."
                        )
                    parameters[new_key] = parameters.pop(key)
                    key = new_key
                parameters[key].update(val)
            else:  # val is not a Parameter dict, thus an injected value
                parameters[key].value = val
        return parameters, new_units

    @classmethod
    def _ensure_correct_parameters(cls, parameters):
        """
        Ensure all input parameters are correct.

        Parameters
        ----------
        parameters : dict of Parameters
            Dict of :py:class:`xclim.core.indicator.Parameter` objects.

        Returns
        -------
        dict
            Same as input, potentially modified.
        """

        # Sort parameters : Var, Opt Var, all params, ds, injected params.
        def sortkey(kv):
            if not kv[1].injected:
                if kv[1].kind in [InputKind.VARIABLE, InputKind.OPTIONAL_VARIABLE, InputKind.KWARGS]:
                    return kv[1].kind
                return 2
            return 99

        return dict(sorted(parameters.items(), key=sortkey))

    @classmethod
    def _update_attrs(cls, attrs, new_attrs):
        """
        Merge parent output attributes with passed specifications.

        Parameters
        ----------
        attrs : list of Output
            List of :py:class:`Output` objects.
        new_attrs : list of dict or list of Output or dict
            The output metadata passed to the indicator constructor.

        Returns
        -------
        list of Output
            The merged list of Output objects, as long as the longest of `attrs` and `new_attrs`.
        """
        if not new_attrs:
            new_attrs = []
        if isinstance(new_attrs, dict):
            new_attrs = [new_attrs]

        # Merging is implemented on Output objects as OR
        return [(oo | nn) for oo, nn in zip_longest(attrs, new_attrs, fillvalue=Output())]

    @classmethod
    def _ensure_correct_attrs(cls, attrs, identifier):
        """
        Ensure all output attributes are correct.

        Parameters
        ----------
        attrs : list of Output
            List of :py:class:`Output` objects.
        identifier : str
            Identifier of the indicator.

        Returns
        -------
        list
            Same as `attrs`, potentially modified.
        """
        # For single output, var_name defaults to identifier.
        if len(attrs) == 1 and attrs[0].var_name is None:
            attrs[0].var_name = identifier.split(".")[-1]

        # check if we have var_names for everybody
        for i, atts in enumerate(attrs, 1):
            if atts.var_name is None:
                raise ValueError(f"Output #{i} of {identifier} is missing a var_name.")
        return attrs

    def __call__(self, *args, **kwargs):
        """Perform the computation."""
        # Put the variables in `das`, parse them according to the following annotations:
        #     das : dict of variables (required + non-None optionals)
        #     params : dict of parameters (var_kwargs as a single argument, if any)
        #     meta : A dict subclasses can use to store things.

        # Merge *args and **kwargs into a single dict, using the signature, applying default
        ba = self.__signature__.bind(*args, **kwargs)
        ba.apply_defaults()
        kwargs = ba.arguments.copy()

        # Split into das, params, get injected parameters, extracts dsattrs
        das, params, meta = self._parse_arguments(kwargs)
        das, params, meta = self._preprocess_and_checks(das, params, meta)

        # get mappings where keys are the actual compute function's argument names
        args = self._get_compute_args(das, params)
        with np.errstate(divide="ignore", invalid="ignore"), units.context(self.context):
            outs = self.compute(**args)

        if isinstance(outs, DataArray):
            outs = [outs]
        else:  # tuple
            outs = list(outs)
        if len(outs) != self.n_outs:
            raise ValueError(
                f"Indicator {self.identifier} was wrongly defined. Expected {self.n_outs} outputs, got {len(outs)}."
            )

        # Name the outputs and convert to output units
        for i, atts in enumerate(self.attrs):
            outs[i] = outs[i].rename(atts.var_name)
            if atts.units is not None:
                u = {"units": atts.units}
                if atts.units_metadata is not None:
                    u["units_metadata"] = atts.units_metadata
                outs[i] = convert_units_to(outs[i], u, self.context)
                # TODO: Should we remove this ? Priority should be given to CF format, no ?
                outs[i].attrs.update(
                    **u
                )  # Override what convert_units_to does, in case atts.units was not CF compliant

        outs, meta = self._postprocess(outs, das, params, meta)

        return self._finalize(outs, das, params, meta)

    def _parse_arguments(self, kwargs):
        """Extract variable and optional variables from call arguments."""
        # Extract variables + inject injected
        das = {}
        params = kwargs.copy()
        for name, param in self._all_parameters.items():
            if not param.injected:
                # If a variable pop the arg
                if is_percentile_dataarray(params[name]):
                    # duplicate percentiles DA in both das and params
                    das[name] = params[name]
                elif param.kind in [InputKind.VARIABLE, InputKind.OPTIONAL_VARIABLE]:
                    data = params.pop(name)
                    # If a non-optional variable OR not None, store the arg
                    # Optional variable that are none are simply dropped here
                    if param.kind == InputKind.VARIABLE or data is not None:
                        das[name] = data
            else:
                params[name] = param.value

        meta = {}
        return das, params, meta

    def _preprocess_and_checks(
        self, das: dict[str, DataArray], params: dict[str, Any], meta: dict[str, Any]
    ) -> tuple[dict, dict, dict]:
        """
        Preprocessing of the input parameters before calling the compute function.

        Parameters
        ----------
        das : dict
            Dictionary of variable (DataArray) inputs.
        params : dict
            Dictionary of non-variable inputs.
        meta : dict
            Dictionary of other metadata not passed to the compute function.

        Returns
        -------
        dict
            Same as `das`, potentially modified.
        dict
            Same as `params`, potentially modified.
        dict
            Same as `meta`, potentially modified.
        """
        return das, params, meta

    def _get_compute_args(self, das, params) -> dict:
        """Rename variables and parameters to match the compute function's names and split VAR_KEYWORD arguments."""
        # Get correct variable names for the compute function
        # compute_name is empty for param added by the class, exclude them
        args = {}
        for key, param in self._all_parameters.items():
            if param.compute_name is not _empty:
                if key in das:
                    args[param.compute_name] = das[key]
                # elif because some args are in both (percentile DataArrays)
                elif key in params:
                    if param.kind == InputKind.KWARGS:
                        args.update(params[key])
                    else:
                        args[param.compute_name] = params[key]
        return args

    def _postprocess(
        self, outs: list[DataArray], das: dict[str, DataArray], params: dict[str, Any], meta: dict[str, Any]
    ) -> tuple[list[DataArray], dict]:
        """
        Postprocessing of the outputs after calling the compute function.

        Parameters
        ----------
        outs : list
            List of the output DataArrays.
        das : dict
            Dictionary of variable (DataArray) inputs.
        params : dict
            Dictionary of non-variable inputs.
        meta : dict
            Dictionary of other metadata not passed to the compute function.

        Returns
        -------
        list
            Same as `outs`.
        dict
            Same as `meta`, potentially modified.
        """
        return outs, meta

    def _finalize(
        self, outs: list[DataArray], das: dict[str, DataArray], params: dict[str, Any], meta: dict[str, Any]
    ) -> Any:
        """
        Finalize the computation.

        Similar to `_postprocess` but done after and returns a single object, the return of the call.

        Parameters
        ----------
        outs : list
            List of the output DataArrays.
        das : dict
            Dictionary of variable (DataArray) inputs.
        params : dict
            Dictionary of non-variable inputs.
        meta : dict
            Dictionary of other metadata not passed to the compute function.

        Returns
        -------
        Any
            The result from the computation of the indicator.
        """
        # Return a single DataArray in case of single output
        if len(outs) == 1:
            return outs[0]

        # Return a NamedTuple for multiple outputs but not as dataset
        NamedOuts = namedtuple(self.identifier.split(".")[-1], [o.name for o in outs])
        return NamedOuts(*outs)

    @classmethod
    def get_parent_ids(cls):
        """
        Return the list of indicator identifiers this indicator was derived from.

        Returns
        -------
        list
            All parent indicator classes of this indicator. Only classes defining an identifier are included.
        """
        parents = []
        for cl in cls.__bases__:
            if getattr(cl, "identifier", None) is not None:
                parents.append(cl.identifier)
            if hasattr(cl, "get_parent_ids"):
                parents.extend(cl.get_parent_ids())
        return parents


class _DatasetIO(IndicatorBase):
    @classmethod
    def _added_parameters(cls):
        return super()._added_parameters() | {
            "ds": Parameter(
                kind=InputKind.DATASET,
                default=None,
                description="A dataset with the variables given by name.",
                annotation=xarray.Dataset,
            )
        }

    @classmethod
    def _ensure_correct_parameters(cls, parameters):
        # Set default values, otherwise the signature binding chokes
        # on missing arguments when passing only `ds`.
        for name, meta in parameters.items():
            if meta.kind == InputKind.OPTIONAL_VARIABLE:
                meta.default = None
                meta.annotation = DataArray | str
            elif meta.kind == InputKind.VARIABLE:
                meta.default = name
                meta.annotation = DataArray | str
        return super()._ensure_correct_parameters(parameters)

    def _parse_arguments(self, kwargs):
        ds = kwargs.get("ds")

        for name in list(kwargs):
            if (kind := self._all_parameters[name].kind) in [InputKind.VARIABLE, InputKind.OPTIONAL_VARIABLE]:
                val = kwargs[name]
                if isinstance(val, str) and ds is None:
                    raise ValueError(
                        f"Passing variable names as string requires giving the `ds` dataset (got {name}='{val}')"
                    )
                if (isinstance(val, str) or val is None) and ds is not None:
                    # Set default name for DataArray
                    key = val or name

                    if key in ds:
                        kwargs[name] = ds[key]
                    elif kind == InputKind.VARIABLE:
                        raise MissingVariableError(
                            f"For input '{name}', variable '{key}' was not found in the input dataset."
                        )
        das, params, meta = super()._parse_arguments(kwargs)
        if ds is not None:
            meta["dsattrs"] = ds.attrs
        return das, params, meta

    def _finalize(self, outs, das, params, meta):
        if OPTIONS[AS_DATASET]:
            out = Dataset({o.name: o for o in outs})
            if xarray.get_options()["keep_attrs"] is not False:
                out.attrs.update(meta.get("dsattrs", {}))

            out.attrs["history"] = update_history(
                self._history_string(das, params, meta),
                out,
                new_name=self.identifier,
            )
            return out
        return super()._finalize(outs, das, params, meta)

    def _history_string(self, das, params, meta):
        """Return a string for history. It will be prefixed by a timestamp and suffixed by xclim's version."""
        kwargs = {**das}
        for k, v in params.items():
            if self._all_parameters[k].injected:
                continue
            if self._all_parameters[k].kind == InputKind.KWARGS:
                kwargs.update(**v)
            elif self._all_parameters[k].kind != InputKind.DATASET:
                kwargs[k] = v
        return gen_call_string(self.identifier, **kwargs)


class _DataTreeIterator(_DatasetIO):
    @classmethod
    def __added_parameters(cls):
        added = super()._added_parameters()
        added["ds"].description = "A dataset with the variables given by name, or a DataTree of such datasets."
        return added

    def _apply_on_tree_node(self, node: Dataset, *args, **kwargs):
        """Compute this indicator on DataTree node."""
        if not node.data_vars:
            # empty node
            return node
        return self(*args, ds=node, **kwargs)

    def __call__(self, *args, **kwargs):
        if isinstance(kwargs.get("ds"), xarray.DataTree):
            dt = kwargs.pop("ds")
            with set_options(as_dataset=True):
                return dt.map_over_datasets(self._apply_on_tree_node, *args, kwargs=kwargs)
        return super().__call__(*args, **kwargs)


class _MetadataFormatter(_DataTreeIterator):
    """Adds metadata formatting abilities to the indicator."""

    _drop_attrs = ["units", "units_metadata"]
    """Attributes that are never preserved from the input."""

    _free_text_fields = ["long_name", "description", "comment"]
    """Attributes that are free text and will be capitalized."""

    def _postprocess(self, outs, das, params, meta):
        """Actions to done after computing."""
        outs, meta = super()._postprocess(outs, das, params, meta)
        # Metadata attributes from templates
        parent_attrs = {}
        if xarray.get_options()["keep_attrs"] is not False and len(das) == 1:
            parent_attrs = {k: v for k, v in list(das.values())[0].attrs.items() if k not in self._drop_attrs}

        fmtargs = self._get_formatter_args(das | params, meta)
        for out, new_attrs in zip(outs, self.attrs, strict=False):
            out.attrs.update(parent_attrs)
            formatted = self._format_attrs(
                new_attrs,
                fmtargs,
                meta,
            )
            if "cell_methods" in parent_attrs and "cell_methods" in formatted:
                formatted["cell_methods"] = f"{parent_attrs['cell_methods']} {formatted['cell_methods']}"
            out.attrs.update(formatted)

            if "{" in new_attrs.var_name:
                out.name = default_formatter.format(new_attrs.var_name, **fmtargs).replace(" ", "")
        return outs, meta

    def _get_formatter_args(self, args, meta):
        """From all inputs to the call, build a dictionary of all values available for formatting."""
        if args is None:
            args = {k: p.default if not p.injected else p.value for k, p in self._all_parameters.items()}

        mba = {}
        # Add formatting {} around values to be able to replace them with _attrs_mapping using format.
        for k, v in args.items():
            if isinstance(v, units.Quantity):
                mba[k] = f"{v:gcf}"
            elif isinstance(v, int | float):
                mba[k] = f"{v:g}"
            # TODO: What about InputKind.NUMBER_SEQUENCE
            elif k == "indexer":
                if v and v not in [_empty, _empty_default]:
                    dk, dv = v.copy().popitem()
                    if dk == "month":
                        dv = f"m{dv}"
                    elif dk in ("doy_bounds", "date_bounds"):
                        dv = f"{dv[0]} to {dv[1]}"
                    mba["indexer"] = dv
                else:
                    mba["indexer"] = args.get("freq") or "YS"
            elif is_percentile_dataarray(v):
                mba.update(get_percentile_metadata(v, k))
            elif isinstance(v, DataArray):
                mba[k] = "<an array>"
            else:
                mba[k] = v

        for name, param in self._all_parameters.items():
            if name != param.compute_name and param.compute_name is not _empty and name in mba:
                mba[param.compute_name] = mba[name]
        return mba

    def _format_attrs(self, attrs, fmtargs, meta=None, formatter=default_formatter):
        """
        Format attributes with the run-time values of `compute` call parameters.

        If there is only one input and xarray's "keep_attrs" is not False, its attributes
        are copied over before updating.

        Parameters
        ----------
        attrs : Output or dict[str, str]
            The attributes to format and update. All will be formatted except `units` or `units_metadata`,
            which were already handled at computation time.
        fmtargs : dict[str, Any]
            Arguments to the formatter, as given by :py:meth:`MetadataFormatter._get_formatter_args`.
        meta : dict, optional
            A dictionary of things subclasses can populate and use.
        formatter : AttrFormatter
            Plaintext mappings for indicator attributes.

        Returns
        -------
        dict
            Attributes, formatted replaced by call argument values.
        """
        out = {}
        for key, val in attrs.items():
            if key in ["units", "units_metadata"]:
                continue

            if callable(val):
                val = val(**fmtargs)

            out[key] = formatter.format(val, **fmtargs)

            if key in self._free_text_fields:
                out[key] = capitalize_free_text(out[key].strip())
        return out

    def json(self, args: dict | None = None) -> dict:
        """
        Return a serializable dictionary representation of the indicator.

        Parameters
        ----------
        args : mapping, optional
            Arguments as passed to the call method of the indicator.
            If not given, the default arguments will be used when formatting the attributes.

        Returns
        -------
        dict
            A dictionary representation of the indicator.

        Notes
        -----
        This is meant to be used by a third-party library wanting to wrap this indicator into another interface.
        """
        out = {key: getattr(self, key) for key in ["identifier", "title", "abstract", "notes"]}
        out["keywords"] = ", ".join(self.keywords)

        # Format attributes
        fmtargs = self._get_formatter_args(args, {})
        out["outputs"] = [self._format_attrs(attrs, fmtargs) | attrs.meta for attrs in self.attrs]
        out["parameters"] = {k: p.json() for k, p in self._all_parameters.items()}
        return out


class _LocaleMetadataFormatter(_MetadataFormatter):
    """Adds support for translating a few metadata fields."""

    _translatable_attrs = ["long_name", "description", "comment"]
    _translatable_props = ["title", "abstract"]

    def _format_attrs(self, attrs, fmtargs, meta=None, formatter=default_formatter):
        out = super()._format_attrs(attrs, fmtargs, meta, formatter)
        for loc in OPTIONS[METADATA_LOCALES]:  # ty: ignore[not-iterable]
            out.update(
                super()._format_attrs(
                    xloc.get_local_attrs(
                        [self.identifier] + self.get_parent_ids(),
                        locale=loc,
                        var_name=attrs.var_name,
                        names=self._translatable_attrs,
                        append_locale_name=True,
                    ),
                    fmtargs,
                    meta,
                    formatter=xloc.get_local_formatter(loc),
                )
            )
        return out

    def translate(self, locale: str, fill_missing: bool = True) -> dict:
        """Return a dictionary of metadata and (unformatted) output attributes for the requested locale."""
        out = xloc.get_local_attrs(
            [self.identifier] + self.get_parent_ids(),
            locale=locale,
            names=self._translatable_props,
            append_locale_name=False,
        )
        out["attrs"] = [
            xloc.get_local_attrs(
                [self.identifier] + self.get_parent_ids(),
                locale=locale,
                var_name=atts.var_name,
                names=self._translatable_attrs,
                append_locale_name=False,
            )
            for atts in self.attrs
        ]
        if fill_missing:
            for attrs, en_attrs in zip(out["attrs"], self.attrs, strict=True):
                for k, v in en_attrs.items():
                    if k not in attrs and k in self._translatable_attrs:
                        attrs[k] = v
        return out


class _DeprecationWarner(_LocaleMetadataFormatter):
    """Adds possibility to warn about a deprecated indicator."""

    _version_deprecated = ""

    def __call__(self, *args, **kwargs):
        if self._version_deprecated:
            alternative = ""
            if isinstance(self._version_deprecated, tuple):
                vv, other = self._version_deprecated
                alternative = f"Please use {other} instead. "
            else:
                vv = self._version_deprecated
            warnings.warn(
                f"`{self.title}` is deprecated as of `xclim` v{vv} and will be removed in a future release. "
                f"{alternative} See the `xclim` release notes for more information: "
                "https://xclim.readthedocs.io/en/stable/changelog.html",
                FutureWarning,
                stacklevel=3,
            )
        return super().__call__(*args, **kwargs)


class _InputChecker(_DeprecationWarner):
    """Adds some checks on the inputs."""

    src_freq: str | list[str] | None = None
    """The expected frequency of the input data. Can be a list for multiple frequencies, or None if irrelevant."""

    def _preprocess_and_checks(self, das, params, meta):
        """Actions to be done after parsing the arguments and before computing."""
        das, params, meta = super()._preprocess_and_checks(das, params, meta)
        # Pre-computation validation checks on DataArray arguments
        self.datacheck(**das)
        self.cfcheck(**das)

        # Choices
        for name, val in params.items():
            param = self._all_parameters[name]
            if "choices" in param:
                if val not in param.choices:
                    raise ValidationError(
                        f"Parameter {name} received value {val}, which is not among valid values {param.choices}."
                    )
        return das, params, meta

    def cfcheck(self, **das) -> None:
        r"""
        Compare metadata attributes to CF-Convention standards.

        Default cfchecks use the specifications in `xclim.core.VARIABLES`,
        assuming the indicator's inputs are using the CMIP6/xclim variable names correctly.
        Variables absent from these default specs are silently ignored.

        When subclassing this method, use functions decorated using `xclim.core.options.cfcheck`.

        Parameters
        ----------
        **das : dict
            A dictionary of DataArrays to check.
        """
        for varname, vardata in das.items():
            try:
                cfcheck_from_name(varname, vardata)
            except KeyError:
                logging.info("Variable unknown. Ignoring cf check.")
                # Silently ignore unknown variables.
                pass

    def datacheck(self, **das) -> None:
        r"""
        Verify that input data is valid.

        For example, checks could include:
        * assert no precipitation is negative
        * assert no temperature has the same value 5 days in a row

        This base datacheck checks that the input data has a valid sampling frequency, as given in self.src_freq.
        If there are multiple inputs, it also checks if they all have the same frequency and the same anchor.

        Parameters
        ----------
        **das : dict
            A dictionary of DataArrays to check.

        Raises
        ------
        ValidationError
            - if the frequency of any input can't be inferred.
            - if inputs have different frequencies.
            - if inputs have a daily or hourly frequency, but they are not given at the same time of day.
        """
        if self.src_freq is not None:
            for da in das.values():
                if "time" in da.coords and da.time.ndim == 1 and len(da.time) > 3:
                    datachecks.check_freq(da, self.src_freq, strict=True)

            datachecks.check_common_time(
                [da for da in das.values() if "time" in da.coords and da.time.ndim == 1 and len(da.time) > 3]
            )


class _Convenience(_InputChecker):
    """
    Adds pre-processing to the constructor arguments so it can accept some v0 names
    and CF attributes passed by name instead of within `attrs`.
    """

    _cf_names: list[str] = [
        "var_name",
        "standard_name",
        "long_name",
        "units",
        "units_metadata",
        "cell_methods",
        "description",
        "comment",
    ]
    """Attribute names that can be passed directly to the constructor."""

    def __new__(cls, **kwargs):
        if "cf_attrs" in kwargs:
            warnings.warn(
                "Indicator argument `cf_attrs` has been renamed to `attrs` in xclim v1.", FutureWarning, stacklevel=2
            )
            kwargs["attrs"] = kwargs.pop("cf_attrs")

        attrs = kwargs.pop("attrs", None) or []
        passed = {}
        for name in cls._cf_names:
            if vals := kwargs.pop(name, None):
                passed[name] = vals
        if passed:
            n = len(attrs)
            if n == 0:
                n = max(len(vals) if isinstance(vals, (list, tuple)) else 1 for vals in passed.values())
                attrs = [{} for i in range(n)]
            for name, vals in passed.items():
                if not isinstance(vals, (list, tuple)):
                    vals = [vals] * n
                if len(vals) != len(attrs):
                    raise ValueError(f"Attribute {name} has {len(vals)} elements but {len(attrs)} were expected.")
                for atts, val in zip(attrs, vals, strict=True):
                    atts[name] = val
        kwargs["attrs"] = attrs

        module = kwargs.get("module", cls.__module__)
        # Infer realm for built-in xclim instances, handle module
        xclim_realm = None
        # If the module is xclim.indicators.XYZ, we assume an "official" indicator, realm is XYZ, module set
        # otherwise everything is up to the caller
        if module.startswith("xclim.indicators."):
            xclim_realm = module.split(".")[2]
            kwargs["module"] = module
        # Priority given to passed realm -> parent's realm -> location of class declaration (xclim indicators only)
        kwargs.setdefault("realm", cls.realm or xclim_realm)
        return super().__new__(cls, **kwargs)

    def __getattr__(self, attr):
        """Return the attribute."""
        if attr in self._cf_names:
            out = [attrs.get(attr, attrs.meta.get(attr, "")) for attrs in self.attrs]
            if len(out) == 1:
                return out[0]
            return out
        raise AttributeError(attr)


class _Registrer(_Convenience):
    """Register the indicator in the xclim registry."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.identifier in registry:
            warnings.warn(f"Indicator {self.identifier} already exists and will be overwritten.", stacklevel=4)
        registry[self.identifier] = self


class Indicator(_Registrer):  # numpydoc ignore=PR01
    r"""
    Climate indicator base class.

    Climate indicator object that, when called, computes an indicator and assigns its output a number of
    CF-compliant attributes. These attributes can be *templated*, allowing metadata to reflect
    the value of call arguments.

    Instantiating a new indicator returns an instance but also registers is
    in :py:data:`xclim.core.indicator.registry`.

    Attributes in `Indicator.attrs` will be formatted and added to the output variable(s).
    This attribute is a list of :py:class:`Output` dict-like objects.

    A lot of the Indicator's metadata is parsed from the underlying `compute` function's
    docstring and signature. Input variables and parameters are listed in
    :py:attr:`xclim.core.indicator.Indicator.parameters`, while parameters that will be
    injected in the compute function are in  :py:attr:`xclim.core.indicator.Indicator.injected_parameters`.

    Compared to their base `compute` function, indicators add the possibility of using a dataset
    or a :py:class:`xarray.DataTree` as input, with the added argument `ds` in the call signature.
    All arguments that were indicated by the compute function to be variables (DataArrays) through
    annotations will be promoted to also accept strings that correspond to variable names
    in the `ds` dataset (or on each DataTree nodes). Also, indicators return Datasets by default,
    while compute functions return one or multiple DataArrays.
    """

    def __init__(
        self,
        identifier: str,
        compute: Callable = None,
        title: str = None,
        abstract: str = None,
        realm: str = None,
        keywords: list[str] = None,
        references: str = None,
        notes: str = None,
        input: dict = None,
        parameters: dict = None,
        attrs: dict = None,
        context: str = "none",
        src_freq: str | list[str] = None,
        **attrs_kwargs,
    ):
        """
        Create a new indicator.

        Parameters
        ----------
        identifier : str
            Unique ID for this indicator. Single-output indicator will use this as their output variable
            name if no `var_name`is passed to the first element of `attrs`.
            All indicators are registered to :py:data:`xclim.core.indicator.registry`, which is case-insensitive.
        compute : func
            The function computing the indicators. It should return one or more DataArray.
            Metadata will first be parsed from it as much as possible.
        title : str, optional
            A succinct description of what is in the computed outputs.
            Parsed from `compute` docstring if None (first paragraph).
        abstract : str, optional
            A long description of what is in the computed outputs.
            Parsed from `compute` docstring if None (second paragraph).
        realm : {'atmos', 'convert', 'seaIce', 'land', 'ocean'}, optional
            General domain of validity of the indicator.
        keywords : list of strings, optional
            Keywords.
        references : str, optional
            Published or web-based references that describe the data or methods used to produce it.
            Parsed from `compute` docstring if None (from the "References" section).
        notes : str, optional
            Notes regarding computing function, for example the mathematical formulation.
            Parsed from `compute` docstring if None (form the "Notes" section).
        input : dict, optional
            Mapping from input variable name in the compute function to known variable name.
            Useful for transforming generic compute functions into variable-specific indicator.
            The new variables names must be defined in :py:data:`xclim.core.VARIABLES`.
        parameters: dict, optional
            Overrides for the parameters. Either value to "inject", removing that parameter from the call signature,
            or dictionaries of properties to override the ones parsed from the docstring.
            See :py:class:`~xclim.core.indicator.Parameter` for valid properties. Additionally,
            `name` can be passed to change the name of the argument in the call signature.
        attrs : list of dict
            Attributes to be formatted and added to the computation's output.
            Any attribute are accepted, but `var_name` is required for multi-output indicators.
            The list must be the same length as the number of outputs of the compute function.
        context : str
            A `pint` unit context enabled during the computation of this indicator.
            For example use 'hydro' to allow conversion from 'kg m-2 s-1' to 'mm/day' for all inputs an outputs.
        src_freq : str or sequence of str, optional
            The expected frequency of the input data. Can be a list for multiple frequencies, or None if irrelevant.
        **attrs_kwargs
            For convenience, output attributes can also be passed by name to the constructor.
        """  # numpydoc ignore=PR01,PR02
        super().__init__(
            identifier=identifier,
            compute=compute,
            title=title,
            abstract=abstract,
            realm=realm,
            keywords=keywords,
            references=references,
            notes=notes,
            input=input or {},
            parameters=parameters or {},
            attrs=attrs or {},
            context=context,
            src_freq=src_freq,
            **attrs_kwargs,
        )


class CheckMissingIndicator(Indicator):  # numpydoc ignore=PR01,PR02 # pylint: disable=too-many-ancestors
    r"""
    Class adding missing value checks to indicators.

    This should not be used as-is, but subclassed by implementing the `_get_missing_freq` method.
    This method will be called in `_postprocess` using the compute parameters as only argument.
    It should return a freq string, the same as the output freq of the computed data.
    It can also be "None" to indicator the full time axis has been reduced, or "False" to skip the missing checks.
    """

    missing: str = "from_context"
    """
    The name of the missing value method. See `xclim.core.missing.MissingBase` to create new custom methods.
    If None, this will be determined by the global configuration (see `xclim.set_options`).
    """
    missing_options: dict = None
    """
    Arguments to pass to the `missing` function.
    If None, this will be determined by the global configuration.
    """

    def __init__(self, **kwds):
        if self.missing == "from_context" and self.missing_options is not None:
            raise ValueError("Cannot set `missing_options` with `missing` method being from context.")
        super().__init__(**kwds)

    def _extra_doc(self):
        extra = super()._extra_doc()
        extra.append(f'This indicator will check for missing values according to the method "{self.missing}".')
        return extra

    def _history_string(self, das, params, meta):
        if self.missing == "from_context":
            missing = OPTIONS[CHECK_MISSING]
        else:
            missing = self.missing
        opt_str = f" with options check_missing={missing}"

        if missing != "skip":
            mopts = self.missing_options or OPTIONS[MISSING_OPTIONS].get(missing).copy()
            if mopts.get("subfreq", "absent") is None:
                mopts.pop("subfreq")  # impertinent default
            if mopts:
                opt_str += f", missing_options={mopts}"

        return super()._history_string(das, params, meta) + opt_str

    def _get_missing_freq(self, params):
        """Return the resampling frequency to be used in the missing values check."""
        raise NotImplementedError("Don't use `CheckMissingIndicator` directly.")

    def _postprocess(self, outs, das, params, meta):
        """Masking of missing values."""
        outs, meta = super()._postprocess(outs, das, params, meta)

        freq = self._get_missing_freq(params)
        method = self.missing if self.missing != "from_context" else OPTIONS[CHECK_MISSING]
        if method != "skip" and freq is not False:
            # Mask results that do not meet criteria defined by the `missing` method.
            # This means all outputs must have the same dimensions as the broadcasted inputs (excluding time)
            options = self.missing_options or OPTIONS[MISSING_OPTIONS].get(method, {})
            misser = MISSING_METHODS[method](**options)

            # We flag periods according to the missing method. skip variables without a time coordinate.
            src_freq = self.src_freq if isinstance(self.src_freq, str) else None
            miss = (
                misser(da, freq, src_freq, **params.get("indexer", {})) for da in das.values() if "time" in da.coords
            )
            # Reduce by or and broadcast to ensure the same length in time
            # When indexing is used and there are no valid points in the last period, mask will not include it
            mask = reduce(np.logical_or, miss)
            if isinstance(mask, DataArray):  # mask might be a bool in some cases
                if "time" in mask.dims and mask.time.size < outs[0].time.size:
                    mask = mask.reindex(time=outs[0].time, fill_value=True)
                # Remove any aux coord to avoid any unwanted dask computation in the alignment within "where"
                mask, _ = split_auxiliary_coordinates(mask)
            outs = [out.where(~mask) for out in outs]

        return outs, meta


class ReducingIndicator(CheckMissingIndicator):  # numpydoc ignore=PR01,PR02 # pylint: disable=too-many-ancestors
    """Indicator that performs a time-reducing computation."""

    def _get_missing_freq(self, params):
        """Return None, to indicate that the full time axis is to be reduced."""
        return None


class ResamplingIndicator(CheckMissingIndicator):  # numpydoc ignore=PR02 # pylint: disable=too-many-ancestors
    """
    Indicator that performs a resampling computation.

    Compared to the base Indicator, this adds the handling of missing data,
    and the check of allowed periods.
    """

    allowed_periods: list[str] = None
    """
    A list of allowed periods, i.e. base parts of the `freq` parameter.
    For example, indicators meant to be computed annually only will have `allowed_periods=["Y"]`.
    `None` means "any period" or that the indicator doesn't take a `freq` argument.
    """

    @classmethod
    def _ensure_correct_parameters(cls, parameters):
        if "freq" not in parameters:
            raise ValueError(
                "ResamplingIndicator require a 'freq' argument, use the base Indicator"
                " class if your computation doesn't perform any resampling."
            )
        return super()._ensure_correct_parameters(parameters)

    def _extra_doc(self):
        extra = super()._extra_doc()
        if not self._all_parameters["freq"].injected and self.allowed_periods is not None:
            extra.append(f"Requested resampling periods are restricted to {', '.join(self.allowed_periods)}")
        return extra

    def _get_missing_freq(self, params):
        return params["freq"]

    def _preprocess_and_checks(self, das, params, meta):
        """Perform parent's checks and also check if freq is allowed."""
        das, params, meta = super()._preprocess_and_checks(das, params, meta)

        # Check if the period is allowed:
        if self.allowed_periods is not None:
            if parse_offset(params["freq"])[1] not in self.allowed_periods:
                raise ValueError(
                    f"Resampling frequency {params['freq']} is not allowed for indicator "
                    f"{self.identifier} (needs something equivalent to one of {self.allowed_periods})."
                )

        return das, params, meta


class IndexingIndicator(Indicator):
    """Indicator that also adds the "indexer" kwargs to subset the inputs before computation."""

    @classmethod
    def _added_parameters(cls):
        """Create a list of tuples for arguments to add (name, Parameter)."""
        return super()._added_parameters() | {
            "indexer": Parameter(
                kind=InputKind.KWARGS,
                description=(
                    "Indexing parameters to compute the indicator on a temporal "
                    "subset of the data. It accepts the same arguments as "
                    ":py:func:`xclim.core.calendar.select_time`."
                ),
            )
        }

    def _preprocess_and_checks(self, das, params, meta):
        """Perform parent's checks and also check if freq is allowed."""
        das, params, meta = super()._preprocess_and_checks(das, params, meta)

        indxr = params.get("indexer")
        if indxr:
            for k, da in filter(lambda kda: "time" in kda[1].coords, das.items()):
                das.update({k: select_time(da, **indxr)})
        return das, params, meta


class ResamplingIndicatorWithIndexing(ResamplingIndicator, IndexingIndicator):
    """Resampling indicator that also adds "indexer" kwargs to subset the inputs before computation."""


class Daily(ResamplingIndicator):
    """Class for daily inputs and resampling computes."""

    src_freq = "D"


class Hourly(ResamplingIndicator):
    """Class for hourly inputs and resampling computes."""

    src_freq = "h"


class StandardizedIndexes(ResamplingIndicator):
    """Resampling but flexible inputs indicators."""

    src_freq = ["D", "MS"]
    context = "hydro"


base_registry["Indicator"] = Indicator
base_registry["ReducingIndicator"] = ReducingIndicator
base_registry["IndexingIndicator"] = IndexingIndicator
base_registry["ResamplingIndicator"] = ResamplingIndicator
base_registry["ResamplingIndicatorWithIndexing"] = ResamplingIndicatorWithIndexing
base_registry["Hourly"] = Hourly
base_registry["Daily"] = Daily

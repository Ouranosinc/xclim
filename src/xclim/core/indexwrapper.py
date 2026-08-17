"""Tools for parsing index-like functions and extract metadata."""

from __future__ import annotations

import logging
import re
from ast import literal_eval
from copy import deepcopy
from dataclasses import dataclass
from inspect import _empty as _empty_default
from inspect import signature
from typing import Any

from docstring_parser import parse_from_object

from xclim.core import InputKind, infer_kind_from_parameter


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
                out[field] = getattr(self, field)
        return out


@dataclass
class IndexOutput:
    """Representation of an element of the "Returns" section of an index-like compute function."""

    type: str
    dimensionality: str
    description: str
    name: str | None = None

    @classmethod
    def from_docstring_parser(cls, obj):
        """
        Initialize from a :py:class:`docstring_parser.common.DocstringReturns` object.

        Parameters
        ----------
        obj : docstring_parser.common.DocstringReturns`
          As parsed from the numpydoc style Returns section.

        Returns
        -------
        IndexOutput
            Object initialized from parsing the docstring.
        """
        if ", " in obj.type_name:
            type, dimensionality = obj.type_name.split(", ")
        else:
            type = obj.type_name
            dimensionality = None
        return cls(name=obj.return_name, type=type, dimensionality=dimensionality, description=obj.description)


@dataclass
class IndexMeta:
    """
    Metadata of an index-like compute function by parsing its signature, docstring and declared units.

    This is meant as a temporary container between the compute function and the indicator class.
    """

    title: str  # First line of the docstring
    abstract: str  # Second paragraph of the docstring

    inputs: dict[str, Parameter]  # Parameter section
    outputs: list[IndexOutput]  # Returns section

    # Other sections of the docstring
    notes: str
    references: str

    @classmethod
    def parse(cls, func):
        """
        Initialize from an index-like compute function.

        Parameters
        ----------
        func : callable
          A function annotated, documentated and wrapped by :py:func:`xclim.core.units.declare_units`
          as explained in :ref:`notebooks/extendxclim:Defining new index-like compute functions`.

        Returns
        -------
        IndexMeta
          Object initialized from parsing the function.
        """
        doc = parse_from_object(func)
        sig = signature(func)
        declared_units = getattr(func, "in_units", {})

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
            raise ValueError(f"Malformed docstring on {func} : the parameters {missing} are absent from the signature.")

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
                        logging.error(
                            f"Choices defined in the description of parameter {name}"
                            f" of function {func} could not be parsed. "
                            f"Got: {choices_raw}."
                        )
                        # If the literal_eval fails, we just ignore the choices.
                        pass

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
        outputs = [IndexOutput.from_docstring_parser(r) for r in doc.many_returns]

        notes = ""
        references = ""
        for section in doc.meta:
            if section.args == ["notes"]:
                notes = section.description
            elif section.args == ["references"]:
                references = section.description

        return cls(title=title, abstract=abstract, inputs=inputs, outputs=outputs, notes=notes, references=references)

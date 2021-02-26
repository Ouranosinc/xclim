# -*- coding: utf-8 -*-
# noqa: D205,D400
"""
Formatting utilities for indicators
===================================
"""
import datetime as dt
import re
import string
from ast import literal_eval
from fnmatch import fnmatch
from typing import Dict, Mapping, Optional, Sequence, Union

import xarray as xr

from .utils import InputKind


class AttrFormatter(string.Formatter):
    """A formatter for frequently used attribute values.

    See the doc of format_field() for more details.
    """

    def __init__(
        self,
        mapping: Mapping[str, Sequence[str]],
        modifiers: Sequence[str],
    ) -> None:
        """Initialize the formatter.

        Parameters
        ----------
        mapping : Mapping[str, Sequence[str]]
            A mapping from values to their possible variations.
        modifiers : Sequence[str]
            The list of modifiers, must be the as long as the longest value of `mapping`.
        """
        super().__init__()
        self.modifiers = modifiers
        self.mapping = mapping

    def format_field(self, value, format_spec):
        """Format a value given a formatting spec.

        If `format_spec` is in this Formatter's modifiers, the corresponding variation
        of value is given. If `format_spec` is not specified but `value` is in the
        mapping, the first variation is returned.

        Examples
        --------
        Let's say the string "The dog is {adj1}, the goose is {adj2}" is to be translated
        to french and that we know that possible values of `adj` are `nice` and `evil`.
        In french, the genre of the noun changes the adjective (cat = chat is masculine,
        and goose = oie is feminine) so we initialize the formatter as:

        >>> fmt = AttrFormatter({'nice': ['beau', 'belle'], 'evil' : ['méchant', 'méchante']}, ['m', 'f'])
        >>> fmt.format("Le chien est {adj1:m}, l'oie est {adj2:f}", adj1='nice', adj2='evil')
        "Le chien est beau, l'oie est méchante"

        The base values may be given using unix shell-like patterns:

        >>> fmt = AttrFormatter({'AS-*': ['annuel', 'annuelle'], 'MS' : ['mensuel', 'mensuelle']}, ['m', 'f'])
        >>> fmt.format("La moyenne {freq:f} est faite sur un échantillon {src_timestep:m}", freq='AS-JUL', src_timestep='MS')
        'La moyenne annuelle est faite sur un échantillon mensuel'
        """
        baseval = self._match_value(value)
        if baseval is not None and not format_spec:
            return self.mapping[baseval][0]

        if format_spec in self.modifiers:
            if baseval is not None:
                return self.mapping[baseval][self.modifiers.index(format_spec)]
            raise ValueError(
                f"No known mapping for string '{value}' with modifier '{format_spec}'"
            )
        return super().format_field(value, format_spec)

    def _match_value(self, value):
        if isinstance(value, str):
            for mapval in self.mapping.keys():
                if fnmatch(value, mapval):
                    return mapval
        return None


# Tag mappings between keyword arguments and long-form text.
default_formatter = AttrFormatter(
    {
        "YS": ["annual", "years"],
        "AS-*": ["annual", "years"],
        "MS": ["monthly", "months"],
        "QS-*": ["seasonal", "seasons"],
        "DJF": ["winter"],
        "MAM": ["spring"],
        "JJA": ["summer"],
        "SON": ["fall"],
        "norm": ["Normal"],
        "m1": ["january"],
        "m2": ["february"],
        "m3": ["march"],
        "m4": ["april"],
        "m5": ["may"],
        "m6": ["june"],
        "m7": ["july"],
        "m8": ["august"],
        "m9": ["september"],
        "m10": ["october"],
        "m11": ["november"],
        "m12": ["december"],
    },
    ["adj", "noun"],
)


def parse_doc(doc: str) -> Dict[str, str]:
    """Crude regex parsing reading an indice docstring and extracting information needed in indicator construction.

    The appropriate docstring syntax is detailed in :ref:`Defining new indices`.

    Parameters
    ----------
    doc : str
      The docstring of an indice function.

    Returns
    -------
    dict
      A dictionary with all parsed sections.
    """
    if doc is None:
        return dict()

    out = dict()

    sections = re.split(r"(\w+\s?\w+)\n\s+-{3,50}", doc)  # obj.__doc__.split('\n\n')
    intro = sections.pop(0)
    if intro:
        intro_content = list(map(str.strip, intro.strip().split("\n\n")))
        if len(intro_content) == 1:
            out["title"] = intro_content[0]
        elif len(intro_content) >= 2:
            out["title"], abstract = intro_content[:2]
            out["abstract"] = " ".join(map(str.strip, abstract.splitlines()))

    for i in range(0, len(sections), 2):
        header, content = sections[i : i + 2]

        if header in ["Notes", "References"]:
            out[header.lower()] = content.replace("\n    ", "\n").strip()
        elif header == "Parameters":
            out["parameters"] = _parse_parameters(content)
        elif header == "Returns":
            rets = _parse_returns(content)
            if rets:
                meta = list(rets.values())[0]
                if "long_name" in meta:
                    out["long_name"] = meta["long_name"]
    return out


def _parse_parameters(section):
    """Parse the parameters section of a docstring into a dictionary
    mapping the parameter name to its description and, potentially, to its set of choices.

    The type annotation are not parsed, except for fixed sets of values
    (listed as "{'a', 'b', 'c'}"). The annotation parsing only accepts
    strings, numbers, `None` and `nan` (to represent `numpy.nan`).
    """
    curr_key = None
    params = {}
    for line in section.split("\n"):
        if line.startswith(" " * 6):  # description
            s = " " if params[curr_key]["description"] else ""
            params[curr_key]["description"] += s + line.strip()
        elif line.startswith(" " * 4) and ":" in line:  # param title
            name, annot = line.split(":", maxsplit=1)
            curr_key = name.strip()
            params[curr_key] = {"description": ""}
            match = re.search(r".*(\{.*\}).*", annot)
            if match:
                try:
                    choices = literal_eval(match.groups()[0])
                    params[curr_key]["choices"] = choices
                except ValueError:
                    pass
    return params


def _parse_returns(section):
    """Parse the returns section of a docstring into a dictionary mapping the parameter name to its description."""
    curr_key = None
    params = {}
    for line in section.split("\n"):
        if line.strip():
            if line.startswith(" " * 6):  # long_name
                s = " " if params[curr_key]["long_name"] else ""
                params[curr_key]["long_name"] += s + line.strip()
            elif line.startswith(" " * 4):  # param title
                annot, *name = reversed(line.split(":", maxsplit=1))
                if name:
                    curr_key = name[0].strip()
                else:
                    curr_key = None
                params[curr_key] = {"long_name": ""}
                annot, *unit = annot.split(",", maxsplit=1)
                if unit:
                    params[curr_key]["units"] = unit[0].strip()
    return params


def merge_attributes(
    attribute: str,
    *inputs_list: Union[xr.DataArray, xr.Dataset],
    new_line: str = "\n",
    missing_str: Optional[str] = None,
    **inputs_kws: Union[xr.DataArray, xr.Dataset],
):
    r"""
    Merge attributes from several DataArrays or Datasets.

    If more than one input is given, its name (if available) is prepended as: "<input name> : <input attribute>".

    Parameters
    ----------
    attribute : str
      The attribute to merge.
    inputs_list : Union[xr.DataArray, xr.Dataset]
      The datasets or variables that were used to produce the new object. Inputs given that way will be prefixed by their `name` attribute if available.
    new_line : str
      The character to put between each instance of the attributes. Usually, in CF-conventions,
      the history attributes uses '\\n' while cell_methods uses ' '.
    missing_str : str
      A string that is printed if an input doesn't have the attribute. Defaults to None, in which
      case the input is simply skipped.
    inputs_kws : Union[xr.DataArray, xr.Dataset]
      Mapping from names to the datasets or variables that were used to produce the new object.
      Inputs given that way will be prefixes by the passed name.

    Returns
    -------
    str
      The new attribute made from the combination of the ones from all the inputs.
    """
    inputs = []
    for in_ds in inputs_list:
        inputs.append((getattr(in_ds, "name", None), in_ds))
    inputs += list(inputs_kws.items())

    merged_attr = ""
    for in_name, in_ds in inputs:
        if attribute in in_ds.attrs or missing_str is not None:
            if in_name is not None and len(inputs) > 1:
                merged_attr += f"{in_name}: "
            merged_attr += in_ds.attrs.get(
                attribute, "" if in_name is None else missing_str
            )
            merged_attr += new_line

    if len(new_line) > 0:
        return merged_attr[: -len(new_line)]  # Remove the last added new_line
    return merged_attr


def update_history(
    hist_str: str,
    *inputs_list: Union[xr.DataArray, xr.Dataset],
    new_name: Optional[str] = None,
    **inputs_kws: Union[xr.DataArray, xr.Dataset],
):
    """Return an history string with the timestamped message and the combination of the history of all inputs.

    The new history entry is formatted as "[<timestamp>] <new_name>: <hist_str> - xclim version: <xclim.__version__>."

    Parameters
    ----------
    hist_str : str
      The string describing what has been done on the data.
    new_name : Optional[str]
      The name of the newly created variable or dataset to prefix hist_msg.
    *inputs_list : Union[xr.DataArray, xr.Dataset]
      The datasets or variables that were used to produce the new object.
      Inputs given that way will be prefixed by their "name" attribute if available.
    **inputs_kws : Union[xr.DataArray, xr.Dataset]
      Mapping from names to the datasets or variables that were used to produce the new object.
      Inputs given that way will be prefixes by the passed name.

    Returns
    -------
    str
      The combine history of all inputs starting with `hist_str`.

    See Also
    --------
    merge_attributes
    """
    from xclim import __version__  # pylint: disable=cyclic-import

    merged_history = merge_attributes(
        "history",
        *inputs_list,
        new_line="\n",
        missing_str="",
        **inputs_kws,
    )
    if len(merged_history) > 0 and not merged_history.endswith("\n"):
        merged_history += "\n"
    merged_history += f"[{dt.datetime.now():%Y-%m-%d %H:%M:%S}] {new_name or ''}: {hist_str} - xclim version: {__version__}."
    return merged_history


def prefix_attrs(source, keys, prefix):
    """Rename some of the keys of a dictionary by adding a prefix.

    Parameters
    ----------
    source : dict
      Source dictionary, for example data attributes.
    keys : sequence
      Names of keys to prefix.
    prefix : str
      Prefix to prepend to keys.

    Returns
    -------
    dict
      Dictionary of attributes with some keys prefixed.
    """
    out = {}
    for key, val in source.items():
        if key in keys:
            out[f"{prefix}{key}"] = val
        else:
            out[key] = val
    return out


def unprefix_attrs(source, keys, prefix):
    """Remove prefix from keys in a dictionary.

    Parameters
    ----------
    source : dict
      Source dictionary, for example data attributes.
    keys : sequence
      Names of original keys for which prefix should be removed.
    prefix : str
      Prefix to remove from keys.

    Returns
    -------
    dict
      Dictionary of attributes whose keys were prefixed, with prefix removed.
    """
    out = {}
    n = len(prefix)
    for key, val in source.items():
        k = key[n:]
        if (k in keys) and key.startswith(prefix):
            out[k] = val
        elif key not in out:
            out[key] = val
    return out


KIND_ANNOTATION = {
    InputKind.VARIABLE: "str or DataArray",
    InputKind.OPTIONAL_VARIABLE: "str or DataArray, optional",
    InputKind.QUANTITY_STR: "quantity (string with units)",
    InputKind.FREQ_STR: "offset alias (string)",
    InputKind.NUMBER: "number",
    InputKind.NUMBER_SEQUENCE: "number or sequence of numbers",
    InputKind.STRING: "str",
    InputKind.DAY_OF_YEAR: "date (string, MM-DD)",
    InputKind.DATE: "date (sting, YYYY-MM-DD)",
    InputKind.DATASET: "Dataset, optional",
    InputKind.KWARGS: "",
    InputKind.OTHER_PARAMETER: "Any",
}


def _gen_parameters_section(names, parameters):
    """Generate the "parameters" section of the indicator docstring.

    Parameters
    ----------
    names : Sequence[str]
      Names of the input parameters, in order. Usually `Ind._parameters`.
    parameters : Mapping[str, Any]
      Parameters dictionary. Usually `Ind.parameters`, As this is missing `ds`, it is added explicitly.
    """
    section = "Parameters\n----------\n"
    for name in names:
        if name == "ds":
            descstr = "Input dataset."
            defstr = "Default: None."
            unitstr = ""
            annotstr = "Dataset, optional"
        else:
            param = parameters[name]
            descstr = param["description"]
            if param["kind"] == InputKind.VARIABLE:
                defstr = f"Default : `ds.{param['default']}`. "
            elif param["kind"] == InputKind.OPTIONAL_VARIABLE:
                defstr = ""
            else:
                defstr = f"Default : {param['default']}. "
            if "choices" in param:
                annotstr = str(param["choices"])
            else:
                annotstr = KIND_ANNOTATION[param["kind"]]
            if param.get("units", False):
                unitstr = f"[Required units : {param['units']}]"
            else:
                unitstr = ""
        section += f"{name} : {annotstr}\n  {descstr}\n  {defstr}{unitstr}\n"
    return section


def _gen_returns_section(cfattrs):
    """Generate the "Returns" section of an indicator's docstring.

    Parameters
    ----------
    cfattrs : Sequence[Dict[str, Any]]
      The list of cf attributes, usually Indicator.cf_attrs.
    """
    section = "Returns\n-------\n"
    for attrs in cfattrs:
        section += f"{attrs['var_name']} : DataArray\n"
        section += f"  {attrs.get('long_name', '')}"
        if "standard_name" in attrs:
            section += f" ({attrs['standard_name']})"
        if "units" in attrs:
            section += f" [{attrs['units']}]"
        section += "\n"
        for key, attr in attrs.items():
            if key not in ["long_name", "standard_name", "units", "var_name"]:
                if callable(attr):
                    attr = "<Dynamically generated string>"
                section += f"  {key}: {attr}\n"
    return section


def generate_indicator_docstring(kwds):
    """Generate an indicator's docstring from keywords.

    Parameters
    ----------
    kwds : dict
      The dict of all class attributes and init keywords as generated in the indicator's __new__ method.
      It should have at least:
        "compute", "_parameters", "parameters", "cf_attrs".
    """
    header = f"{kwds.get('title','')} (realm: {kwds.get('realm')})\n\n{kwds.get('abstract', '')}\n"

    special = f"This indicator will check for missing values according to the method \"{kwds.get('missing', 'from_context')}\".\n"
    if hasattr(kwds["compute"], "__func__") and hasattr(
        kwds["compute"].__func__, "__module__"
    ):
        special += f"Based on indice :py:func:`{kwds['compute'].__func__.__module__}.{kwds['compute'].__func__.__name__}`.\n"
    if "keywords" in kwds:
        special += f"Keywords : {kwds['keywords']}.\n"

    parameters = _gen_parameters_section(kwds["_parameters"], kwds["parameters"])

    returns = _gen_returns_section(kwds["cf_attrs"])

    extras = ""
    for section in ["notes", "references"]:
        if section in kwds:
            extras += (
                f"{section.capitalize()}\n{'-' * len(section)}\n{kwds[section]}\n\n"
            )

    doc = f"{header}\n{special}\n{parameters}\n{returns}\n{extras}"
    return doc

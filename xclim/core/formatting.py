# -*- coding: utf-8 -*-
# noqa: D205,D400
"""
Formatting utilities for indicators
===================================
"""
import datetime as dt
import re
import string
from fnmatch import fnmatch
from typing import Mapping, Optional, Sequence, Union

import xarray as xr


class AttrFormatter(string.Formatter):
    """A formatter for frequently used attribute values.

    See the doc of format_field() for more details.
    """

    def __init__(
        self,
        mapping: Mapping[str, Sequence[str]],
        modifiers: Sequence[str],
    ):
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


def parse_doc(doc):
    """Crude regex parsing."""
    if doc is None:
        return {}

    out = {}

    sections = re.split(r"(\w+\s?\w+)\n\s+-{3,50}", doc)  # obj.__doc__.split('\n\n')
    intro = sections.pop(0)
    if intro:
        content = list(map(str.strip, intro.strip().split("\n\n")))
        if len(content) == 1:
            out["title"] = content[0]
        elif len(content) >= 2:
            out["title"], abstract = content[:2]
            out["abstract"] = " ".join(map(str.strip, abstract.splitlines()))

    for i in range(0, len(sections), 2):
        header, content = sections[i : i + 2]

        if header in ["Notes", "References"]:
            out[header.lower()] = content.replace("\n    ", "\n").strip()
        elif header == "Parameters":
            out["parameters"] = _parse_parameters(content)
        elif header == "Returns":
            match = re.search(r"xarray\.DataArray\s*(.*)", content)
            if match:
                out["long_name"] = match.groups()[0]

    return out


def _parse_parameters(section):
    """Parse the parameters section of a docstring into a dictionary mapping the parameter name to its description."""
    curr_key = None
    params = {}
    for line in section.split("\n"):
        if line.strip():
            if line.startswith(" " * 6):  # description
                s = " " if params[curr_key]["description"] else ""
                params[curr_key]["description"] += s + line.strip()
            elif line.startswith(" " * 4):  # param title
                name, annot = line.split(":", maxsplit=1)
                curr_key = name.strip()
                params[curr_key] = {"description": ""}
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

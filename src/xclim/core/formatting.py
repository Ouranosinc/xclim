"""
Formatting Utilities for Indicators
===================================
"""

from __future__ import annotations

import datetime as dt
import itertools
import re
import string
import textwrap
import warnings
from ast import literal_eval
from collections.abc import Callable, Sequence
from fnmatch import fnmatch
from inspect import _empty, signature  # noqa
from typing import Any

import pandas as pd
import xarray as xr
from boltons.funcutils import wraps

from xclim.core.utils import InputKind

DEFAULT_FORMAT_PARAMS = {
    "tasmin_per_thresh": "{unknown}",
    "tasmin_per_window": "{unknown}",
    "tasmin_per_period": "{unknown}",
    "tas_per_thresh": "{unknown}",
    "tas_per_window": "{unknown}",
    "tas_per_period": "{unknown}",
    "tasmax_per_thresh": "{unknown}",
    "tasmax_per_window": "{unknown}",
    "tasmax_per_period": "{unknown}",
    "pr_per_thresh": "{unknown}",
    "pr_per_window": "{unknown}",
    "pr_per_period": "{unknown}",
}


class AttrFormatter(string.Formatter):
    """
    A formatter for frequently used attribute values.

    Parameters
    ----------
    mapping : dict of str, sequence of str
        A mapping from values to their possible variations.
    modifiers : sequence of str
        The list of modifiers.
        Must at least match the length of the longest value of `mapping`.
        Cannot include reserved modifier 'r'.

    Notes
    -----
    See the doc of :py:meth:`format_field` for more details.
    """

    def __init__(
        self,
        mapping: dict[str, Sequence[str]],
        modifiers: Sequence[str],
    ) -> None:
        """
        Initialize the formatter.

        Parameters
        ----------
        mapping : dict[str, Sequence[str]]
            A mapping from values to their possible variations.
        modifiers : Sequence[str]
            The list of modifiers.
            Must at least match the length of the longest value of `mapping`.
            Cannot include reserved modifier 'r'.
        """
        super().__init__()
        if "r" in modifiers:
            raise ValueError("Modifier 'r' is reserved for default raw formatting.")
        self.modifiers = modifiers
        self.mapping = mapping

    def format(self, format_string: str, /, *args, **kwargs: dict) -> str:
        r"""
        Format a string.

        Parameters
        ----------
        format_string : str
            The string to format.
        *args : Any
            Arguments to format.
        **kwargs : dict
            Keyword arguments to format.

        Returns
        -------
        str
            The formatted string.
        """
        for k, v in DEFAULT_FORMAT_PARAMS.items():
            if k not in kwargs:
                kwargs.update({k: v})
        return super().format(format_string, *args, **kwargs)

    def format_field(self, value, format_spec: str) -> str:
        """
        Format a value given a formatting spec.

        If `format_spec` is in this Formatter's modifiers, the corresponding variation
        of value is given. If `format_spec` is 'r' (raw), the value is returned unmodified.
        If `format_spec` is not specified but `value` is in the mapping, the first variation is returned.

        Parameters
        ----------
        value : Any
            The value to format.
        format_spec : str
            The formatting spec.

        Returns
        -------
        str
            The formatted value.

        Examples
        --------
        Let's say the string "The dog is {adj1}, the goose is {adj2}" is to be translated
        to French and that we know that possible values of `adj` are `nice` and `evil`.
        In French, the genre of the noun changes the adjective (cat = chat is masculine,
        and goose = oie is feminine) so we initialize the formatter as:

        >>> fmt = AttrFormatter(
        ...     {
        ...         "nice": ["beau", "belle"],
        ...         "evil": ["méchant", "méchante"],
        ...         "smart": ["intelligent", "intelligente"],
        ...     },
        ...     ["m", "f"],
        ... )
        >>> fmt.format(
        ...     "Le chien est {adj1:m}, l'oie est {adj2:f}, le gecko est {adj3:r}",
        ...     adj1="nice",
        ...     adj2="evil",
        ...     adj3="smart",
        ... )
        "Le chien est beau, l'oie est méchante, le gecko est smart"

        The base values may be given using unix shell-like patterns:

        >>> fmt = AttrFormatter(
        ...     {"YS-*": ["annuel", "annuelle"], "MS": ["mensuel", "mensuelle"]},
        ...     ["m", "f"],
        ... )
        >>> fmt.format(
        ...     "La moyenne {freq:f} est faite sur un échantillon {src_timestep:m}",
        ...     freq="YS-JUL",
        ...     src_timestep="MS",
        ... )
        'La moyenne annuelle est faite sur un échantillon mensuel'
        """
        baseval = self._match_value(value)
        if baseval is None:  # Not something we know how to translate
            if format_spec in self.modifiers + ["r"]:  # Woops, however a known format spec was asked
                warnings.warn(f"Requested formatting `{format_spec}` for unknown string `{value}`.")
                format_spec = ""
            return super().format_field(value, format_spec)
        # Thus, known value

        if not format_spec:  # (None or '') No modifiers, return first
            return self.mapping[baseval][0]

        if format_spec == "r":  # Raw modifier
            return super().format_field(value, "")

        if format_spec in self.modifiers:  # Known modifier
            if len(self.mapping[baseval]) == 1:  # But unmodifiable entry
                return self.mapping[baseval][0]
            # Known modifier, modifiable entry
            return self.mapping[baseval][self.modifiers.index(format_spec)]
        # Known value but unknown modifier, must be a built-in one, only works for the default val...
        return super().format_field(self.mapping[baseval][0], format_spec)

    def _match_value(self, value):
        if isinstance(value, str):
            for mapval in self.mapping.keys():
                if fnmatch(value, mapval):
                    return mapval
        return None


# Tag mappings between keyword arguments and long-form text.
default_formatter = AttrFormatter(
    {
        # Arguments to "freq"
        "D": ["daily", "days"],
        "YS": ["annual", "years"],
        "YS-*": ["annual", "years"],
        "MS": ["monthly", "months"],
        "QS-*": ["seasonal", "seasons"],
        # Arguments to "indexer"
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
        # Arguments to "op / reducer / stat" (for example for generic.stats)
        "integral": ["integrated", "integral"],
        "count": ["count"],
        "doymin": ["day of minimum"],
        "doymax": ["day of maximum"],
        "mean": ["average"],
        "max": ["maximal", "maximum"],
        "min": ["minimal", "minimum"],
        "sum": ["total", "sum"],
        "std": ["standard deviation"],
        "var": ["variance"],
        "absamp": ["absolute amplitude"],
        "relamp": ["relative amplitude"],
        # For when we are formatting indicator classes with empty options
        "<class 'inspect._empty'>": ["<empty>"],
    },
    ["adj", "noun"],
)


def parse_doc(doc: str) -> dict:
    """
    Crude regex parsing reading an indice docstring and extracting information needed in indicator construction.

    The appropriate docstring syntax is detailed in :ref:`notebooks/extendxclim:Defining new indices`.

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
        return {}

    # Retrocompatilbity as python 3.13 does this by default
    doc = textwrap.dedent(doc)
    out = {}

    sections = re.split(r"(\w+\s?\w+)\n-{3,50}", doc)  # obj.__doc__.split('\n\n')
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
    """
    Parse the 'parameters' section of a docstring into a dictionary.

    Works by mapping the parameter name to its description and, potentially, to its set of choices.
    The type annotation are not parsed, except for fixed sets of values (listed as "{'a', 'b', 'c'}").
    The annotation parsing only accepts strings, numbers, `None` and `nan` (to represent `numpy.nan`).
    """
    curr_key = None
    params = {}

    for line in section.split("\n"):
        if line.startswith(" "):  # description
            s = " " if params[curr_key]["description"] else ""
            params[curr_key]["description"] += s + line.strip()
        elif not line.startswith(" ") and ":" in line:  # param title
            name, annot = line.split(":", maxsplit=1)
            curr_key = name.strip()
            params[curr_key] = {"description": ""}
            match = re.search(r".*(\{.*\}).*", annot)
            if match:
                try:
                    choices = literal_eval(match.groups()[0])
                    params[curr_key]["choices"] = choices
                except ValueError:  # noqa: S110
                    # If the literal_eval fails, we just ignore the choices.
                    pass
    return params


def _parse_returns(section):
    """Parse the returns section of a docstring into a dictionary mapping the parameter name to its description."""
    curr_key = None
    params = {}
    for line in section.split("\n"):
        if line.strip():
            if line.startswith(" "):  # long_name
                s = " " if params[curr_key]["long_name"] else ""
                params[curr_key]["long_name"] += s + line.strip()
            elif not line.startswith(" "):  # param title
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
    *inputs_list,  # : xr.DataArray | xr.Dataset
    new_line: str = "\n",
    missing_str: str | None = None,
    **inputs_kws: xr.DataArray | xr.Dataset,
) -> str:
    r"""
    Merge attributes from several DataArrays or Datasets.

    If more than one input is given, its name (if available) is prepended as: "<input name> : <input attribute>".

    Parameters
    ----------
    attribute : str
        The attribute to merge.
    *inputs_list : xr.DataArray or xr.Dataset
        The datasets or variables that were used to produce the new object.
        Inputs given that way will be prefixed by their `name` attribute if available.
    new_line : str
        The character to put between each instance of the attributes. Usually, in CF-conventions,
        the history attributes uses '\\n' while cell_methods uses ' '.
    missing_str : str
        A string that is printed if an input doesn't have the attribute. Defaults to None, in which
        case the input is simply skipped.
    **inputs_kws : xr.DataArray or xr.Dataset
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
            merged_attr += in_ds.attrs.get(attribute, "" if in_name is None else missing_str)
            merged_attr += new_line

    if len(new_line) > 0:
        return merged_attr[: -len(new_line)]  # Remove the last added new_line
    return merged_attr


def update_history(
    hist_str: str,
    *inputs_list,  # : xr.DataArray | xr.Dataset,
    new_name: str | None = None,
    **inputs_kws: xr.DataArray | xr.Dataset,
) -> str:
    r"""
    Return a history string with the timestamped message and the combination of the history of all inputs.

    The new history entry is formatted as "[<timestamp>] <new_name>: <hist_str> - xclim version: <xclim.__version__>."

    Parameters
    ----------
    hist_str : str
        The string describing what has been done on the data.
    *inputs_list : xr.DataArray or xr.Dataset
        The datasets or variables that were used to produce the new object.
        Inputs given that way will be prefixed by their "name" attribute if available.
    new_name : str, optional
        The name of the newly created variable or dataset to prefix hist_msg.
    **inputs_kws : xr.DataArray or xr.Dataset
        Mapping from names to the datasets or variables that were used to produce the new object.
        Inputs given that way will be prefixes by the passed name.

    Returns
    -------
    str
        The combine history of all inputs starting with `hist_str`.

    See Also
    --------
    merge_attributes : Merge attributes from several DataArrays or Datasets.
    """
    from xclim import (  # pylint: disable=cyclic-import,import-outside-toplevel
        __version__,
    )

    merged_history = merge_attributes(
        "history",
        *inputs_list,
        new_line="\n",
        missing_str="",
        **inputs_kws,
    )
    if len(merged_history) > 0 and not merged_history.endswith("\n"):
        merged_history += "\n"
    merged_history += (
        f"[{dt.datetime.now():%Y-%m-%d %H:%M:%S}] {new_name or ''}: {hist_str} - xclim version: {__version__}"
    )
    return merged_history


def update_xclim_history(func: Callable) -> Callable:
    """
    Decorator that auto-generates and fills the history attribute.

    The history is generated from the signature of the function and added to the first output.
    Because of a limitation of the `boltons` wrapper, all arguments passed to the wrapped function
    will be printed as keyword arguments.

    Parameters
    ----------
    func : Callable
        The function to decorate.

    Returns
    -------
    Callable
        The decorated function.
    """

    @wraps(func)
    def _call_and_add_history(*args, **kwargs):
        """Call the function and then generate and add the history attr."""
        outs = func(*args, **kwargs)

        if isinstance(outs, tuple):
            out = outs[0]
        else:
            out = outs

        if not isinstance(out, xr.DataArray | xr.Dataset):
            raise TypeError(f"Decorated `update_xclim_history` received a non-xarray output from {func.__name__}.")

        da_list = [arg for arg in args if isinstance(arg, xr.DataArray)]
        da_dict = {name: arg for name, arg in kwargs.items() if isinstance(arg, xr.DataArray)}

        # The wrapper hides how the user passed the arguments (positional or keyword)
        # Instead of having it all position, we have it all keyword-like for explicitness.
        bound_args = signature(func).bind(*args, **kwargs)
        attr = update_history(
            gen_call_string(func.__name__, **bound_args.arguments),
            *da_list,
            new_name=out.name if isinstance(out, xr.DataArray) else None,
            **da_dict,
        )
        out.attrs["history"] = attr
        return outs

    return _call_and_add_history


def gen_call_string(funcname: str, *args, **kwargs) -> str:
    r"""
    Generate a signature string for use in the history attribute.

    DataArrays and Dataset are replaced with their name, while Nones, floats, ints and strings are printed directly.
    All other objects have their type printed between < >.

    Arguments given through positional arguments are printed positionnally and those
    given through keywords are printed prefixed by their name.

    Parameters
    ----------
    funcname : str
        Name of the function.
    *args : Any
        Arguments given to the function.
    **kwargs : dict
        Keyword arguments given to the function.

    Returns
    -------
    str
        The formatted string.

    Examples
    --------
    >>> A = xr.DataArray([1], dims=("x",), name="A")
    >>> gen_call_string("func", A, b=2.0, c="3", d=[10] * 100)
    "func(A, b=2.0, c='3', d=<list>)"
    """
    elements = []
    chain = itertools.chain(zip([None] * len(args), args, strict=False), kwargs.items())
    for name, val in chain:
        if isinstance(val, xr.DataArray):
            rep = val.name or "<array>"
        elif isinstance(val, int | float | str | bool) or val is None:
            rep = repr(val)
        else:
            rep = repr(val)
            if len(rep) > 50:
                rep = f"<{type(val).__name__}>"

        if name is not None:
            rep = f"{name}={rep}"

        elements.append(rep)

    return f"{funcname}({', '.join(elements)})"


def prefix_attrs(source: dict, keys: Sequence, prefix: str) -> dict:
    """
    Rename some keys of a dictionary by adding a prefix.

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


def unprefix_attrs(source: dict, keys: Sequence, prefix: str) -> dict:
    """
    Remove prefix from keys in a dictionary.

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
    InputKind.QUANTIFIED: "quantity (string or DataArray, with units)",
    InputKind.FREQ_STR: "offset alias (string)",
    InputKind.NUMBER: "number",
    InputKind.NUMBER_SEQUENCE: "number or sequence of numbers",
    InputKind.STRING: "str",
    InputKind.DAY_OF_YEAR: "date (string, MM-DD)",
    InputKind.DATE: "date (string, YYYY-MM-DD)",
    InputKind.BOOL: "boolean",
    InputKind.DICT: "dict",
    InputKind.DATASET: "Dataset, optional",
    InputKind.KWARGS: "",
    InputKind.OTHER_PARAMETER: "Any",
}


def _gen_parameters_section(parameters: dict[str, dict[str, Any]], allowed_periods: list[str] | None = None) -> str:
    """
    Generate the "parameters" section of the indicator docstring.

    Parameters
    ----------
    parameters : dict
        Parameters dictionary (`Ind.parameters`).
    allowed_periods : list of str, optional
        Restrict parameters to specific periods. Default: None.

    Returns
    -------
    str
        The formatted section.
    """
    section = "Parameters\n----------\n"
    for name, param in parameters.items():
        desc_str = param.description
        if param.kind == InputKind.FREQ_STR:
            desc_str += (
                " See https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset"
                "-aliases for available options."
            )
            if allowed_periods is not None:
                desc_str += f" Restricted to frequencies equivalent to one of {allowed_periods}"
        if param.kind == InputKind.VARIABLE:
            defstr = f"Default : `ds.{param.default}`. "
        elif param.kind == InputKind.OPTIONAL_VARIABLE:
            defstr = ""
        elif param.default is not _empty:
            defstr = f"Default : {param.default}. "
        else:
            defstr = "Required. "
        if "choices" in param:
            annotstr = str(param.choices)
        else:
            annotstr = KIND_ANNOTATION[param.kind]
        if "units" in param and param.units is not None:
            unitstr = f"[Required units : {param.units}]"
        else:
            unitstr = ""
        section += f"{name} {': ' if annotstr else ''}{annotstr}\n    {desc_str}\n    {defstr}{unitstr}\n"
    return section


def _gen_returns_section(cf_attrs: Sequence[dict[str, Any]]) -> str:
    """
    Generate the "Returns" section of an indicator's docstring.

    Parameters
    ----------
    cf_attrs : Sequence[Dict[str, Any]]
        The list of attributes, usually Indicator.cf_attrs.

    Returns
    -------
    str
        The formatted section.
    """
    section = "Returns\n-------\n"
    for attrs in cf_attrs:
        if not section.endswith("\n"):
            section += "\n"
        section += f"{attrs['var_name']} : DataArray\n"
        section += f"    {attrs.get('long_name', '')}"
        if "standard_name" in attrs:
            section += f" ({attrs['standard_name']})"
        if "units" in attrs:
            section += f" [{attrs['units']}]"
        added_section = ""
        for key, attr in attrs.items():
            if key not in ["long_name", "standard_name", "units", "var_name"]:
                if callable(attr):
                    attr = "<Dynamically generated string>"
                added_section += f" **{key}**: {attr};"
        if added_section:
            section = f"{section}, with additional attributes:{added_section[:-1]}"
    section += "\n"
    return section


def generate_indicator_docstring(ind) -> str:
    """
    Generate an indicator's docstring from keywords.

    Parameters
    ----------
    ind : Indicator
        An Indicator instance.

    Returns
    -------
    str
        The docstring.
    """
    header = f"{ind.title} (realm: {ind.realm})\n\n{ind.abstract}\n"

    special = ""

    if hasattr(ind, "missing"):  # Only ResamplingIndicators
        special += f'This indicator will check for missing values according to the method "{ind.missing}".\n'
    if hasattr(ind.compute, "__module__"):
        special += f"Based on indice :py:func:`~{ind.compute.__module__}.{ind.compute.__name__}`.\n"
        if ind.injected_parameters:
            special += "With injected parameters: "
            special += ", ".join([f"{k}={v}" for k, v in ind.injected_parameters.items()])
            special += ".\n"
    if ind.keywords:
        special += f"Keywords : {ind.keywords}.\n"

    parameters = _gen_parameters_section(ind.parameters, getattr(ind, "allowed_periods", None))

    returns = _gen_returns_section(ind.cf_attrs)

    extras = ""
    for section in ["notes", "references"]:
        if getattr(ind, section):
            extras += f"{section.capitalize()}\n{'-' * len(section)}\n{getattr(ind, section)}\n\n"

    doc = f"{header}\n{special}\n{parameters}\n{returns}\n{extras}"
    return doc


def get_percentile_metadata(data: xr.DataArray, prefix: str) -> dict[str, str]:
    """
    Get the metadata related to percentiles from the given DataArray as a dictionary.

    Parameters
    ----------
    data : xr.DataArray
        Must be a percentile DataArray, this means the necessary metadata
        must be available in its attributes and coordinates.
    prefix : str
        The prefix to be used in the metadata key.
        Usually this takes the form of "tasmin_per" or equivalent.

    Returns
    -------
    dict
        A mapping of the configuration used to compute these percentiles.
    """
    # handle case where da was created with `quantile()` method
    if "quantile" in data.coords:
        percs = data.coords["quantile"].values * 100
    elif "percentiles" in data.coords:
        percs = data.coords["percentiles"].values
    else:
        percs = "<unknown percentiles>"
    clim_bounds = data.attrs.get("climatology_bounds", "<unknown bounds>")

    return {
        f"{prefix}_thresh": percs,
        f"{prefix}_window": data.attrs.get("window", "<unknown window>"),
        f"{prefix}_period": clim_bounds,
    }


# Adapted from xarray.structure.merge_attrs
def _merge_attrs_drop_conflicts(*objs):
    """Merge attributes from different xarray objects, dropping any attributes that conflict."""
    out = {}
    dropped = set()
    for obj in objs:
        attrs = obj.attrs
        out.update({key: value for key, value in attrs.items() if key not in out and key not in dropped})
        out = {key: value for key, value in out.items() if key not in attrs or _equivalent_attrs(attrs[key], value)}
        dropped |= {key for key in attrs if key not in out}
    return out


# Adapted from xarray.core.utils.equivalent
def _equivalent_attrs(first, second) -> bool:
    """Return whether two attributes are identical or not."""
    if first is second:
        return True
    if isinstance(first, list) or isinstance(second, list):
        if len(first) != len(second):
            return False
        return all(_equivalent_attrs(f, s) for f, s in zip(first, second, strict=False))
    return (first == second) or (pd.isnull(first) and pd.isnull(second))

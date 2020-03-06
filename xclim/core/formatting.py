# -*- coding: utf-8 -*-
"""
Formatting utilities for indicators
===================================
"""
import re
import string
from typing import Mapping
from typing import Sequence


class AttrFormatter(string.Formatter):
    """A formatter for frequently used attribute values.

    See the doc of format_field() for more details.
    """

    def __init__(
        self, mapping: Mapping[str, Sequence[str]], modifiers: Sequence[str],
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

        If `format_spec` is in this Formatter's modifiers, the correspong variation
        of value is given. If `format_spec` is not specified but `value` is in the
        mapping, the first variation is returned.

        Example
        -------
        Let's say the string "The dog is {adj1}, the goose is {adj2}" is to be translated
        to french and that we know that possible values of `adj` are `nice` and `evil`.
        In french, the genre of the noun changes the adjective (cat = chat is masculine,
        and goose = oie is feminine) so we initialize the formatter as:

        >>> fmt = AttrFormatter({'nice': ['beau', 'belle'], 'evil' : ['méchant', 'méchante']},
                                ['m', 'f'])
        >>> fmt.format("Le chien est {adj1:m}, l'oie est {adj2:f}",
                       adj1='nice', adj2='evil')
        "Le chien est beau, l'oie est méchante"
        """
        if value in self.mapping and not format_spec:
            return self.mapping[value][0]

        if format_spec in self.modifiers:
            if value in self.mapping:
                return self.mapping[value][self.modifiers.index(format_spec)]
            raise ValueError(
                f"No known mapping for string '{value}' with modifier '{format_spec}'"
            )
        return super().format_field(value, format_spec)


# Tag mappings between keyword arguments and long-form text.
default_formatter = AttrFormatter(
    {
        "YS": ["annual", "years"],
        "MS": ["monthly", "months"],
        "QS-DEC": ["seasonal", "seasons"],
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

    sections = re.split(r"(\w+)\n\s+-{4,50}", doc)  # obj.__doc__.split('\n\n')
    intro = sections.pop(0)
    if intro:
        content = list(map(str.strip, intro.strip().split("\n\n")))
        if len(content) == 1:
            out["title"] = content[0]
        elif len(content) == 2:
            out["title"], out["abstract"] = content

    for i in range(0, len(sections), 2):
        header, content = sections[i : i + 2]

        if header in ["Notes", "References"]:
            out[header.lower()] = content.replace("\n    ", "\n")
        elif header == "Parameters":
            pass
        elif header == "Returns":
            match = re.search(r"xarray\.DataArray\s*(.*)", content)
            if match:
                out["long_name"] = match.groups()[0]

    return out

"""
Testing and Tutorial Utilities' Module
======================================
"""

# Some of this code was copied and adapted from xarray
from __future__ import annotations

import logging
import os
import platform
import re
import sys
from collections.abc import Sequence
from importlib import import_module
from io import StringIO
from pathlib import Path
from typing import TextIO

_xclim_deps = [
    "xclim",
    "xarray",
    "statsmodels",
    "sklearn",
    "scipy",
    "pint",
    "pandas",
    "numpy",
    "numba",
    "lmoments3",
    "jsonpickle",
    "flox",
    "dask",
    "cf_xarray",
    "cftime",
    "clisops",
    "click",
    "bottleneck",
    "boltons",
]


logger = logging.getLogger("xclim")


__all__ = [
    "list_input_variables",
    "publish_release_notes",
    "run_doctests",
    "show_versions",
]


def list_input_variables(
    submodules: Sequence[str] | None = None, realms: Sequence[str] | None = None
) -> dict:
    """List all possible variables names used in xclim's indicators.

    Made for development purposes. Parses all indicator parameters with the
    :py:attr:`xclim.core.utils.InputKind.VARIABLE` or `OPTIONAL_VARIABLE` kinds.

    Parameters
    ----------
    realms: Sequence of str, optional
      Restrict the output to indicators of a list of realms only. Default None, which parses all indicators.
    submodules: str, optional
      Restrict the output to indicators of a list of submodules only. Default None, which parses all indicators.

    Returns
    -------
    dict
      A mapping from variable name to indicator class.
    """
    from collections import defaultdict  # pylint: disable=import-outside-toplevel

    from xclim import indicators  # pylint: disable=import-outside-toplevel
    from xclim.core.indicator import registry  # pylint: disable=import-outside-toplevel
    from xclim.core.utils import InputKind  # pylint: disable=import-outside-toplevel

    submodules = submodules or [
        sub for sub in dir(indicators) if not sub.startswith("__")
    ]
    realms = realms or ["atmos", "ocean", "land", "seaIce"]

    variables = defaultdict(list)
    for name, ind in registry.items():
        if "." in name:
            # external submodule, submodule name is prepended to registry key
            if name.split(".")[0] not in submodules:
                continue
        elif ind.realm not in submodules:
            # official indicator : realm == submodule
            continue
        if ind.realm not in realms:
            continue

        # ok we want this one.
        for varname, meta in ind._all_parameters.items():
            if meta.kind in [
                InputKind.VARIABLE,
                InputKind.OPTIONAL_VARIABLE,
            ]:
                var = meta.default or varname
                variables[var].append(ind)

    return variables


def run_doctests():
    """Run the doctests for the module."""
    import pytest

    cmd = [
        f"--rootdir={Path(__file__).absolute().parent}",
        "--numprocesses=0",
        "--xdoctest",
        f"{Path(__file__).absolute().parents[1]}",
    ]

    sys.exit(pytest.main(cmd))


def publish_release_notes(
    style: str = "md",
    file: os.PathLike | StringIO | TextIO | None = None,
    changes: str | os.PathLike | None = None,
) -> str | None:
    """Format release notes in Markdown or ReStructuredText.

    Parameters
    ----------
    style : {"rst", "md"}
        Use ReStructuredText formatting or Markdown. Default: Markdown.
    file : {os.PathLike, StringIO, TextIO}, optional
        If provided, prints to the given file-like object. Otherwise, returns a string.
    changes : {str, os.PathLike}, optional
        If provided, manually points to the file where the changelog can be found.
        Assumes a relative path otherwise.

    Returns
    -------
    str, optional

    Notes
    -----
    This function is used solely for development and packaging purposes.
    """
    if isinstance(changes, (str, Path)):
        changes_file = Path(changes).absolute()
    else:
        changes_file = Path(__file__).absolute().parents[2].joinpath("CHANGELOG.rst")

    if not changes_file.exists():
        raise FileNotFoundError("Changelog file not found in xclim folder tree.")

    with open(changes_file) as hf:
        changes = hf.read()

    if style == "rst":
        hyperlink_replacements = {
            r":issue:`([0-9]+)`": r"`GH/\1 <https://github.com/Ouranosinc/xclim/issues/\1>`_",
            r":pull:`([0-9]+)`": r"`PR/\1 <https://github.com/Ouranosinc/xclim/pull/\>`_",
            r":user:`([a-zA-Z0-9_.-]+)`": r"`@\1 <https://github.com/\1>`_",
        }
    elif style == "md":
        hyperlink_replacements = {
            r":issue:`([0-9]+)`": r"[GH/\1](https://github.com/Ouranosinc/xclim/issues/\1)",
            r":pull:`([0-9]+)`": r"[PR/\1](https://github.com/Ouranosinc/xclim/pull/\1)",
            r":user:`([a-zA-Z0-9_.-]+)`": r"[@\1](https://github.com/\1)",
        }
    else:
        raise NotImplementedError()

    for search, replacement in hyperlink_replacements.items():
        changes = re.sub(search, replacement, changes)

    if style == "md":
        changes = changes.replace("=========\nChangelog\n=========", "# Changelog")

        titles = {r"\n(.*?)\n([\-]{1,})": "-", r"\n(.*?)\n([\^]{1,})": "^"}
        for title_expression, level in titles.items():
            found = re.findall(title_expression, changes)
            for grouping in found:
                fixed_grouping = (
                    str(grouping[0]).replace("(", r"\(").replace(")", r"\)")
                )
                search = rf"({fixed_grouping})\n([\{level}]{'{' + str(len(grouping[1])) + '}'})"
                replacement = f"{'##' if level == '-' else '###'} {grouping[0]}"
                changes = re.sub(search, replacement, changes)

        link_expressions = r"[\`]{1}([\w\s]+)\s<(.+)>`\_"
        found = re.findall(link_expressions, changes)
        for grouping in found:
            search = rf"`{grouping[0]} <.+>`\_"
            replacement = f"[{str(grouping[0]).strip()}]({grouping[1]})"
            changes = re.sub(search, replacement, changes)

    if not file:
        return changes
    if isinstance(file, (Path, os.PathLike)):
        with Path(file).open("w") as f:
            print(changes, file=f)
    else:
        print(changes, file=file)
    return None


def show_versions(
    file: os.PathLike | StringIO | TextIO | None = None,
    deps: list[str] | None = None,
) -> str | None:
    """Print the versions of xclim and its dependencies.

    Parameters
    ----------
    file : {os.PathLike, StringIO, TextIO}, optional
        If provided, prints to the given file-like object. Otherwise, returns a string.
    deps : list of str, optional
        A list of dependencies to gather and print version information from. Otherwise, prints `xclim` dependencies.

    Returns
    -------
    str or None
    """
    dependencies: list[str]
    if deps is None:
        dependencies = _xclim_deps
    else:
        dependencies = deps

    dependency_versions = [(d, lambda mod: mod.__version__) for d in dependencies]

    deps_blob: list[tuple[str, str | None]] = []
    for modname, ver_f in dependency_versions:
        try:
            if modname in sys.modules:
                mod = sys.modules[modname]
            else:
                mod = import_module(modname)
        except (KeyError, ModuleNotFoundError):
            deps_blob.append((modname, None))
        else:
            try:
                ver = ver_f(mod)
                deps_blob.append((modname, ver))
            except AttributeError:
                deps_blob.append((modname, "installed"))

    modules_versions = "\n".join([f"{k}: {stat}" for k, stat in sorted(deps_blob)])

    installed_versions = [
        "INSTALLED VERSIONS",
        "------------------",
        f"python: {platform.python_version()}",
        f"{modules_versions}",
        f"Anaconda-based environment: {'yes' if Path(sys.base_prefix).joinpath('conda-meta').exists() else 'no'}",
    ]

    message = "\n".join(installed_versions)

    if not file:
        return message
    if isinstance(file, (Path, os.PathLike)):
        with Path(file).open("w") as f:
            print(message, file=f)
    else:
        print(message, file=file)
    return None

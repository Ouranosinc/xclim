"""Testing and tutorial utilities' module."""
# Some of this code was copied and adapted from xarray
from __future__ import annotations

import hashlib
import importlib
import json
import logging
import os
import platform
import re
import sys
import warnings
from io import StringIO
from pathlib import Path
from typing import Sequence, TextIO
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import urlopen, urlretrieve

import pandas as pd
from xarray import Dataset
from xarray import open_dataset as _open_dataset
from yaml import safe_dump, safe_load

from xclim import __version__ as xclim_version

_default_cache_dir = Path.home() / ".xclim_testing_data"

LOGGER = logging.getLogger("xclim")

__all__ = [
    "get_all_CMIP6_variables",
    "list_datasets",
    "list_input_variables",
    "open_dataset",
    "publish_release_notes",
    "update_variable_yaml",
    "show_versions",
]


def file_md5_checksum(fname):
    hash_md5 = hashlib.md5()  # nosec
    with open(fname, "rb") as f:
        hash_md5.update(f.read())
    return hash_md5.hexdigest()


def _get(
    fullname: Path,
    github_url: str,
    branch: str,
    suffix: str,
    cache_dir: Path,
) -> Path:
    cache_dir = cache_dir.absolute()
    local_file = cache_dir / branch / fullname
    md5_name = fullname.with_suffix(f"{suffix}.md5")
    md5_file = cache_dir / branch / md5_name

    if not github_url.lower().startswith("http"):
        raise ValueError(f"GitHub URL not safe: '{github_url}'.")

    if local_file.is_file():
        local_md5 = file_md5_checksum(local_file)
        try:
            url = "/".join((github_url, "raw", branch, md5_name.as_posix()))
            LOGGER.info("Attempting to fetch remote file md5: %s" % md5_name.as_posix())
            urlretrieve(url, md5_file)  # nosec
            with open(md5_file) as f:
                remote_md5 = f.read()
            if local_md5.strip() != remote_md5.strip():
                local_file.unlink()
                msg = (
                    f"MD5 checksum for {local_file.as_posix()} does not match upstream md5. "
                    "Attempting new download."
                )
                warnings.warn(msg)
        except (HTTPError, URLError):
            msg = f"{md5_name.as_posix()} not accessible online. Unable to determine validity with upstream repo."
            warnings.warn(msg)

    if not local_file.is_file():
        # This will always leave this directory on disk.
        # We may want to add an option to remove it.
        local_file.parent.mkdir(parents=True, exist_ok=True)

        url = "/".join((github_url, "raw", branch, fullname.as_posix()))
        LOGGER.info("Fetching remote file: %s" % fullname.as_posix())
        urlretrieve(url, local_file)  # nosec
        try:
            url = "/".join((github_url, "raw", branch, md5_name.as_posix()))
            LOGGER.info("Fetching remote file md5: %s" % md5_name.as_posix())
            urlretrieve(url, md5_file)  # nosec
        except HTTPError as e:
            msg = f"{md5_name.as_posix()} not found. Aborting file retrieval."
            local_file.unlink()
            raise FileNotFoundError(msg) from e

        local_md5 = file_md5_checksum(local_file)
        try:
            with open(md5_file) as f:
                remote_md5 = f.read()
            if local_md5.strip() != remote_md5.strip():
                local_file.unlink()
                msg = (
                    f"{local_file.as_posix()} and md5 checksum do not match. "
                    "There may be an issue with the upstream origin data."
                )
                raise OSError(msg)
        except OSError as e:
            LOGGER.error(e)

    return local_file


# idea copied from raven that it borrowed from xclim that borrowed it from xarray that was borrowed from Seaborn
def open_dataset(
    name: str,
    suffix: str | None = None,
    dap_url: str | None = None,
    github_url: str = "https://github.com/Ouranosinc/xclim-testdata",
    branch: str = "main",
    cache: bool = True,
    cache_dir: Path = _default_cache_dir,
    **kwargs,
) -> Dataset:
    """
    Open a dataset from the online GitHub-like repository.

    If a local copy is found then always use that to avoid network traffic.

    Parameters
    ----------
    name : str
        Name of the file containing the dataset.
    suffix : str, optional
        If no suffix is given, assumed to be netCDF ('.nc' is appended). For no suffix, set "".
    dap_url : str, optional
        URL to OPeNDAP folder where the data is stored. If supplied, supersedes github_url.
    github_url : str
        URL to GitHub repository where the data is stored.
    branch : str, optional
        For GitHub-hosted files, the branch to download from.
    cache_dir : Path
        The directory in which to search for and write cached data.
    cache : bool
        If True, then cache data locally for use on subsequent calls.
    kwargs
        For NetCDF files, keywords passed to :py:func:`xarray.open_dataset`.

    Returns
    -------
    Union[Dataset, Path]

    See Also
    --------
    xarray.open_dataset
    """
    name = Path(name)
    if suffix is None:
        suffix = ".nc"
    fullname = name.with_suffix(suffix)

    if not github_url.lower().startswith("http"):
        raise ValueError(f"GitHub URL not safe: '{github_url}'.")

    if dap_url is not None:
        if not dap_url.lower().startswith("http"):
            raise ValueError(f"OPeNDAP URL not safe: '{dap_url}'.")

        dap_file = urljoin(dap_url, str(name))
        try:
            ds = _open_dataset(dap_file, **kwargs)
            return ds
        except OSError:
            msg = "OPeNDAP file not read. Verify that the service is available."
            LOGGER.error(msg)
            raise OSError(msg)

    local_file = _get(
        fullname=fullname,
        github_url=github_url,
        branch=branch,
        suffix=suffix,
        cache_dir=cache_dir,
    )

    try:
        ds = _open_dataset(local_file, **kwargs)
        if not cache:
            ds = ds.load()
            local_file.unlink()
        return ds
    except OSError:
        raise


def list_datasets(github_repo="Ouranosinc/xclim-testdata", branch="main"):
    """Return a DataFrame listing all xclim test datasets available on the GitHub repo for the given branch.

    The result includes the filepath, as passed to `open_dataset`, the file size (in KB) and the html url to the file.
    This uses an unauthenticated call to GitHub's REST API, so it is limited to 60 requests per hour (per IP).
    A single call of this function triggers one request per subdirectory, so use with parsimony.
    """
    res = urlopen(  # nosec
        f"https://api.github.com/repos/{github_repo}/contents?ref={branch}"
    )
    base = json.loads(res.read().decode())
    records = []
    for folder in base:
        if folder["path"].startswith(".") or folder["size"] > 0:
            # drop hidden folders and other files.
            continue
        res = urlopen(folder["url"])  # nosec
        listing = json.loads(res.read().decode())
        for file in listing:
            if file["path"].endswith(".nc"):
                records.append(
                    {
                        "name": file["path"],
                        "size": file["size"] / 2**10,
                        "url": file["html_url"],
                    }
                )
    df = pd.DataFrame.from_records(records).set_index("name")
    print(f"Found {len(df)} datasets.")
    return df


def list_input_variables(
    submodules: Sequence[str] = None, realms: Sequence[str] = None
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
    from collections import defaultdict

    from xclim import indicators
    from xclim.core.indicator import registry
    from xclim.core.utils import InputKind

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


def get_all_CMIP6_variables(get_cell_methods=True):  # noqa
    data = pd.read_excel(
        "http://proj.badc.rl.ac.uk/svn/exarch/CMIP6dreq/tags/01.00.33/dreqPy/docs/CMIP6_MIP_tables.xlsx",
        sheet_name=None,
    )
    data.pop("Notes")

    variables = {}

    def summarize_cell_methods(rawstr):
        words = str(rawstr).split(" ")
        iskey = [word.endswith(":") for word in words]
        cms = {}
        for i in range(len(words)):
            if iskey[i] and i + 1 < len(words) and not iskey[i + 1]:
                cms[words[i][:-1]] = words[i + 1]
        return cms

    for table, df in data.items():
        for i, row in df.iterrows():
            varname = row["Variable Name"]
            vardata = {
                "standard_name": row["CF Standard Name"],
                "canonical_units": row["units"],
            }
            if get_cell_methods:
                vardata["cell_methods"] = [summarize_cell_methods(row["cell_methods"])]
            if varname in variables and get_cell_methods:
                if vardata["cell_methods"] not in variables[varname]["cell_methods"]:
                    variables[varname]["cell_methods"].append(vardata["cell_methods"])
            else:
                variables[varname] = vardata

    return variables


def update_variable_yaml(filename=None, xclim_needs_only=True):
    """Update a variable from a yaml file."""
    print("Downloading CMIP6 variables.")
    all_vars = get_all_CMIP6_variables(get_cell_methods=False)

    if xclim_needs_only:
        print("Filtering with xclim-implemented variables.")
        xc_vars = list_input_variables()
        all_vars = {k: v for k, v in all_vars.items() if k in xc_vars}

    filepath = Path(
        filename or (Path(__file__).parent.parent / "data" / "variables.yml")
    )

    if filepath.exists():
        with filepath.open() as f:
            std_vars = safe_load(f)

        for var, data in all_vars.items():
            if var not in std_vars["variables"]:
                print(f"Added {var}")
                new_data = data.copy()
                new_data.update(description="")
                std_vars["variables"][var] = new_data
    else:
        std_vars = all_vars

    with filepath.open("w") as f:
        safe_dump(std_vars, f)


def publish_release_notes(
    style: str = "md", file: os.PathLike | StringIO | TextIO | None = None
) -> str | None:
    """Format release history in Markdown or ReStructuredText.

    Parameters
    ----------
    style: {"rst", "md"}
      Use ReStructuredText formatting or Markdown. Default: Markdown.
    file: {os.PathLike, StringIO, TextIO}, optional
      If provided, prints to the given file-like object. Otherwise, returns a string.

    Returns
    -------
    str, optional

    Notes
    -----
    This function is solely for development purposes.
    """
    history_file = Path(__file__).parent.parent.parent.joinpath("HISTORY.rst")

    if not history_file.exists():
        raise FileNotFoundError("History file not found in xclim file tree.")

    with open(history_file) as hf:
        history = hf.read()

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
        history = re.sub(search, replacement, history)

    if style == "md":
        history = history.replace("=======\nHistory\n=======", "# History")

        titles = {r"\n(.*?)\n([\-]{1,})": "-", r"\n(.*?)\n([\^]{1,})": "^"}
        for title_expression, level in titles.items():
            found = re.findall(title_expression, history)
            for grouping in found:
                fixed_grouping = (
                    str(grouping[0]).replace("(", r"\(").replace(")", r"\)")
                )
                search = rf"({fixed_grouping})\n([\{level}]{'{' + str(len(grouping[1])) + '}'})"
                replacement = f"{'##' if level=='-' else '###'} {grouping[0]}"
                history = re.sub(search, replacement, history)

        link_expressions = r"[\`]{1}([\w\s]+)\s<(.+)>`\_"
        found = re.findall(link_expressions, history)
        for grouping in found:
            search = rf"`{grouping[0]} <.+>`\_"
            replacement = f"[{str(grouping[0]).strip()}]({grouping[1]})"
            history = re.sub(search, replacement, history)

    if not file:
        return history
    elif isinstance(file, (Path, os.PathLike)):
        file = Path(file).open("w")
    print(history, file=file)


def show_versions(file: os.PathLike | StringIO | TextIO | None = None) -> str | None:
    """Print the versions of xclim and its dependencies.

    Parameters
    ----------
    file : {os.PathLike, StringIO, TextIO}, optional
      If provided, prints to the given file-like object. Otherwise, returns a string.

    Returns
    -------
    str or None
    """
    deps = [
        ("xarray", lambda mod: mod.__version__),
        ("sklearn", lambda mod: mod.__version__),
        ("scipy", lambda mod: mod.__version__),
        ("pint", lambda mod: mod.__version__),
        ("pandas", lambda mod: mod.__version__),
        ("numba", lambda mod: mod.__version__),
        ("dask", lambda mod: mod.__version__),
        ("cf_xarray", lambda mod: mod.__version__),
        ("cftime", lambda mod: mod.__version__),
        ("clisops", lambda mod: mod.__version__),
        ("bottleneck", lambda mod: mod.__version__),
        ("boltons", lambda mod: mod.__version__),
    ]

    deps_blob = []
    for (modname, ver_f) in deps:
        try:
            if modname in sys.modules:
                mod = sys.modules[modname]
            else:
                mod = importlib.import_module(modname)
        except (KeyError, ModuleNotFoundError):
            deps_blob.append((modname, None))
        else:
            try:
                ver = ver_f(mod)
                deps_blob.append((modname, ver))
            except AttributeError:
                deps_blob.append((modname, "installed"))

    modules_versions = "\n".join([f"{k}: {stat}" for k, stat in sorted(deps_blob)])

    installed_versions = (
        "\n"
        "INSTALLED VERSIONS\n"
        "------------------\n"
        f"python: {platform.python_version()}\n"
        f"xclim: {xclim_version}\n"
        f"{modules_versions}\n"
        f"Anaconda-based environment: {'yes' if Path(sys.base_prefix).joinpath('conda-meta').exists() else 'no'}"
    )

    if not file:
        return installed_versions
    elif isinstance(file, (Path, os.PathLike)):
        file = Path(file).open("w")
    print(installed_versions, file=file)

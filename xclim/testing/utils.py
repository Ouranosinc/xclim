"""
Testing and Tutorial Utilities' Module
======================================
"""
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
from shutil import copy
from typing import Sequence, TextIO
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import urlopen, urlretrieve

import pandas as pd
from xarray import Dataset
from xarray import open_dataset as _open_dataset

_xclim_deps = [
    "xclim",
    "xarray",
    "statsmodels",
    "sklearn",
    "scipy",
    "pint",
    "pandas",
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


_default_cache_dir = Path.home() / ".xclim_testing_data"

logger = logging.getLogger("xclim")

__all__ = [
    "_default_cache_dir",
    "get_file",
    "get_local_testdata",
    "list_datasets",
    "list_input_variables",
    "open_dataset",
    "publish_release_notes",
    "show_versions",
]


def file_md5_checksum(f_name):
    hash_md5 = hashlib.md5()  # nosec
    with open(f_name, "rb") as f:
        hash_md5.update(f.read())
    return hash_md5.hexdigest()


def get_file(
    name: str | os.PathLike | Sequence[str | os.PathLike],
    github_url: str = "https://github.com/Ouranosinc/xclim-testdata",
    branch: str = "master",
    cache_dir: Path = _default_cache_dir,
) -> Path | list[Path]:
    """Return a file from an online GitHub-like repository.

    If a local copy is found then always use that to avoid network traffic.

    Parameters
    ----------
    name : str | os.PathLike | Sequence[str | os.PathLike]
        Name of the file or list/tuple of names of files containing the dataset(s) including suffixes.
    github_url : str
        URL to GitHub repository where the data is stored.
    branch : str, optional
        For GitHub-hosted files, the branch to download from.
    cache_dir : Path
        The directory in which to search for and write cached data.

    Returns
    -------
    Path | list[Path]
    """
    if isinstance(name, (str, Path)):
        name = [name]

    files = list()
    for n in name:
        fullname = Path(n)
        suffix = fullname.suffix
        files.append(
            _get(
                fullname=fullname,
                github_url=github_url,
                branch=branch,
                suffix=suffix,
                cache_dir=cache_dir,
            )
        )
    if len(files) == 1:
        return files[0]
    return files


def get_local_testdata(
    patterns: str | Sequence[str],
    temp_folder: str | os.PathLike,
    branch: str = "master",
    _local_cache: str | os.PathLike = _default_cache_dir,
) -> Path | list[Path]:
    """Copy specific testdata from a default cache to a temporary folder.

    Return files matching `pattern` in the default cache dir and move to a local temp folder.

    Parameters
    ----------
    patterns : str | Sequence[str]
        Glob patterns, which must include the folder.
    temp_folder : str | os.PathLike
        Target folder to copy files and filetree to.
    branch : str
        For GitHub-hosted files, the branch to download from.
    _local_cache : str | os.PathLike
        Local cache of testing data.

    Returns
    -------
    Path | list[Path]
    """
    temp_paths = []

    if isinstance(patterns, str):
        patterns = [patterns]

    for pattern in patterns:
        potential_paths = [
            path for path in Path(temp_folder).joinpath(branch).glob(pattern)
        ]
        if potential_paths:
            temp_paths.extend(potential_paths)
            continue

        testdata_path = Path(_local_cache)
        if not testdata_path.exists():
            raise RuntimeError(f"{testdata_path} does not exists")
        paths = [path for path in testdata_path.joinpath(branch).glob(pattern)]
        if not paths:
            raise FileNotFoundError(
                f"No data found for {pattern} at {testdata_path}/{branch}."
            )

        main_folder = Path(temp_folder).joinpath(branch).joinpath(Path(pattern).parent)
        main_folder.mkdir(exist_ok=True, parents=True)

        for file in paths:
            temp_file = main_folder.joinpath(file.name)
            if not temp_file.exists():
                copy(file, main_folder)
            temp_paths.append(temp_file)

    # Return item directly when singleton, for convenience
    return temp_paths[0] if len(temp_paths) == 1 else temp_paths


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
            logger.info(f"Attempting to fetch remote file md5: {md5_name.as_posix()}")
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
        logger.info(f"Fetching remote file: {fullname.as_posix()}")
        urlretrieve(url, local_file)  # nosec
        try:
            url = "/".join((github_url, "raw", branch, md5_name.as_posix()))
            logger.info(f"Fetching remote file md5: {md5_name.as_posix()}")
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
            logger.error(e)

    return local_file


# idea copied from raven that it borrowed from xclim that borrowed it from xarray that was borrowed from Seaborn
def open_dataset(
    name: str | os.PathLike,
    suffix: str | None = None,
    dap_url: str | None = None,
    github_url: str = "https://github.com/Ouranosinc/xclim-testdata",
    branch: str = "main",
    cache: bool = True,
    cache_dir: Path = _default_cache_dir,
    **kwargs,
) -> Dataset:
    r"""Open a dataset from the online GitHub-like repository.

    If a local copy is found then always use that to avoid network traffic.

    Parameters
    ----------
    name : str or os.PathLike
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
    \*\*kwargs
        For NetCDF files, keywords passed to :py:func:`xarray.open_dataset`.

    Returns
    -------
    Union[Dataset, Path]

    See Also
    --------
    xarray.open_dataset
    """
    if isinstance(name, str):
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
        except OSError as err:
            msg = "OPeNDAP file not read. Verify that the service is available."
            logger.error(msg)
            raise OSError(msg) from err

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
    except OSError as err:
        raise err


def list_datasets(github_repo="Ouranosinc/xclim-testdata", branch="main"):
    """Return a DataFrame listing all xclim test datasets available on the GitHub repo for the given branch.

    The result includes the filepath, as passed to `open_dataset`, the file size (in KB) and the html url to the file.
    This uses an unauthenticated call to GitHub's REST API, so it is limited to 60 requests per hour (per IP).
    A single call of this function triggers one request per subdirectory, so use with parsimony.
    """
    with urlopen(  # nosec
        f"https://api.github.com/repos/{github_repo}/contents?ref={branch}"
    ) as res:
        base = json.loads(res.read().decode())
    records = []
    for folder in base:
        if folder["path"].startswith(".") or folder["size"] > 0:
            # drop hidden folders and other files.
            continue
        with urlopen(folder["url"]) as res:  # nosec
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
        changes_file = Path(__file__).absolute().parents[2].joinpath("CHANGES.rst")

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
                replacement = f"{'##' if level=='-' else '###'} {grouping[0]}"
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
        file = Path(file).open("w")
    print(changes, file=file)


def show_versions(
    file: os.PathLike | StringIO | TextIO | None = None,
    deps: list | None = None,
) -> str | None:
    """Print the versions of xclim and its dependencies.

    Parameters
    ----------
    file : {os.PathLike, StringIO, TextIO}, optional
        If provided, prints to the given file-like object. Otherwise, returns a string.
    deps : list, optional
        A list of dependencies to gather and print version information from. Otherwise, prints `xclim` dependencies.

    Returns
    -------
    str or None
    """
    if deps is None:
        deps = _xclim_deps

    dependency_versions = [(d, lambda mod: mod.__version__) for d in deps]

    deps_blob = []
    for modname, ver_f in dependency_versions:
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
        file = Path(file).open("w")
    print(message, file=file)

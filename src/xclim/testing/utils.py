"""
Testing and Tutorial Utilities' Module
======================================
"""

from __future__ import annotations

import importlib.resources as ilr
import logging
import os
import platform
import re
import sys
import time
import warnings
from collections.abc import Callable, Sequence
from datetime import datetime as dt
from functools import wraps
from importlib import import_module
from io import StringIO
from pathlib import Path
from shutil import copytree
from typing import IO, TextIO
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlparse
from urllib.request import urlretrieve

from filelock import FileLock
from packaging.version import Version
from xarray import Dataset
from xarray import open_dataset as _open_dataset

import xclim
from xclim import __version__ as __xclim_version__

try:
    import pytest
    from pytest_socket import SocketBlockedError
except ImportError:
    pytest = None
    SocketBlockedError = None

try:
    import pooch
except ImportError:
    warnings.warn("The `pooch` library is not installed. The default cache directory for testing data will not be set.")
    pooch = None


logger = logging.getLogger("xclim")


__all__ = [
    "TESTDATA_BRANCH",
    "TESTDATA_CACHE_DIR",
    "TESTDATA_REPO_URL",
    "audit_url",
    "default_testdata_cache",
    "default_testdata_repo_url",
    "default_testdata_version",
    "gather_testing_data",
    "list_input_variables",
    "nimbus",
    "open_dataset",
    "populate_testing_data",
    "publish_release_notes",
    "run_doctests",
    "show_versions",
    "testing_setup_warnings",
]

default_testdata_version = "v2025.3.11"
"""Default version of the testing data to use when fetching datasets."""

default_testdata_repo_url = "https://raw.githubusercontent.com/Ouranosinc/xclim-testdata/"
"""Default URL of the testing data repository to use when fetching datasets."""

try:
    default_testdata_cache = Path(pooch.os_cache("xclim-testdata"))
    """Default location for the testing data cache."""
except AttributeError:
    default_testdata_cache = None

TESTDATA_REPO_URL = str(os.getenv("XCLIM_TESTDATA_REPO_URL", default_testdata_repo_url))
"""
Sets the URL of the testing data repository to use when fetching datasets.

Notes
-----
When running tests locally, this can be set for both `pytest` and `tox` by exporting the variable:

.. code-block:: console

    $ export XCLIM_TESTDATA_REPO_URL="https://github.com/my_username/xclim-testdata"

or setting the variable at runtime:

.. code-block:: console

    $ env XCLIM_TESTDATA_REPO_URL="https://github.com/my_username/xclim-testdata" pytest
"""

TESTDATA_BRANCH = str(os.getenv("XCLIM_TESTDATA_BRANCH", default_testdata_version))
"""
Sets the branch of the testing data repository to use when fetching datasets.

Notes
-----
When running tests locally, this can be set for both `pytest` and `tox` by exporting the variable:

.. code-block:: console

    $ export XCLIM_TESTDATA_BRANCH="my_testing_branch"

or setting the variable at runtime:

.. code-block:: console

    $ env XCLIM_TESTDATA_BRANCH="my_testing_branch" pytest
"""

TESTDATA_CACHE_DIR = os.getenv("XCLIM_TESTDATA_CACHE_DIR", default_testdata_cache)
"""
Sets the directory to store the testing datasets.

If not set, the default location will be used (based on ``platformdirs``, see :func:`pooch.os_cache`).

Notes
-----
When running tests locally, this can be set for both `pytest` and `tox` by exporting the variable:

.. code-block:: console

    $ export XCLIM_TESTDATA_CACHE_DIR="/path/to/my/data"

or setting the variable at runtime:

.. code-block:: console

    $ env XCLIM_TESTDATA_CACHE_DIR="/path/to/my/data" pytest
"""


def list_input_variables(submodules: Sequence[str] | None = None, realms: Sequence[str] | None = None) -> dict:
    """
    List all possible variables names used in xclim's indicators.

    Made for development purposes. Parses all indicator parameters with the
    :py:attr:`xclim.core.utils.InputKind.VARIABLE` or `OPTIONAL_VARIABLE` kinds.

    Parameters
    ----------
    submodules : str, optional
        Restrict the output to indicators of a list of submodules only. Default None, which parses all indicators.
    realms : Sequence of str, optional
        Restrict the output to indicators of a list of realms only. Default None, which parses all indicators.

    Returns
    -------
    dict
        A mapping from variable name to indicator class.
    """
    from collections import defaultdict  # pylint: disable=import-outside-toplevel

    from xclim import indicators  # pylint: disable=import-outside-toplevel
    from xclim.core.indicator import registry  # pylint: disable=import-outside-toplevel
    from xclim.core.utils import InputKind  # pylint: disable=import-outside-toplevel

    submodules = submodules or [sub for sub in dir(indicators) if not sub.startswith("__")]
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


# Publishing Tools ###


def publish_release_notes(
    style: str = "md",
    file: os.PathLike[str] | StringIO | TextIO | None = None,
    changes: str | os.PathLike[str] | None = None,
) -> str | None:
    """
    Format release notes in Markdown or ReStructuredText.

    Parameters
    ----------
    style : {"rst", "md"}
        Use ReStructuredText formatting or Markdown. Default: Markdown.
    file : {os.PathLike, StringIO, TextIO}, optional
        If provided, prints to the given file-like object. Otherwise, returns a string.
    changes : str or os.PathLike[str], optional
        If provided, manually points to the file where the changelog can be found.
        Assumes a relative path otherwise.

    Returns
    -------
    str, optional
        If `file` not provided, the formatted release notes.

    Notes
    -----
    This function is used solely for development and packaging purposes.
    """
    if isinstance(changes, str | Path):
        changes_file = Path(changes).absolute()
    else:
        changes_file = Path(__file__).absolute().parents[3].joinpath("CHANGELOG.rst")

    if not changes_file.exists():
        raise FileNotFoundError("Changelog file not found in xclim folder tree.")

    with open(changes_file, encoding="utf-8") as hf:
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
        msg = f"Formatting style not supported: {style}"
        raise NotImplementedError(msg)

    for search, replacement in hyperlink_replacements.items():
        changes = re.sub(search, replacement, changes)

    if style == "md":
        changes = changes.replace("=========\nChangelog\n=========", "# Changelog")

        titles = {r"\n(.*?)\n([\-]{1,})": "-", r"\n(.*?)\n([\^]{1,})": "^"}
        for title_expression, level in titles.items():
            found = re.findall(title_expression, changes)
            for grouping in found:
                fixed_grouping = str(grouping[0]).replace("(", r"\(").replace(")", r"\)")
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
    if isinstance(file, Path | os.PathLike):
        with open(file, "w", encoding="utf-8") as f:
            print(changes, file=f)
    else:
        print(changes, file=file)
    return None


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


def show_versions(
    file: os.PathLike | StringIO | TextIO | None = None,
    deps: list[str] | None = None,
) -> str | None:
    """
    Print the versions of xclim and its dependencies.

    Parameters
    ----------
    file : {os.PathLike, StringIO, TextIO}, optional
        If provided, prints to the given file-like object. Otherwise, returns a string.
    deps : list of str, optional
        A list of dependencies to gather and print version information from.
        Otherwise, prints `xclim` dependencies.

    Returns
    -------
    str or None
        If `file` not provided, the versions of xclim and its dependencies.
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
    if isinstance(file, Path | os.PathLike):
        with open(file, "w", encoding="utf-8") as f:
            print(message, file=f)
    else:
        print(message, file=file)
    return None


# Test Data Utilities ###


def run_doctests():
    """Run the doctests for the module."""
    if pytest is None:
        raise ImportError(
            "The `pytest` package is required to run the doctests. "
            "You can install it with `pip install pytest` or `pip install xclim[dev]`."
        )

    cmd = [
        f"--rootdir={Path(__file__).absolute().parent}",
        "--numprocesses=0",
        "--xdoctest",
        f"{Path(__file__).absolute().parents[1]}",
    ]

    sys.exit(pytest.main(cmd))


def testing_setup_warnings():
    """Warn users about potential incompatibilities between xclim and xclim-testdata versions."""
    if re.match(r"^\d+\.\d+\.\d+$", __xclim_version__) and TESTDATA_BRANCH != default_testdata_version:
        # This does not need to be emitted on GitHub Workflows and ReadTheDocs
        if not os.getenv("CI") and not os.getenv("READTHEDOCS"):
            warnings.warn(
                f"`xclim` stable ({__xclim_version__}) is running tests against a non-default "
                f"branch of the testing data. It is possible that changes to the testing data may "
                f"be incompatible with some assertions in this version. "
                f"Please be sure to check {TESTDATA_REPO_URL} for more information.",
            )

    if re.match(r"^v\d+\.\d+\.\d+", TESTDATA_BRANCH):
        # Find the date of last modification of xclim source files to generate a calendar version
        install_date = dt.strptime(
            time.ctime(os.path.getmtime(xclim.__file__)),
            "%a %b %d %H:%M:%S %Y",
        )
        install_calendar_version = f"{install_date.year}.{install_date.month}.{install_date.day}"

        if Version(TESTDATA_BRANCH) > Version(install_calendar_version):
            warnings.warn(
                f"The installation date of `xclim` ({install_date.ctime()}) "
                f"predates the last release of testing data ({TESTDATA_BRANCH}). "
                "It is very likely that the testing data is incompatible with this build of `xclim`.",
            )


def load_registry(branch: str = TESTDATA_BRANCH, repo: str = TESTDATA_REPO_URL) -> dict[str, str]:
    """
    Load the registry file for the test data.

    Parameters
    ----------
    branch : str
        Branch of the repository to use when fetching testing datasets.
    repo : str
        URL of the repository to use when fetching testing datasets.

    Returns
    -------
    dict
        Dictionary of filenames and hashes.
    """
    if not repo.endswith("/"):
        repo = f"{repo}/"
    remote_registry = audit_url(
        urljoin(
            urljoin(repo, branch if branch.endswith("/") else f"{branch}/"),
            "data/registry.txt",
        )
    )

    if repo != default_testdata_repo_url:
        external_repo_name = urlparse(repo).path.split("/")[-2]
        external_branch_name = branch.split("/")[-1]
        registry_file = Path(
            str(ilr.files("xclim").joinpath(f"testing/registry.{external_repo_name}.{external_branch_name}.txt"))
        )
        urlretrieve(remote_registry, registry_file)  # noqa: S310

    elif branch != default_testdata_version:
        custom_registry_folder = Path(str(ilr.files("xclim").joinpath(f"testing/{branch}")))
        custom_registry_folder.mkdir(parents=True, exist_ok=True)
        registry_file = custom_registry_folder.joinpath("registry.txt")
        urlretrieve(remote_registry, registry_file)  # noqa: S310

    else:
        registry_file = Path(str(ilr.files("xclim").joinpath("testing/registry.txt")))

    if not registry_file.exists():
        raise FileNotFoundError(f"Registry file not found: {registry_file}")

    # Load the registry file
    with registry_file.open(encoding="utf-8") as f:
        registry = {line.split()[0]: line.split()[1] for line in f}
    return registry


def nimbus(
    repo: str = TESTDATA_REPO_URL,
    branch: str = TESTDATA_BRANCH,
    cache_dir: str | Path = TESTDATA_CACHE_DIR,
    data_updates: bool = True,
):
    """
    Pooch registry instance for xclim test data.

    Parameters
    ----------
    repo : str
        URL of the repository to use when fetching testing datasets.
    branch : str
        Branch of repository to use when fetching testing datasets.
    cache_dir : str or Path
        The path to the directory where the data files are stored.
    data_updates : bool
        If True, allow updates to the data files. Default is True.

    Returns
    -------
    pooch.Pooch
        The Pooch instance for accessing the xclim testing data.

    Notes
    -----
    There are three environment variables that can be used to control the behaviour of this registry:
        - ``XCLIM_TESTDATA_CACHE_DIR``: If this environment variable is set, it will be used as the
          base directory to store the data files.
          The directory should be an absolute path (i.e., it should start with ``/``).
          Otherwise,the default location will be used (based on ``platformdirs``, see :py:func:`pooch.os_cache`).
        - ``XCLIM_TESTDATA_REPO_URL``: If this environment variable is set, it will be used as the URL of
          the repository to use when fetching datasets. Otherwise, the default repository will be used.
        - ``XCLIM_TESTDATA_BRANCH``: If this environment variable is set, it will be used as the branch of
          the repository to use when fetching datasets. Otherwise, the default branch will be used.

    Examples
    --------
    Using the registry to download a file:

    .. code-block:: python

        import xarray as xr
        from xclim.testing.helpers import nimbus

        example_file = nimbus().fetch("example.nc")
        data = xr.open_dataset(example_file)
    """
    if pooch is None:
        raise ImportError(
            "The `pooch` package is required to fetch the xclim testing data. "
            "You can install it with `pip install pooch` or `pip install xclim[dev]`."
        )
    if not repo.endswith("/"):
        repo = f"{repo}/"
    remote = audit_url(urljoin(urljoin(repo, branch if branch.endswith("/") else f"{branch}/"), "data"))

    _nimbus = pooch.create(
        path=cache_dir,
        base_url=remote,
        version=default_testdata_version,
        version_dev=branch,
        allow_updates=data_updates,
        registry=load_registry(branch=branch, repo=repo),
    )

    # Add a custom fetch method to the Pooch instance
    # Needed to address: https://github.com/readthedocs/readthedocs.org/issues/11763
    # Fix inspired by @bjlittle (https://github.com/bjlittle/geovista/pull/1202)
    _nimbus.fetch_diversion = _nimbus.fetch

    # Overload the fetch method to add user-agent headers
    @wraps(_nimbus.fetch_diversion)
    def _fetch(*args, **kwargs: bool | Callable) -> str:  # numpydoc ignore=GL08  # *args: str
        def _downloader(
            url: str,
            output_file: str | IO,
            poocher: pooch.Pooch,
            check_only: bool | None = False,
        ) -> None:
            """Download the file from the URL and save it to the save_path."""
            headers = {"User-Agent": f"xclim ({__xclim_version__})"}
            downloader = pooch.HTTPDownloader(headers=headers)
            return downloader(url, output_file, poocher, check_only=check_only)

        # default to our http/s downloader with user-agent headers
        kwargs.setdefault("downloader", _downloader)
        return _nimbus.fetch_diversion(*args, **kwargs)

    # Replace the fetch method with the custom fetch method
    _nimbus.fetch = _fetch

    return _nimbus


# FIXME: This function is soon to be deprecated.
# idea copied from raven that it borrowed from xclim that borrowed it from xarray that was borrowed from Seaborn
def open_dataset(
    name: str | os.PathLike[str],
    dap_url: str | None = None,
    branch: str = TESTDATA_BRANCH,
    repo: str = TESTDATA_REPO_URL,
    cache_dir: str | os.PathLike[str] | None = TESTDATA_CACHE_DIR,
    **kwargs,
) -> Dataset:
    r"""
    Open a dataset from the online GitHub-like repository.

    If a local copy is found then always use that to avoid network traffic.

    Parameters
    ----------
    name : str
        Name of the file containing the dataset.
    dap_url : str, optional
        URL to OPeNDAP folder where the data is stored. If supplied, supersedes github_url.
    branch : str
        Branch of the repository to use when fetching datasets.
    repo : str
        URL of the repository to use when fetching testing datasets.
    cache_dir : Path
        The directory in which to search for and write cached data.
    **kwargs : dict
        For NetCDF files, keywords passed to :py:func:`xarray.open_dataset`.

    Returns
    -------
    Union[Dataset, Path]
        The dataset.

    Raises
    ------
    OSError
        If the file is not found in the cache directory or cannot be read.

    See Also
    --------
    xarray.open_dataset : Open and read a dataset from a file or file-like object.
    """
    if cache_dir is None:
        raise ValueError(
            "The cache directory must be set. "
            "Please set the `cache_dir` parameter or the `XCLIM_DATA_DIR` environment variable."
        )

    if dap_url:
        dap_target = urljoin(dap_url, str(name))
        try:
            return _open_dataset(audit_url(dap_target, context="OPeNDAP"), **kwargs)
        except URLError:
            raise
        except OSError as err:
            msg = f"OPeNDAP file not read. Verify that the service is available: {dap_target}"
            raise OSError(msg) from err

    local_file = Path(cache_dir).joinpath(name)
    if not local_file.exists():
        try:
            local_file = nimbus(branch=branch, repo=repo, cache_dir=cache_dir).fetch(name)
        except OSError as err:
            msg = f"File not found locally. Verify that the testing data is available in remote: {local_file}"
            raise OSError(msg) from err
    try:
        ds = _open_dataset(local_file, **kwargs)
        return ds
    except OSError:
        raise


def populate_testing_data(
    temp_folder: Path | None = None,
    repo: str = TESTDATA_REPO_URL,
    branch: str = TESTDATA_BRANCH,
    local_cache: Path = TESTDATA_CACHE_DIR,
) -> None:
    """
    Populate the local cache with the testing data.

    Parameters
    ----------
    temp_folder : Path, optional
        Path to a temporary folder to use as the local cache. If not provided, the default location will be used.
    repo : str, optional
        URL of the repository to use when fetching testing datasets.
    branch : str, optional
        Branch of xclim-testdata to use when fetching testing datasets.
    local_cache : Path
        The path to the local cache. Defaults to the location set by the platformdirs library.
        The testing data will be downloaded to this local cache.
    """
    # Create the Pooch instance
    n = nimbus(repo=repo, branch=branch, cache_dir=temp_folder or local_cache)

    # Download the files
    errored_files = []
    for file in load_registry():
        try:
            n.fetch(file)
        except HTTPError:
            msg = f"File `{file}` not accessible in remote repository."
            logging.error(msg)
            errored_files.append(file)
        except SocketBlockedError as err:  # noqa
            msg = (
                "Unable to access registry file online. Testing suite is being run with `--disable-socket`. "
                "If you intend to run tests with this option enabled, please download the file beforehand with the "
                "following console command: `$ xclim prefetch_testing_data`."
            )
            raise SocketBlockedError(msg) from err
        else:
            logging.info("Files were downloaded successfully.")

    if errored_files:
        logging.error(
            "The following files were unable to be downloaded: %s",
            errored_files,
        )


def gather_testing_data(
    worker_cache_dir: str | os.PathLike[str] | Path,
    worker_id: str,
    _cache_dir: str | os.PathLike[str] | None = TESTDATA_CACHE_DIR,
) -> None:
    """
    Gather testing data across workers.

    Parameters
    ----------
    worker_cache_dir : str or Path
        The directory to store the testing data.
    worker_id : str
        The worker ID.
    _cache_dir : str or Path, optional
        The directory to store the testing data. Default is None.

    Raises
    ------
    ValueError
        If the cache directory is not set.
    FileNotFoundError
        If the testing data is not found.
    """
    if _cache_dir is None:
        raise ValueError(
            "The cache directory must be set. "
            "Please set the `cache_dir` parameter or the `XCLIM_DATA_DIR` environment variable."
        )
    cache_dir = Path(_cache_dir)

    if worker_id == "master":
        populate_testing_data(branch=TESTDATA_BRANCH)
    else:
        if platform.system() == "Windows":
            if not cache_dir.joinpath(default_testdata_version).exists():
                raise FileNotFoundError(
                    "Testing data not found and UNIX-style file-locking is not supported on Windows. "
                    "Consider running `$ xclim prefetch_testing_data` to download testing data beforehand."
                )
        else:
            cache_dir.mkdir(exist_ok=True, parents=True)
            lockfile = cache_dir.joinpath(".lock")
            test_data_being_written = FileLock(lockfile)
            with test_data_being_written:
                # This flag prevents multiple calls from re-attempting to download testing data in the same pytest run
                populate_testing_data(branch=TESTDATA_BRANCH)
                cache_dir.joinpath(".data_written").touch()
            with test_data_being_written.acquire():
                if lockfile.exists():
                    lockfile.unlink()
        copytree(cache_dir.joinpath(default_testdata_version), worker_cache_dir)


# Testing Utilities ###


def audit_url(url: str, context: str | None = None) -> str:
    """
    Check if the URL is well-formed.

    Parameters
    ----------
    url : str
        The URL to check.
    context : str, optional
        Additional context to include in the error message. Default is None.

    Returns
    -------
    str
        The URL if it is well-formed.

    Raises
    ------
    URLError
        If the URL is not well-formed.
    """
    msg = ""
    result = urlparse(url)
    if result.scheme == "http":
        msg = f"{context if context else ''} URL is not using secure HTTP: '{url}'".strip()
    if not all([result.scheme, result.netloc]):
        msg = f"{context if context else ''} URL is not well-formed: '{url}'".strip()

    if msg:
        logger.error(msg)
        raise URLError(msg)
    return url

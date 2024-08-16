"""Module for loading testing data."""

from __future__ import annotations

import importlib.resources as ilr
import logging
import os
import re
import shutil
import tempfile
import time
import warnings
from datetime import datetime as dt
from pathlib import Path
from shutil import copytree
from sys import platform
from urllib.error import HTTPError

import numpy as np
import pandas as pd
import pooch
import xarray as xr
from dask.diagnostics import Callback
from filelock import FileLock
from packaging.version import Version

try:
    from pytest_socket import SocketBlockedError
except ImportError:
    SocketBlockedError = None

import xclim
from xclim import __version__ as __xclim_version__
from xclim.core import calendar
from xclim.core.utils import VARIABLES
from xclim.indices import (
    longwave_upwelling_radiation_from_net_downwelling,
    shortwave_upwelling_radiation_from_net_downwelling,
)
from xclim.testing.utils import _default_cache_dir  # noqa
from xclim.testing.utils import open_dataset as _open_dataset

TESTDATA_REPO_URL = str(
    os.getenv("XCLIM_TESTDATA_REPO_URL", "https://github.com/Ouranosinc/xclim-testdata")
)
"""Sets the URL of the testing data repository to use when fetching datasets.

Notes
-----
When running tests locally, this can be set for both `pytest` and `tox` by exporting the variable:

.. code-block:: console

    $ export XCLIM_TESTDATA_REPO_URL="https://github.com/my_username/xclim-testdata"

or setting the variable at runtime:

.. code-block:: console

    $ env XCLIM_TESTDATA_REPO_URL="https://github.com/my_username/xclim-testdata" pytest
"""

TESTDATA_BRANCH = str(os.getenv("XCLIM_TESTDATA_BRANCH", "main"))
"""Sets the branch of the testing data repository to use when fetching datasets.

Notes
-----
When running tests locally, this can be set for both `pytest` and `tox` by exporting the variable:

.. code-block:: console

    $ export XCLIM_TESTDATA_BRANCH="my_testing_branch"

or setting the variable at runtime:

.. code-block:: console

    $ env XCLIM_TESTDATA_BRANCH="my_testing_branch" pytest
"""

PREFETCH_TESTING_DATA = bool(os.getenv("XCLIM_PREFETCH_TESTING_DATA"))
"""Indicates whether the testing data should be downloaded when running tests.

Notes
-----
When running tests multiple times, this flag allows developers to significantly speed up the pytest suite
by preventing sha256sum checks for all downloaded files. Proceed with caution.

This can be set for both `pytest` and `tox` by exporting the variable:

.. code-block:: console

    $ export XCLIM_PREFETCH_TESTING_DATA=1

or setting the variable at runtime:

.. code-block:: console

    $ env XCLIM_PREFETCH_TESTING_DATA=1 pytest
"""

CACHE_DIR = os.getenv("XCLIM_DATA_DIR", _default_cache_dir)
"""Sets the directory to store the testing datasets.

If not set, the default location will be used (based on ``platformdirs``, see :func:`pooch.os_cache`).

Notes
-----
When running tests locally, this can be set for both `pytest` and `tox` by exporting the variable:

.. code-block:: console

    $ export XCLIM_DATA_DIR="/path/to/my/data"

or setting the variable at runtime:

.. code-block:: console

    $ env XCLIM_DATA_DIR="/path/to/my/data" pytest
"""

DATA_UPDATES = bool(os.getenv("XCLIM_DATA_UPDATES"))
"""Sets whether to allow updates to the testing datasets.

If set to ``True``, the data files will be downloaded even if the upstream hashes do not match.

Notes
-----
When running tests locally, this can be set for both `pytest` and `tox` by exporting the variable:

.. code-block:: console

    $ export XCLIM_DATA_UPDATES=True

or setting the variable at runtime:

.. code-block:: console

    $ env XCLIM_DATA_UPDATES=True pytest
"""


DATA_URL = f"{TESTDATA_REPO_URL}/raw/{TESTDATA_BRANCH}/data"

__all__ = [
    "DATA_UPDATES",
    "DATA_URL",
    "PREFETCH_TESTING_DATA",
    "TESTDATA_BRANCH",
    "add_example_file_paths",
    "assert_lazy",
    "generate_atmos",
    "test_timeseries",
]


def testing_setup_warnings():
    """Warn users about potential incompatibilities between xclim and xclim-testdata versions."""
    if re.match(r"^\d+\.\d+\.\d+$", __xclim_version__) and TESTDATA_BRANCH == "main":
        # This does not need to be emitted on GitHub Workflows and ReadTheDocs
        if not os.getenv("CI") and not os.getenv("READTHEDOCS"):
            warnings.warn(
                f'`xclim` {__xclim_version__} is running tests against the "main" branch of the testing data. '
                "It is possible that changes to the testing data may be incompatible with some assertions in this version. "
                f"Please be sure to check {TESTDATA_REPO_URL} for more information.",
            )

    if re.match(r"^v\d+\.\d+\.\d+", TESTDATA_BRANCH):
        # Find the date of last modification of xclim source files to generate a calendar version
        install_date = dt.strptime(
            time.ctime(os.path.getmtime(xclim.__file__)),
            "%a %b %d %H:%M:%S %Y",
        )
        install_calendar_version = (
            f"{install_date.year}.{install_date.month}.{install_date.day}"
        )

        if Version(TESTDATA_BRANCH) > Version(install_calendar_version):
            warnings.warn(
                f"The installation date of `xclim` ({install_date.ctime()}) predates the last release of testing data ({TESTDATA_BRANCH}). "
                "It is very likely that the testing data is incompatible with this build of `xclim`.",
            )


def load_registry(
    file: str | Path | None = None, remote: str = DATA_URL
) -> dict[str, str]:
    """Load the registry file for the test data.

    Parameters
    ----------
    file : str or Path, optional
        Path to the registry file. If not provided, the registry file found within the package data will be used.
    remote : str
        URL to the remote registry folder.

    Returns
    -------
    dict
        Dictionary of filenames and hashes.
    """

    def _fetcher(f: str, r: str, c: str) -> str:
        try:
            return pooch.retrieve(
                url=f"{r}/{f}",
                known_hash=None,
                path=c,
                fname="registry.txt",
            )
        except HTTPError:
            raise
        except SocketBlockedError:
            raise

    # Get registry file from package_data
    if file is None:
        registry_file = Path(str(ilr.files("xclim").joinpath("testing/registry.txt")))
        if not registry_file.exists():
            registry_file.touch()
        try:
            with tempfile.TemporaryDirectory() as tempdir:
                remote_registry_file = _fetcher(registry_file.name, remote, tempdir)
                # Check if the local registry file matches the remote registry
                if pooch.file_hash(remote_registry_file) != pooch.file_hash(
                    registry_file.as_posix()
                ):
                    warnings.warn(
                        "Local registry file does not match remote registry file."
                    )
                    shutil.move(remote_registry_file, registry_file)
        except FileNotFoundError:
            warnings.warn(
                "Registry file not accessible in remote repository. "
                "Aborting file retrieval and using local registry file."
            )
        except SocketBlockedError:
            warnings.warn(
                "Testing suite is being run with `--disable-socket`. Using local registry file."
            )
        if not registry_file.exists():
            raise FileNotFoundError(
                f"Local registry file not found: {registry_file}. "
                "Testing setup cannot proceed without registry file."
            )
    else:
        registry_file = Path(file)
        if not registry_file.exists():
            raise FileNotFoundError(f"Registry file not found: {registry_file}")

    logging.info("Registry file found: %s", registry_file)

    # Load the registry file
    registry = dict()
    with registry_file.open() as buffer:
        for entry in buffer.readlines():
            registry[entry.split()[0]] = entry.split()[1]

    return registry


def nimbus(  # noqa: PR01
    data_dir: str | Path = CACHE_DIR,
    data_updates: bool = DATA_UPDATES,
    data_url: str = DATA_URL,
):
    """Pooch registry instance for xhydro test data.

    Parameters
    ----------
    data_dir : str or Path
        Path to the directory where the data files are stored.
    data_updates : bool
        If True, allow updates to the data files.
    data_url : str
        Base URL to download the data files.

    Returns
    -------
    pooch.Pooch
        Pooch instance for the xhydro test data.

    Notes
    -----
    There are three environment variables that can be used to control the behaviour of this registry:
        - ``XCLIM_DATA_DIR``: If this environment variable is set, it will be used as the base directory to store the data
          files. The directory should be an absolute path (i.e., it should start with ``/``). Otherwise,
          the default location will be used (based on ``platformdirs``, see :py:func:`pooch.os_cache`).
        - ``XCLIM_DATA_UPDATES``: If this environment variable is set, then the data files will be downloaded even if the
          upstream hashes do not match. This is useful if you want to always use the latest version of the data files.
        - ``XCLIM_DATA_URL``: If this environment variable is set, it will be used as the base URL to download the data files.

    Examples
    --------
    Using the registry to download a file:

    .. code-block:: python

        import xarray as xr
        from xclim.testing.helpers import nimbus

        example_file = nimbus().fetch("example.nc")
        data = xr.open_dataset(example_file)
    """
    return pooch.create(
        path=data_dir,
        base_url=data_url,
        version=__xclim_version__,
        version_dev="main",
        allow_updates=data_updates,
        registry=load_registry(remote=data_url),
    )


def populate_testing_data(
    registry_file: str | Path | None = None,
    temp_folder: Path | None = None,
    branch: str = TESTDATA_BRANCH,
    _data_url: str = DATA_URL,
    _local_cache: Path = _default_cache_dir,
) -> None:
    """Populate the local cache with the testing data.

    Parameters
    ----------
    registry_file : str or Path, optional
        Path to the registry file. If not provided, the registry file from package_data will be used.
    temp_folder : Path, optional
        Path to a temporary folder to use as the local cache. If not provided, the default location will be used.
    branch : str, optional
        Branch of hydrologie/xhydro-testdata to use when fetching testing datasets.
    _data_url : Path
        URL for the testing data.
        Set via the `DATA_URL` environment variable ({TESTDATA_REPO_URL}/raw/{TESTDATA_BRANCH}/data).
    _local_cache : Path
        Path to the local cache. Defaults to the location set by the platformdirs library.
        The testing data will be downloaded to this local cache.

    Returns
    -------
    None
    """
    # Get registry file from package_data or provided path
    registry = load_registry(registry_file)
    # Set the local cache to the temp folder
    if temp_folder is not None:
        _local_cache = temp_folder

    # Create the Pooch instance
    n = nimbus(data_url=_data_url)

    # Set the branch
    n.version_dev = branch
    # Set the local cache
    n.path = _local_cache

    # Download the files
    errored_files = []
    for file in registry.keys():
        try:
            n.fetch(file)
        except HTTPError:
            msg = f"File `{file}` not accessible in remote repository."
            logging.error(msg)
            errored_files.append(file)
        except SocketBlockedError as e:
            msg = (
                "Unable to access registry file online. Testing suite is being run with `--disable-socket`. "
                "If you intend to run tests with this option enabled, please download the file beforehand with the "
                "following console command: `$ xclim prefetch_testing_data`."
            )
            raise SocketBlockedError(msg) from e
        else:
            logging.info("Files were downloaded successfully.")
        finally:
            if errored_files:
                logging.error(
                    "The following files were unable to be downloaded: %s",
                    errored_files,
                )


def generate_atmos(cache_dir: Path) -> dict[str, xr.DataArray]:
    """Create the `atmosds` synthetic testing dataset."""
    with _open_dataset(
        "ERA5/daily_surface_cancities_1990-1993.nc",
        cache_dir=cache_dir,
        branch=TESTDATA_BRANCH,
        engine="h5netcdf",
    ) as ds:
        tn10 = calendar.percentile_doy(ds.tasmin, per=10)
        t10 = calendar.percentile_doy(ds.tas, per=10)
        t90 = calendar.percentile_doy(ds.tas, per=90)
        tx90 = calendar.percentile_doy(ds.tasmax, per=90)

        rsus = shortwave_upwelling_radiation_from_net_downwelling(ds.rss, ds.rsds)
        rlus = longwave_upwelling_radiation_from_net_downwelling(ds.rls, ds.rlds)

        ds = ds.assign(
            rsus=rsus,
            rlus=rlus,
            tn10=tn10,
            t10=t10,
            t90=t90,
            tx90=tx90,
        )

        # Create a file in session scoped temporary directory
        atmos_file = cache_dir.joinpath("atmosds.nc")
        ds.to_netcdf(atmos_file, engine="h5netcdf")

    # Give access to dataset variables by name in namespace
    namespace = dict()
    with _open_dataset(
        atmos_file, branch=TESTDATA_BRANCH, cache_dir=cache_dir, engine="h5netcdf"
    ) as ds:
        for variable in ds.data_vars:
            namespace[f"{variable}_dataset"] = ds.get(variable)
    return namespace


def gather_testing_data(threadsafe_data_dir: Path, worker_id: str):
    """Gather testing data across workers."""
    if (
        not _default_cache_dir.joinpath(TESTDATA_BRANCH).exists()
        or PREFETCH_TESTING_DATA
    ):
        if PREFETCH_TESTING_DATA:
            print("`XCLIM_PREFETCH_TESTING_DATA` set. Prefetching testing data...")
        if platform == "win32":
            raise OSError(
                "UNIX-style file-locking is not supported on Windows. "
                "Consider running `$ xclim prefetch_testing_data` to download testing data."
            )
        elif worker_id in ["master"]:
            populate_testing_data(branch=TESTDATA_BRANCH)
        else:
            _default_cache_dir.mkdir(exist_ok=True, parents=True)
            lockfile = _default_cache_dir.joinpath(".lock")
            test_data_being_written = FileLock(lockfile)
            with test_data_being_written:
                # This flag prevents multiple calls from re-attempting to download testing data in the same pytest run
                populate_testing_data(branch=TESTDATA_BRANCH)
                _default_cache_dir.joinpath(".data_written").touch()
            with test_data_being_written.acquire():
                if lockfile.exists():
                    lockfile.unlink()
    copytree(_default_cache_dir, threadsafe_data_dir)


def add_example_file_paths() -> dict[str, str | list[xr.DataArray]]:
    """Create a dictionary of relevant datasets to be patched into the xdoctest namespace."""
    namespace: dict = dict()
    namespace["path_to_ensemble_file"] = "EnsembleReduce/TestEnsReduceCriteria.nc"
    namespace["path_to_pr_file"] = "NRCANdaily/nrcan_canada_daily_pr_1990.nc"
    namespace["path_to_sfcWind_file"] = "ERA5/daily_surface_cancities_1990-1993.nc"
    namespace["path_to_tas_file"] = "ERA5/daily_surface_cancities_1990-1993.nc"
    namespace["path_to_tasmax_file"] = "NRCANdaily/nrcan_canada_daily_tasmax_1990.nc"
    namespace["path_to_tasmin_file"] = "NRCANdaily/nrcan_canada_daily_tasmin_1990.nc"

    # For core.utils.load_module example
    namespace["path_to_example_py"] = (
        Path(__file__).parent.parent.parent.parent / "docs" / "notebooks" / "example.py"
    )

    time = xr.cftime_range("1990-01-01", "2049-12-31", freq="D")
    namespace["temperature_datasets"] = [
        xr.DataArray(
            12 * np.random.random_sample(time.size) + 273,
            coords={"time": time},
            name="tas",
            dims=("time",),
            attrs={
                "units": "K",
                "cell_methods": "time: mean within days",
                "standard_name": "air_temperature",
            },
        ),
        xr.DataArray(
            12 * np.random.random_sample(time.size) + 273,
            coords={"time": time},
            name="tas",
            dims=("time",),
            attrs={
                "units": "K",
                "cell_methods": "time: mean within days",
                "standard_name": "air_temperature",
            },
        ),
    ]

    return namespace


def add_doctest_filepaths():
    """Add filepaths to the xdoctest namespace."""
    namespace: dict = dict()
    namespace["np"] = np
    namespace["xclim"] = xclim
    namespace["tas"] = test_timeseries(
        np.random.rand(365) * 20 + 253.15, variable="tas"
    )
    namespace["pr"] = test_timeseries(np.random.rand(365) * 5, variable="pr")

    return namespace


def test_timeseries(
    values,
    variable,
    start: str = "2000-07-01",
    units: str | None = None,
    freq: str = "D",
    as_dataset: bool = False,
    cftime: bool = False,
) -> xr.DataArray | xr.Dataset:
    """Create a generic timeseries object based on pre-defined dictionaries of existing variables."""
    if cftime:
        coords = xr.cftime_range(start, periods=len(values), freq=freq)
    else:
        coords = pd.date_range(start, periods=len(values), freq=freq)

    if variable in VARIABLES:
        attrs = {
            a: VARIABLES[variable].get(a, "")
            for a in ["description", "standard_name", "cell_methods"]
        }
        attrs["units"] = VARIABLES[variable]["canonical_units"]

    else:
        warnings.warn(f"Variable {variable} not recognised. Attrs will not be filled.")
        attrs = {}

    if units is not None:
        attrs["units"] = units

    da = xr.DataArray(values, coords=[coords], dims="time", name=variable, attrs=attrs)

    if as_dataset:
        return da.to_dataset()
    else:
        return da


def _raise_on_compute(dsk: dict):
    """Raise an AssertionError mentioning the number triggered tasks."""
    raise AssertionError(
        f"Not lazy. Computation was triggered with a graph of {len(dsk)} tasks."
    )


assert_lazy = Callback(start=_raise_on_compute)
"""Context manager that raises an AssertionError if any dask computation is triggered."""

"""Module for loading testing data."""

from __future__ import annotations

import importlib.resources as ilr
import logging
import os
import re
import time
import warnings
from datetime import datetime as dt
from pathlib import Path
from shutil import copytree
from sys import platform
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlparse

import numpy as np
import pandas as pd
import xarray as xr
from dask.callbacks import Callback
from filelock import FileLock
from packaging.version import Version
from xarray import Dataset
from xarray import open_dataset as _open_dataset

try:
    from pytest_socket import SocketBlockedError
except ImportError:
    SocketBlockedError = None

try:
    import pooch
except ImportError:
    warnings.warn(
        "The `pooch` library is not installed. "
        "The default cache directory for testing data will not be set."
    )
    pooch = None

import xclim
from xclim import __version__ as __xclim_version__
from xclim.core import calendar
from xclim.core.utils import VARIABLES
from xclim.indices import (
    longwave_upwelling_radiation_from_net_downwelling,
    shortwave_upwelling_radiation_from_net_downwelling,
)

logger = logging.getLogger("xclim")

default_testdata_version = "v2023.12.14"
"""Default version of the testing data to use when fetching datasets."""

try:
    default_cache_dir = Path(pooch.os_cache("xclim-testdata"))
    """Default location for the testing data cache."""
except AttributeError:
    default_cache_dir = None

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

TESTDATA_BRANCH = str(os.getenv("XCLIM_TESTDATA_BRANCH", default_testdata_version))
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

CACHE_DIR = os.getenv("XCLIM_DATA_DIR", default_cache_dir)
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


__all__ = [
    "CACHE_DIR",
    "TESTDATA_BRANCH",
    "TESTDATA_REPO_URL",
    "add_example_file_paths",
    "assert_lazy",
    "default_cache_dir",
    "default_testdata_version",
    "generate_atmos",
    "nimbus",
    "open_dataset",
    "populate_testing_data",
    "test_timeseries",
]


def testing_setup_warnings():
    """Warn users about potential incompatibilities between xclim and xclim-testdata versions."""
    if (
        re.match(r"^\d+\.\d+\.\d+$", __xclim_version__)
        and TESTDATA_BRANCH != default_testdata_version
    ):
        # This does not need to be emitted on GitHub Workflows and ReadTheDocs
        if not os.getenv("CI") and not os.getenv("READTHEDOCS"):
            warnings.warn(
                f"`xclim` stable ({__xclim_version__}) is running tests against a non-default branch of the testing data. "
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
                f"The installation date of `xclim` ({install_date.ctime()}) "
                f"predates the last release of testing data ({TESTDATA_BRANCH}). "
                "It is very likely that the testing data is incompatible with this build of `xclim`.",
            )


def load_registry() -> dict[str, str]:
    """Load the registry file for the test data.

    Returns
    -------
    dict
        Dictionary of filenames and hashes.
    """
    registry_file = Path(str(ilr.files("xclim").joinpath("testing/registry.txt")))
    if not registry_file.exists():
        raise FileNotFoundError(f"Registry file not found: {registry_file}")

    # Load the registry file
    with registry_file.open() as f:
        registry = {line.split()[0]: line.split()[1] for line in f}
    return registry


def nimbus(  # noqa: PR01
    data_dir: str | Path = CACHE_DIR,
    repo: str = TESTDATA_REPO_URL,
    branch: str = TESTDATA_BRANCH,
    data_updates: bool = True,
):
    """Pooch registry instance for xclim test data.

    Parameters
    ----------
    data_dir : str or Path
        Path to the directory where the data files are stored.
    repo : str
        URL of the repository to use when fetching testing datasets.
    branch : str
        Branch of repository to use when fetching testing datasets.
    data_updates : bool
        If True, allow updates to the data files. Default is True.

    Returns
    -------
    pooch.Pooch
        The Pooch instance for accessing the xclim testing data.

    Notes
    -----
    There are three environment variables that can be used to control the behaviour of this registry:
        - ``XCLIM_DATA_DIR``: If this environment variable is set, it will be used as the base directory to store the data
          files. The directory should be an absolute path (i.e., it should start with ``/``). Otherwise,
          the default location will be used (based on ``platformdirs``, see :py:func:`pooch.os_cache`).
        - ``XCLIM_TESTDATA_REPO_URL``: If this environment variable is set, it will be used as the URL of the repository
          to use when fetching datasets. Otherwise, the default repository will be used.
        - ``XCLIM_TESTDATA_BRANCH``: If this environment variable is set, it will be used as the branch of the repository
          to use when fetching datasets. Otherwise, the default branch will be used.

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

    remote = f"{repo}/raw/{branch}/data"
    return pooch.create(
        path=data_dir,
        base_url=remote,
        version=default_testdata_version,
        version_dev=branch,
        allow_updates=data_updates,
        registry=load_registry(),
    )


# idea copied from raven that it borrowed from xclim that borrowed it from xarray that was borrowed from Seaborn
def open_dataset(
    name: str | os.PathLike[str],
    dap_url: str | None = None,
    cache_dir: str | os.PathLike[str] | None = CACHE_DIR,
    **kwargs,
) -> Dataset:
    r"""Open a dataset from the online GitHub-like repository.

    If a local copy is found then always use that to avoid network traffic.

    Parameters
    ----------
    name : str
        Name of the file containing the dataset.
    dap_url : str, optional
        URL to OPeNDAP folder where the data is stored. If supplied, supersedes github_url.
    cache_dir : Path
        The directory in which to search for and write cached data.
    \*\*kwargs
        For NetCDF files, keywords passed to :py:func:`xarray.open_dataset`.

    Returns
    -------
    Union[Dataset, Path]

    See Also
    --------
    xarray.open_dataset
    """
    if cache_dir is None:
        raise ValueError(
            "The cache directory must be set. "
            "Please set the `cache_dir` parameter or the `XCLIM_DATA_DIR` environment variable."
        )

    if dap_url:
        try:
            return _open_dataset(
                audit_url(urljoin(dap_url, str(name)), context="OPeNDAP"), **kwargs
            )
        except (OSError, URLError):
            msg = f"OPeNDAP file not read. Verify that the service is available: '{urljoin(dap_url, str(name))}'"
            logger.error(msg)
            raise

    local_file = Path(cache_dir).joinpath(name)
    if not local_file.exists():
        raise OSError(f"File not found: {local_file}")
    try:
        ds = _open_dataset(local_file, **kwargs)
        return ds
    except OSError as err:
        raise err


def populate_testing_data(
    temp_folder: Path | None = None,
    repo: str = TESTDATA_REPO_URL,
    branch: str = TESTDATA_BRANCH,
    local_cache: Path = CACHE_DIR,
) -> None:
    """Populate the local cache with the testing data.

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

    Returns
    -------
    None
    """
    # Create the Pooch instance
    n = nimbus(data_dir=temp_folder or local_cache, repo=repo, branch=branch)

    # Download the files
    errored_files = []
    for file in load_registry():
        try:
            n.fetch(file)
        except HTTPError:
            msg = f"File `{file}` not accessible in remote repository."
            logging.error(msg)
            errored_files.append(file)
        except SocketBlockedError as e:  # noqa
            msg = (
                "Unable to access registry file online. Testing suite is being run with `--disable-socket`. "
                "If you intend to run tests with this option enabled, please download the file beforehand with the "
                "following console command: `$ xclim prefetch_testing_data`."
            )
            raise SocketBlockedError(msg) from e
        else:
            logging.info("Files were downloaded successfully.")

    if errored_files:
        logging.error(
            "The following files were unable to be downloaded: %s",
            errored_files,
        )


def generate_atmos(cache_dir: str | os.PathLike[str] | Path) -> dict[str, xr.DataArray]:
    """Create the `atmosds` synthetic testing dataset."""
    with open_dataset(
        "ERA5/daily_surface_cancities_1990-1993.nc",
        cache_dir=cache_dir,
        engine="h5netcdf",
    ) as ds:
        rsus = shortwave_upwelling_radiation_from_net_downwelling(ds.rss, ds.rsds)
        rlus = longwave_upwelling_radiation_from_net_downwelling(ds.rls, ds.rlds)
        tn10 = calendar.percentile_doy(ds.tasmin, per=10)
        t10 = calendar.percentile_doy(ds.tas, per=10)
        t90 = calendar.percentile_doy(ds.tas, per=90)
        tx90 = calendar.percentile_doy(ds.tasmax, per=90)

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
    with open_dataset(atmos_file, cache_dir=cache_dir, engine="h5netcdf") as ds:
        namespace = {f"{var}_dataset": ds[var] for var in ds.data_vars}
    return namespace


def gather_testing_data(
    threadsafe_data_dir: str | os.PathLike[str] | Path,
    worker_id: str,
    cache_dir: str | os.PathLike[str] | None = CACHE_DIR,
):
    """Gather testing data across workers."""
    if cache_dir is None:
        raise ValueError(
            "The cache directory must be set. "
            "Please set the `cache_dir` parameter or the `XCLIM_DATA_DIR` environment variable."
        )
    cache_dir = Path(cache_dir)

    if worker_id == "master":
        populate_testing_data(branch=TESTDATA_BRANCH)
    else:
        if platform == "win32":
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
        copytree(cache_dir.joinpath(default_testdata_version), threadsafe_data_dir)


def add_ensemble_dataset_objects() -> dict[str, str]:
    namespace = {
        "nc_files_simple": [
            "EnsembleStats/BCCAQv2+ANUSPLIN300_ACCESS1-0_historical+rcp45_r1i1p1_1950-2100_tg_mean_YS.nc",
            "EnsembleStats/BCCAQv2+ANUSPLIN300_BNU-ESM_historical+rcp45_r1i1p1_1950-2100_tg_mean_YS.nc",
            "EnsembleStats/BCCAQv2+ANUSPLIN300_CCSM4_historical+rcp45_r1i1p1_1950-2100_tg_mean_YS.nc",
            "EnsembleStats/BCCAQv2+ANUSPLIN300_CCSM4_historical+rcp45_r2i1p1_1950-2100_tg_mean_YS.nc",
        ],
        "nc_files_extra": [
            "EnsembleStats/BCCAQv2+ANUSPLIN300_CNRM-CM5_historical+rcp45_r1i1p1_1970-2050_tg_mean_YS.nc"
        ],
    }
    namespace["nc_files"] = namespace["nc_files_simple"] + namespace["nc_files_extra"]
    return namespace


def add_example_file_paths() -> dict[str, str | list[xr.DataArray]]:
    """Create a dictionary of relevant datasets to be patched into the xdoctest namespace."""
    namespace = {
        "path_to_ensemble_file": "EnsembleReduce/TestEnsReduceCriteria.nc",
        "path_to_pr_file": "NRCANdaily/nrcan_canada_daily_pr_1990.nc",
        "path_to_sfcWind_file": "ERA5/daily_surface_cancities_1990-1993.nc",
        "path_to_tas_file": "ERA5/daily_surface_cancities_1990-1993.nc",
        "path_to_tasmax_file": "NRCANdaily/nrcan_canada_daily_tasmax_1990.nc",
        "path_to_tasmin_file": "NRCANdaily/nrcan_canada_daily_tasmin_1990.nc",
        "path_to_example_py": (
            Path(__file__).parent.parent.parent.parent
            / "docs"
            / "notebooks"
            / "example.py"
        ),
    }

    # For core.utils.load_module example
    sixty_years = xr.cftime_range("1990-01-01", "2049-12-31", freq="D")
    namespace["temperature_datasets"] = [
        xr.DataArray(
            12 * np.random.random_sample(sixty_years.size) + 273,
            coords={"time": sixty_years},
            name="tas",
            dims=("time",),
            attrs={
                "units": "K",
                "cell_methods": "time: mean within days",
                "standard_name": "air_temperature",
            },
        ),
        xr.DataArray(
            12 * np.random.random_sample(sixty_years.size) + 273,
            coords={"time": sixty_years},
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


def add_doctest_filepaths() -> dict[str, Any]:
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


def audit_url(url: str, context: str | None = None) -> str:
    """Check if the URL is well-formed.

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

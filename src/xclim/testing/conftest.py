"""Specialized setup for running xclim doctests."""

# This file is the setup for the doctest suite.
# This must be run using the following commands:
# python -c "from xclim.testing.utils import run_doctests; run_doctests()"

from __future__ import annotations

import os
from pathlib import Path

import pytest

from xclim.testing.helpers import (
    add_doctest_filepaths,
    add_example_file_paths,
    generate_atmos,
)
from xclim.testing.utils import (
    TESTDATA_BRANCH,
    TESTDATA_CACHE_DIR,
    TESTDATA_REPO_URL,
    gather_testing_data,
    testing_setup_warnings,
)
from xclim.testing.utils import nimbus as _nimbus
from xclim.testing.utils import open_dataset as _open_dataset


@pytest.fixture(autouse=True, scope="session")
def threadsafe_data_dir(tmp_path_factory):  # numpydoc ignore=PR01
    """
    Return a threadsafe temporary directory for storing testing data.

    Yields
    ------
    Path
        The path to the temporary directory.
    """
    yield Path(tmp_path_factory.getbasetemp().joinpath("data"))


@pytest.fixture(scope="session")
def nimbus(threadsafe_data_dir, worker_id):  # numpydoc ignore=PR01
    """
    Return a nimbus object for the test data.

    Returns
    -------
    nimbus
        An preconfigured pooch object.
    """
    return _nimbus(
        repo=TESTDATA_REPO_URL,
        branch=TESTDATA_BRANCH,
        cache_dir=(TESTDATA_CACHE_DIR if worker_id == "master" else threadsafe_data_dir),
    )


@pytest.fixture(scope="session")
def open_dataset(nimbus):  # numpydoc ignore=PR01
    """
    Return a function that opens a dataset from the test data.

    Returns
    -------
    function
        A function that opens a dataset from the test data.
    """

    def _open_session_scoped_file(file: str | os.PathLike, **xr_kwargs):
        xr_kwargs.setdefault("cache", True)
        xr_kwargs.setdefault("engine", "h5netcdf")
        return _open_dataset(
            file,
            branch=TESTDATA_BRANCH,
            repo=TESTDATA_REPO_URL,
            cache_dir=nimbus.path,
            **xr_kwargs,
        )

    return _open_session_scoped_file


@pytest.fixture(scope="session", autouse=True)
def is_matplotlib_installed(xdoctest_namespace) -> None:  # numpydoc ignore=PR01
    """Skip tests that require matplotlib if it is not installed."""

    def _is_matplotlib_installed():
        try:
            import matplotlib  # noqa: F401

        except ImportError:
            pytest.skip("This doctest requires matplotlib to be installed.")

    xdoctest_namespace["is_matplotlib_installed"] = _is_matplotlib_installed


@pytest.fixture(scope="session", autouse=True)
def doctest_setup(xdoctest_namespace, nimbus, worker_id, open_dataset) -> None:  # numpydoc ignore=PR01
    """Gather testing data on doctest run."""
    testing_setup_warnings()
    gather_testing_data(worker_cache_dir=nimbus.path, worker_id=worker_id)
    xdoctest_namespace.update(generate_atmos(branch=TESTDATA_BRANCH, cache_dir=nimbus.path))

    class AttrDict(dict):  # numpydoc ignore=PR01
        """A dictionary that allows access to its keys as attributes."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.__dict__ = self

    xdoctest_namespace["open_dataset"] = open_dataset
    xdoctest_namespace["xr"] = AttrDict()
    xdoctest_namespace["xr"].update({"open_dataset": open_dataset})
    xdoctest_namespace.update(add_doctest_filepaths())
    xdoctest_namespace.update(add_example_file_paths())

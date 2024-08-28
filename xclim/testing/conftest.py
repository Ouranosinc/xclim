# noqa: D100
# This file is the setup for the doctest suite.
# This must be run using the following commands:
# python -c "from xclim.testing.utils import run_doctests; run_doctests()"

from __future__ import annotations

import os
from pathlib import Path

import pytest

from xclim.testing import helpers
from xclim.testing.helpers import open_dataset as _open_dataset


@pytest.fixture(autouse=True, scope="session")
def threadsafe_data_dir(tmp_path_factory):
    """Return a threadsafe temporary directory for storing testing data."""
    yield Path(tmp_path_factory.getbasetemp().joinpath("data"))


@pytest.fixture(scope="session")
def nimbus(threadsafe_data_dir, worker_id):
    """Return a nimbus object for the test data."""
    return helpers.nimbus(
        repo=helpers.TESTDATA_REPO_URL,
        branch=helpers.TESTDATA_BRANCH,
        cache_dir=(
            helpers.TESTDATA_CACHE_DIR if worker_id == "master" else threadsafe_data_dir
        ),
    )


@pytest.fixture(scope="session")
def open_dataset(nimbus):
    """Return a function that opens a dataset from the test data."""

    def _open_session_scoped_file(file: str | os.PathLike, **xr_kwargs):
        xr_kwargs.setdefault("cache", True)
        xr_kwargs.setdefault("engine", "h5netcdf")
        return _open_dataset(
            file,
            branch=helpers.TESTDATA_BRANCH,
            repo=helpers.TESTDATA_REPO_URL,
            cache_dir=nimbus.path,
            **xr_kwargs,
        )

    return _open_session_scoped_file


@pytest.fixture(scope="session", autouse=True)
def is_matplotlib_installed(xdoctest_namespace) -> None:
    """Skip tests that require matplotlib if it is not installed."""

    def _is_matplotlib_installed():
        try:
            import matplotlib  # noqa

            return
        except ImportError:
            return pytest.skip("This doctest requires matplotlib to be installed.")

    xdoctest_namespace["is_matplotlib_installed"] = _is_matplotlib_installed


@pytest.fixture(scope="session", autouse=True)
def doctest_setup(xdoctest_namespace, nimbus, worker_id, open_dataset) -> None:
    """Gather testing data on doctest run."""
    helpers.testing_setup_warnings()
    helpers.gather_testing_data(worker_cache_dir=nimbus.path, worker_id=worker_id)
    xdoctest_namespace.update(helpers.generate_atmos(cache_dir=nimbus.path))

    class AttrDict(dict):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.__dict__ = self

    xdoctest_namespace["open_dataset"] = open_dataset
    xdoctest_namespace["xr"] = AttrDict()
    xdoctest_namespace["xr"].update({"open_dataset": open_dataset})
    xdoctest_namespace.update(helpers.add_doctest_filepaths())
    xdoctest_namespace.update(helpers.add_example_file_paths())

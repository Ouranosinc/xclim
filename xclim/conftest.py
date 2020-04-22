# Use this to inject objects into the doctest namespace.
from pathlib import Path

import numpy
import pytest
import xarray

import xclim

TD = Path(__file__).parent.parent / "tests" / "testdata"


@pytest.fixture(autouse=True)
def add_imports(doctest_namespace):
    ns = doctest_namespace
    ns["np"] = numpy
    ns["xr"] = xarray
    ns["xclim"] = xclim


@pytest.fixture(autouse=True)
def add_example_file_paths(doctest_namespace):
    ns = doctest_namespace
    ns["path_to_pr_file"] = str(
        TD
        / "HadGEM2-CC_360day"
        / "pr_day_HadGEM2-CC_rcp85_r1i1p1_na10kgrid_qm-moving-50bins-detrend_2095.nc"
    )

    ns["path_to_tasmax_file"] = str(
        TD
        / "HadGEM2-CC_360day"
        / "tasmax_day_HadGEM2-CC_rcp85_r1i1p1_na10kgrid_qm-moving-50bins-detrend_2095.nc"
    )

    ns["path_to_tasmin_file"] = str(
        TD
        / "HadGEM2-CC_360day"
        / "tasmin_day_HadGEM2-CC_rcp85_r1i1p1_na10kgrid_qm-moving-50bins-detrend_2095.nc"
    )

    ns["path_to_tas_file"] = str(
        TD / "cmip3" / "tas.sresb1.giss_model_e_r.run1.atm.da.nc"
    )

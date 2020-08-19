import os

import numpy as np
import pytest
import xarray as xr

from xclim.sdba.loess import (
    _constant_regression,
    _gaussian_weighting,
    _linear_regression,
    _loess_nb,
    _tricube_weighting,
    loess_smoothing,
)

TESTS_HOME = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TESTS_DATA = os.path.join(TESTS_HOME, "testdata")


@pytest.mark.parametrize(
    "d,f,w,n,exp",
    [
        (0, 0.2, _tricube_weighting, 1, [-0.0698081, -0.3623449]),
        (0, 0.2, _tricube_weighting, 2, [-0.0679962, -0.3426567]),
        (1, 0.2, _tricube_weighting, 1, [-0.0698081, -0.8652001]),
        (1, 0.2, _tricube_weighting, 4, [-0.0691396, -0.9155697]),
        (1, 0.4, _gaussian_weighting, 2, [0.00287228, -0.4469015]),
    ],
)
def test_loess_nb(d, f, w, n, exp):
    regfun = {0: _constant_regression, 1: _linear_regression}[d]
    x = np.linspace(0, 1, num=100)
    y = np.sin(x * np.pi * 10)
    ys = _loess_nb(x, y, f=f, reg_func=regfun, weight_func=w, niter=n)

    assert np.isclose(ys[50], exp[0])
    assert np.isclose(ys[-1], exp[1])


@pytest.mark.parametrize("use_dask", [True, False])
def test_loess_smoothing(use_dask):
    tas = xr.open_dataset(
        os.path.join(TESTS_DATA, "cmip3/tas.sresb1.giss_model_e_r.run1.atm.da.nc"),
        chunks={"lat": 1} if use_dask else None,
    ).tas.isel(lon=0, time=slice(0, 730))

    tasmooth = loess_smoothing(tas)

    assert np.isclose(tasmooth.isel(lat=0, time=0), 265.76342659)

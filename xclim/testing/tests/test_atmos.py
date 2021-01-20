# This File is for testing atmos indicators that fit neither in test_precip or test_temperature
# Expected values might be the same as for the indices tests, see test_indices comments.
import numpy as np
import xarray as xr

from xclim import atmos

K2C = 273.16


def test_wind_speed_from_vectors():
    uas = xr.DataArray(np.array([3.0, -3.0]), dims=["x"])
    uas.attrs["units"] = "m s-1"
    vas = xr.DataArray(np.array([4.0, -4.0]), dims=["x"])
    vas.attrs["units"] = "m s-1"

    wind, winddir = atmos.wind_speed_from_vector(uas=uas, vas=vas)
    np.testing.assert_allclose(wind, [5.0, 5.0])
    np.testing.assert_allclose(winddir, [216.86989764584402, 36.86989764584402])

    # missing values
    uas[0] = np.nan
    wind, winddir = atmos.wind_speed_from_vector(uas=uas, vas=vas)
    np.testing.assert_array_equal(wind.isnull(), [True, False])
    np.testing.assert_array_equal(winddir.isnull(), [True, False])

    # Calm thresh and northerly
    uas[:] = 0
    vas[0] = 0.9
    vas[1] = -1.1
    wind, winddir = atmos.wind_speed_from_vector(
        uas=uas, vas=vas, calm_wind_thresh="1 m/s"
    )
    np.testing.assert_array_equal(wind, [0.9, 1.1])
    np.testing.assert_allclose(winddir, [0.0, 360.0])


def test_wind_vector_from_speed():
    sfcWind = xr.DataArray(np.array([3.0, 5.0, 0.2]), dims=["x"])
    sfcWind.attrs["units"] = "m s-1"
    sfcWindfromdir = xr.DataArray(np.array([360.0, 36.86989764584402, 0.0]), dims=["x"])
    sfcWindfromdir.attrs["units"] = "degree"

    uas, vas = atmos.wind_vector_from_speed(
        sfcWind=sfcWind, sfcWindfromdir=sfcWindfromdir
    )
    np.testing.assert_allclose(uas, [0.0, -3.0, 0.0], atol=1e-14)
    np.testing.assert_allclose(vas, [-3.0, -4.0, -0.2], atol=1e-14)

    # missing values
    sfcWind[0] = np.nan
    sfcWindfromdir[1] = np.nan
    uas, vas = atmos.wind_vector_from_speed(
        sfcWind=sfcWind, sfcWindfromdir=sfcWindfromdir
    )
    np.testing.assert_array_equal(uas.isnull(), [True, True, False])
    np.testing.assert_array_equal(vas.isnull(), [True, True, False])


def test_relative_humidity_dewpoint(tas_series, rh_series):
    np.testing.assert_allclose(
        atmos.relative_humidity_from_dewpoint(
            tas=tas_series(np.array([-20, -10, -1, 10, 20, 25, 30, 40, 60]) + K2C),
            dtas=tas_series(np.array([-15, -10, -2, 5, 10, 20, 29, 20, 30]) + K2C),
        ),
        rh_series([np.nan, 100, 93, 71, 52, 73, 94, 31, 20]),
        rtol=0.02,
        atol=1,
    )


def test_saturation_vapor_pressure(tas_series):
    tas = tas_series(np.array([-20, -10, -1, 10, 20, 25, 30, 40, 60]) + K2C)
    e_sat_exp = [103, 260, 563, 1228, 2339, 3169, 4247, 7385, 19947]
    e_sat = atmos.saturation_vapor_pressure(
        tas=tas,
        method="sonntag90",
        ice_thresh="0 degC",
    )
    np.testing.assert_allclose(e_sat, e_sat_exp, atol=0.5, rtol=0.005)
    assert e_sat.name == "e_sat"


def test_relative_humidity(tas_series, rh_series, huss_series, ps_series):
    tas = tas_series(np.array([-10, -10, 10, 20, 35, 50, 75, 95]) + K2C)
    rh_exp = rh_series([np.nan, 63.0, 66.0, 34.0, 14.0, 6.0, 1.0, 0.0])
    ps = ps_series([101325] * 8)
    huss = huss_series([0.003, 0.001] + [0.005] * 7)

    rh = atmos.relative_humidity(
        tas=tas,
        huss=huss,
        ps=ps,
        method="sonntag90",
        ice_thresh="0 degC",
    )
    np.testing.assert_allclose(rh, rh_exp, atol=0.5, rtol=0.005)
    assert rh.name == "rh"


def test_specific_humidity(tas_series, rh_series, huss_series, ps_series):
    tas = tas_series(np.array([20, -10, 10, 20, 35, 50, 75, 95]) + K2C)
    rh = rh_series([150, 10, 90, 20, 80, 50, 70, 40, 30])
    ps = ps_series(1000 * np.array([100] * 4 + [101] * 4))
    huss_exp = huss_series(
        [np.nan, 1.6e-4, 6.9e-3, 3.0e-3, 2.9e-2, 4.1e-2, 2.1e-1, 5.7e-1]
    )

    huss = atmos.specific_humidity(
        tas=tas,
        rh=rh,
        ps=ps,
        method="sonntag90",
        ice_thresh="0 degC",
    )
    np.testing.assert_allclose(huss, huss_exp, atol=1e-4, rtol=0.05)
    assert huss.name == "huss"


def test_snowfall_approximation(pr_series, tasmax_series):
    pr = pr_series(np.ones(10))
    tasmax = tasmax_series(np.arange(10) + K2C)

    prsn = atmos.snowfall_approximation(
        pr, tas=tasmax, thresh="5 degC", method="binary"
    )

    np.testing.assert_allclose(
        prsn, [1, 1, 1, 1, 1, 0, 0, 0, 0, 0], atol=1e-5, rtol=1e-3
    )


def test_rain_approximation(pr_series, tas_series):
    pr = pr_series(np.ones(10))
    tas = tas_series(np.arange(10) + K2C)

    prlp = atmos.rain_approximation(pr, tas=tas, thresh="5 degC", method="binary")

    np.testing.assert_allclose(
        prlp, [0, 0, 0, 0, 0, 1, 1, 1, 1, 1], atol=1e-5, rtol=1e-3
    )

# This File is for testing atmos indicators that fit neither in test_precip or test_temperature
# Expected values might be the same as for the indices tests, see test_indices comments.
from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from xclim import atmos, set_options

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


def test_relative_humidity_dewpoint(tas_series, hurs_series):
    np.testing.assert_allclose(
        atmos.relative_humidity_from_dewpoint(
            tas=tas_series(np.array([-20, -10, -1, 10, 20, 25, 30, 40, 60]) + K2C),
            tdps=tas_series(np.array([-15, -10, -2, 5, 10, 20, 29, 20, 30]) + K2C),
        ),
        hurs_series([np.nan, 100, 93, 71, 52, 73, 94, 31, 20]),
        rtol=0.02,
        atol=1,
    )


def test_humidex(tas_series):

    tas = tas_series([15, 25, 35, 40])
    tas.attrs["units"] = "C"

    dtas = tas_series([10, 15, 25, 25])
    dtas.attrs["units"] = "C"

    # expected values from https://en.wikipedia.org/wiki/Humidex
    h = atmos.humidex(tas, dtas)
    np.testing.assert_array_almost_equal(h, [16, 29, 47, 52], 0)
    assert h.name == "humidex"


def test_heat_index(atmosds):
    # Keep just Montreal values for summertime as we need tas > 20 degC
    tas = atmosds.tasmax[1][150:170]
    hurs = atmosds.hurs[1][150:170]

    expected = np.array(
        [
            25.0,
            27.0,
            29.0,
            27.0,
            24.0,
            np.nan,
            np.nan,
            23.0,
            24.0,
            np.nan,
            np.nan,
            24.0,
            28.0,
            25.0,
            30.0,
            26.0,
            31.0,
            33.0,
            34.0,
            28.0,
        ]
    )

    with set_options(cf_compliance="raise"):
        hi = atmos.heat_index(tas, hurs)
    np.testing.assert_array_almost_equal(hi, expected, 0)
    assert hi.name == "heat_index"


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


def test_relative_humidity(tas_series, hurs_series, huss_series, ps_series):
    tas = tas_series(np.array([-10, -10, 10, 20, 35, 50, 75, 95]) + K2C)
    hurs_exp = hurs_series([np.nan, 63.0, 66.0, 34.0, 14.0, 6.0, 1.0, 0.0])
    ps = ps_series([101325] * 8)
    huss = huss_series([0.003, 0.001] + [0.005] * 7)

    hurs = atmos.relative_humidity(
        tas=tas,
        huss=huss,
        ps=ps,
        method="sonntag90",
        ice_thresh="0 degC",
    )
    np.testing.assert_allclose(hurs, hurs_exp, atol=0.5, rtol=0.005)
    assert hurs.name == "hurs"


def test_specific_humidity(tas_series, hurs_series, huss_series, ps_series):
    tas = tas_series(np.array([20, -10, 10, 20, 35, 50, 75, 95]) + K2C)
    hurs = hurs_series([150, 10, 90, 20, 80, 50, 70, 40, 30])
    ps = ps_series(1000 * np.array([100] * 4 + [101] * 4))
    huss_exp = huss_series(
        [np.nan, 1.6e-4, 6.9e-3, 3.0e-3, 2.9e-2, 4.1e-2, 2.1e-1, 5.7e-1]
    )

    huss = atmos.specific_humidity(
        tas=tas,
        hurs=hurs,
        ps=ps,
        method="sonntag90",
        ice_thresh="0 degC",
    )
    np.testing.assert_allclose(huss, huss_exp, atol=1e-4, rtol=0.05)
    assert huss.name == "huss"


def test_specific_humidity_from_dewpoint(tas_series, ps_series, huss_series):
    tdps = tas_series([272, 283, 293])
    ps = ps_series([100000, 105000, 110000])
    # Computed from MetPy
    # >>> from metpy.units import units as u
    # >>> from metpy.calc import specific_humidity_from_dewpoint as sh
    # >>> sh([100000, 105000, 110000] * u.Pa, [272, 283, 293] * u.degK)
    # array([0.0035031, 0.00722795, 0.01319614]) < Unit('dimensionless') >
    huss_exp = huss_series([0.0035031, 0.00722795, 0.01319614])

    huss = atmos.specific_humidity_from_dewpoint(
        tdps=tdps,
        ps=ps,
        method="sonntag90",
    )
    np.testing.assert_allclose(huss, huss_exp, atol=1e-4, rtol=0.05)
    assert huss.name == "huss_fromdewpoint"


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


def test_high_precip_low_temp(pr_series, tasmin_series):
    pr = np.zeros(365)
    pr[1:3] = [1, 2]
    pr = pr_series(pr, start="1999-01-01")

    tas = np.zeros(365)
    tas[2:4] = [1, 1]
    tas += K2C
    tas = tasmin_series(tas, start="1999-01-01")

    out = atmos.high_precip_low_temp(
        pr, tas, pr_thresh="1 kg m-2 s-1", tas_thresh="1 C"
    )
    np.testing.assert_array_equal(out, [1])


def test_wind_chill_index(atmosds):
    out = atmos.wind_chill_index(ds=atmosds)

    np.testing.assert_allclose(
        out.isel(time=0), [np.nan, -6.116, -36.064, -7.153, np.nan], rtol=1e-3
    )

    out_us = atmos.wind_chill_index(ds=atmosds, method="US")

    np.testing.assert_allclose(
        out_us.isel(time=0), [-1.041, -6.116, -36.064, -7.153, 2.951], rtol=1e-3
    )


class TestPotentialEvapotranspiration:
    def test_convert_units(self, open_dataset):
        ds = open_dataset(
            "ERA5/daily_surface_cancities_1990-1993.nc",
        )

        tn = ds.tasmin
        tx = ds.tasmax
        tm = ds.tas
        hurs = ds.hurs
        rsds = ds.rsds
        rsus = ds.rsus
        rlds = ds.rlds
        rlus = ds.rlus
        uas = ds.uas
        vas = ds.vas

        sfcWind, _ = atmos.wind_speed_from_vector(uas, vas)

        with xr.set_options(keep_attrs=True):
            tnC = tn - K2C
            tnC.attrs["units"] = "degC"
            tmC = tm - K2C
            tmC.attrs["units"] = "degC"

        pet_br65 = atmos.potential_evapotranspiration(tn, tx, method="BR65")
        pet_br65C = atmos.potential_evapotranspiration(tnC, tx, method="BR65")
        pet_hg85 = atmos.potential_evapotranspiration(tn, tx, method="HG85")
        pet_hg85C = atmos.potential_evapotranspiration(tnC, tx, method="HG85")
        pet_tw48 = atmos.potential_evapotranspiration(tas=tm, method="TW48")
        pet_tw48C = atmos.potential_evapotranspiration(tas=tmC, method="TW48")
        pet_mb05 = atmos.potential_evapotranspiration(tn, tx, method="MB05")
        pet_mb05C = atmos.potential_evapotranspiration(tnC, tx, method="MB05")
        pet_fao_pm98 = atmos.potential_evapotranspiration(
            tn,
            tx,
            hurs=hurs,
            rsds=rsds,
            rsus=rsus,
            rlds=rlds,
            rlus=rlus,
            sfcWind=sfcWind,
            method="FAO_PM98",
        )
        pet_fao_pm98C = atmos.potential_evapotranspiration(
            tnC,
            tx,
            hurs=hurs,
            rsds=rsds,
            rsus=rsus,
            rlds=rlds,
            rlus=rlus,
            sfcWind=sfcWind,
            method="FAO_PM98",
        )

        np.testing.assert_allclose(pet_br65, pet_br65C, atol=1)
        np.testing.assert_allclose(pet_hg85, pet_hg85C, atol=1)
        np.testing.assert_allclose(pet_tw48, pet_tw48C, atol=1)
        np.testing.assert_allclose(pet_mb05, pet_mb05C, atol=1)
        np.testing.assert_allclose(pet_fao_pm98, pet_fao_pm98C, atol=1)

    def test_nan_values(self, open_dataset):
        ds = open_dataset(
            "ERA5/daily_surface_cancities_1990-1993.nc",
        )

        tn = ds.tasmin
        tx = ds.tasmax
        tm = ds.tas
        hurs = ds.hurs
        rsds = ds.rsds
        rsus = ds.rsus
        rlds = ds.rlds
        rlus = ds.rlus
        uas = ds.uas
        vas = ds.vas

        sfcWind, _ = atmos.wind_speed_from_vector(uas, vas)

        tn[0, 100] = np.nan
        tx[0, 101] = np.nan

        pet_br65 = atmos.potential_evapotranspiration(tn, tx, method="BR65")
        pet_hg85 = atmos.potential_evapotranspiration(tn, tx, method="HG85")
        pet_fao_pm98 = atmos.potential_evapotranspiration(
            tn,
            tx,
            hurs=hurs,
            rsds=rsds,
            rsus=rsus,
            rlds=rlds,
            rlus=rlus,
            sfcWind=sfcWind,
            method="FAO_PM98",
        )

        tm[0, 0:31] = np.nan

        pet_tw48 = atmos.potential_evapotranspiration(tas=tm, method="TW48")

        np.testing.assert_allclose(
            pet_br65.isel(location=0, time=slice(100, 102)), [np.nan, np.nan]
        )
        np.testing.assert_allclose(
            pet_hg85.isel(location=0, time=slice(100, 102)), [np.nan, np.nan]
        )
        np.testing.assert_allclose(
            pet_fao_pm98.isel(location=0, time=slice(100, 102)),
            [np.nan, np.nan],
        )
        np.testing.assert_allclose(pet_tw48.isel(location=0, time=0), [np.nan])


class TestWaterBudget:
    def test_convert_units(self, open_dataset):
        ds = open_dataset(
            "ERA5/daily_surface_cancities_1990-1993.nc",
        )

        tn = ds.tasmin
        tx = ds.tasmax
        pr = ds.pr
        pet = ds.evspsblpot
        hurs = ds.hurs
        rsds = ds.rsds
        rsus = ds.rsus
        rlds = ds.rlds
        rlus = ds.rlus
        uas = ds.uas
        vas = ds.vas

        sfcWind, _ = atmos.wind_speed_from_vector(uas, vas)

        with xr.set_options(keep_attrs=True):
            tnC = tn - K2C
            tnC.attrs["units"] = "degC"
            prR = pr * 86400
            prR.attrs["units"] = "mm/day"
            petR = pet * 86400
            petR.attrs["units"] = "mm/day"

        p_pet_br65 = atmos.water_budget_from_tas(
            pr, tasmin=tn, tasmax=tx, method="BR65"
        )
        p_pet_br65C = atmos.water_budget_from_tas(
            prR, tasmin=tnC, tasmax=tx, method="BR65"
        )
        p_pet_hg85 = atmos.water_budget_from_tas(
            pr, tasmin=tn, tasmax=tx, method="HG85"
        )
        p_pet_hg85C = atmos.water_budget_from_tas(
            prR, tasmin=tnC, tasmax=tx, method="HG85"
        )
        p_pet_tw48 = atmos.water_budget_from_tas(
            pr, tasmin=tn, tasmax=tx, method="TW48"
        )
        p_pet_tw48C = atmos.water_budget_from_tas(
            prR, tasmin=tnC, tasmax=tx, method="TW48"
        )

        p_pet_fao_pm98 = atmos.water_budget_from_tas(
            pr=pr,
            tasmin=tn,
            tasmax=tx,
            hurs=hurs,
            rsds=rsds,
            rsus=rsus,
            rlds=rlds,
            rlus=rlus,
            sfcWind=sfcWind,
            method="FAO_PM98",
        )
        p_pet_fao_pm98R = atmos.water_budget_from_tas(
            pr=prR,
            tasmin=tn,
            tasmax=tx,
            hurs=hurs,
            rsds=rsds,
            rsus=rsus,
            rlds=rlds,
            rlus=rlus,
            sfcWind=sfcWind,
            method="FAO_PM98",
        )

        p_pet_evpot = atmos.water_budget(pr, evspsblpot=pet)
        p_pet_evpotR = atmos.water_budget(prR, evspsblpot=petR)

        np.testing.assert_allclose(p_pet_br65, p_pet_br65C, atol=1)
        np.testing.assert_allclose(p_pet_hg85, p_pet_hg85C, atol=1)
        np.testing.assert_allclose(p_pet_tw48, p_pet_tw48C, atol=1)
        np.testing.assert_allclose(p_pet_fao_pm98, p_pet_fao_pm98R, atol=1)
        np.testing.assert_allclose(p_pet_evpot, p_pet_evpotR, atol=1)

    def test_nan_values(self, open_dataset):
        ds = open_dataset(
            "ERA5/daily_surface_cancities_1990-1993.nc",
        )

        tn = ds.tasmin
        tx = ds.tasmax
        tm = ds.tas
        pr = ds.pr
        pet = ds.evspsblpot
        hurs = ds.hurs
        rsds = ds.rsds
        rsus = ds.rsus
        rlds = ds.rlds
        rlus = ds.rlus
        uas = ds.uas
        vas = ds.vas

        sfcWind, _ = atmos.wind_speed_from_vector(uas, vas)

        tn[0, 100] = np.nan
        tx[0, 101] = np.nan

        p_pet_br65 = atmos.water_budget_from_tas(
            pr, tasmin=tn, tasmax=tx, method="BR65"
        )
        p_pet_hg85 = atmos.water_budget_from_tas(
            pr, tasmin=tn, tasmax=tx, method="HG85"
        )
        p_pet_fao_pm98 = atmos.water_budget_from_tas(
            pr=pr,
            tasmin=tn,
            tasmax=tx,
            hurs=hurs,
            rsds=rsds,
            rsus=rsus,
            rlds=rlds,
            rlus=rlus,
            sfcWind=sfcWind,
            method="FAO_PM98",
        )

        tm[0, 0:31] = np.nan

        p_pet_tw48 = atmos.water_budget_from_tas(pr, tas=tm, method="TW48")

        pet[0, 0:31] = np.nan

        p_pet_evpot = atmos.water_budget(pr, evspsblpot=pet)

        np.testing.assert_allclose(p_pet_br65[0, 100:102], [np.nan, np.nan])
        np.testing.assert_allclose(p_pet_hg85[0, 100:102], [np.nan, np.nan])
        np.testing.assert_allclose(p_pet_fao_pm98[0, 100:102], [np.nan, np.nan])
        np.testing.assert_allclose(p_pet_tw48[0, 0], [np.nan])
        np.testing.assert_allclose(p_pet_evpot[0, 0], [np.nan])


class TestUTCI:
    def test_universal_thermal_climate_index(self, open_dataset):
        dataset = open_dataset(
            "ERA5/daily_surface_cancities_1990-1993.nc",
        )

        tas = dataset.tas
        hurs = dataset.hurs
        sfcWind, sfcWindfromdir = atmos.wind_speed_from_vector(
            uas=dataset.uas, vas=dataset.vas
        )
        rsds = dataset.rsds
        rsus = dataset.rsus
        rlds = dataset.rlds
        rlus = dataset.rlus
        # Expected values
        utci_exp = [256.8, 258.0, 237.4, 258.5, 266.2]

        utci = atmos.universal_thermal_climate_index(
            tas=tas,
            hurs=hurs,
            sfcWind=sfcWind,
            rsds=rsds,
            rsus=rsus,
            rlds=rlds,
            rlus=rlus,
            stat="average",
        )

        np.testing.assert_allclose(utci.isel(time=0), utci_exp, rtol=1e-03)


@pytest.mark.skip(reason="Causes thread problems? It's a mystery.")
class TestMeanRadiantTemperature:
    def test_mean_radiant_temperature(self, open_dataset):
        dataset = open_dataset(
            "ERA5/daily_surface_cancities_1990-1993.nc",
        )
        rsds = dataset.rsds
        rsus = dataset.rsus
        rlds = dataset.rlds
        rlus = dataset.rlus

        # Expected values
        exp_sun = [np.nan, np.nan, np.nan, np.nan, np.nan]
        exp_ins = [277.1, 274.6, 243.5, 268.1, 309.1]
        exp_avg = [277.1, 274.6, 243.5, 268.1, 278.4]

        mrt_sun = atmos.mean_radiant_temperature(rsds, rsus, rlds, rlus, stat="sunlit")
        mrt_ins = atmos.mean_radiant_temperature(rsds, rsus, rlds, rlus, stat="instant")
        mrt_avg = atmos.mean_radiant_temperature(rsds, rsus, rlds, rlus, stat="average")
        rtol = 1e-03
        np.testing.assert_allclose(mrt_sun.isel(time=0), exp_sun, rtol=rtol)
        np.testing.assert_allclose(mrt_ins.isel(time=0), exp_ins, rtol=rtol)
        np.testing.assert_allclose(mrt_avg.isel(time=0), exp_avg, rtol=rtol)

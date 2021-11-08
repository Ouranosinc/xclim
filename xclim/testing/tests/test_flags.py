import numpy as np
import pytest
import xarray as xr

from xclim.core import dataflags as df
from xclim.testing import open_dataset

K2C = 273.15


class TestDataFlags:
    @pytest.mark.parametrize(
        "vars_dropped, flags",
        [
            (["tasmin"], dict(tas_exceeds_tasmax=False, tas_below_tasmin=None)),
            (["tasmax"], dict(tas_exceeds_tasmax=None, tas_below_tasmin=False)),
            ([], dict(tas_exceeds_tasmax=False, tas_below_tasmin=False)),
        ],
    )
    def test_tas_temperature_flags(
        self, vars_dropped, flags, tas_series, tasmax_series, tasmin_series
    ):
        ds = xr.Dataset()
        for series, val in zip(
            [tas_series, tasmax_series, tasmin_series], [0, 10, -10]
        ):
            vals = val + K2C + np.sin(2 * np.pi * np.arange(366 * 3) / 366)
            arr = series(vals, start="1971-01-01")
            ds = xr.merge([ds, arr])

        ds = ds.drop_vars(vars_dropped)
        flagged_ds = df.data_flags(ds.tas, ds)

        np.testing.assert_equal(flagged_ds.temperature_extremely_high.values, False)
        np.testing.assert_equal(flagged_ds.temperature_extremely_low.values, False)
        np.testing.assert_equal(
            flagged_ds.values_repeating_for_5_or_more_days.values, False
        )
        np.testing.assert_equal(
            flagged_ds.outside_5_standard_deviations_of_climatology.values, False
        )

        for flag, val in flags.items():
            np.testing.assert_equal(getattr(flagged_ds, flag).values, val)

    def test_pr_precipitation_flags(self, pr_series):
        pr = pr_series(np.zeros(365), start="1971-01-01")
        pr += 1 / 3600 / 24
        pr[0:7] += 10 / 3600 / 24
        pr[-7:] += 11 / 3600 / 24

        flagged = df.data_flags(pr)
        print(flagged)
        np.testing.assert_equal(flagged.negative_accumulation_values.values, False)
        np.testing.assert_equal(flagged.very_large_precipitation_events.values, False)
        np.testing.assert_equal(
            flagged.values_eq_5_repeating_for_5_or_more_days.values,
            False,
        )
        np.testing.assert_equal(
            flagged.values_eq_1_repeating_for_10_or_more_days.values,
            False,
        )

    def test_suspicious_pr_data(self, pr_series):
        bad_pr = pr_series(np.zeros(365), start="1971-01-01")

        # Add some strangeness
        bad_pr[8] = -1e-6  # negative values
        bad_pr[120] = 301 / 3600 / 24  # 301mm/day
        bad_pr[121:141] = 1.1574074074074072e-05  # 1mm/day
        bad_pr[200:300] = 5.787037037037036e-05  # 5mm/day

        flagged = df.data_flags(bad_pr)
        np.testing.assert_equal(flagged.negative_accumulation_values.values, True)
        np.testing.assert_equal(flagged.very_large_precipitation_events.values, True)
        np.testing.assert_equal(
            flagged.values_eq_1_repeating_for_10_or_more_days.values, True
        )
        np.testing.assert_equal(
            flagged.values_eq_5_repeating_for_5_or_more_days.values, True
        )

    def test_suspicious_tas_data(self, tas_series, tasmax_series, tasmin_series):
        bad_ds = xr.Dataset()
        for series, val in zip(
            [tas_series, tasmax_series, tasmin_series], [0, 10, -10]
        ):
            vals = val + K2C + np.sin(2 * np.pi * np.arange(366 * 7) / 366)
            arr = series(vals, start="1971-01-01")
            bad_ds = xr.merge([bad_ds, arr])

        # Swap entire variable arrays
        bad_ds["tasmin"].values, bad_ds["tasmax"].values = (
            bad_ds.tasmax.values,
            bad_ds.tasmin.values,
        )

        bad_tas = bad_ds.tas.values
        # Add some jankiness to tas
        bad_tas[5] = 58 + K2C  # Fluke event beyond 5 standard deviations
        bad_tas[600:610] = 80 + K2C  # Repeating values above hot extreme
        bad_tas[950] = -95 + K2C  # Cold extreme
        bad_ds["tas"].values = bad_tas

        flagged = df.data_flags(bad_ds.tas, bad_ds)
        np.testing.assert_equal(flagged.temperature_extremely_high.values, True)
        np.testing.assert_equal(flagged.temperature_extremely_low.values, True)
        np.testing.assert_equal(
            flagged.values_repeating_for_5_or_more_days.values, True
        )
        np.testing.assert_equal(
            flagged.outside_5_standard_deviations_of_climatology.values,
            True,
        )
        np.testing.assert_equal(flagged.tas_exceeds_tasmax.values, True)
        np.testing.assert_equal(flagged.tas_below_tasmin.values, True)

    def test_raises(self, tasmax_series, tasmin_series):
        bad_ds = xr.Dataset()
        for series, val in zip([tasmax_series, tasmin_series], [10, -10]):
            vals = val + K2C + np.sin(2 * np.pi * np.arange(366 * 3) / 366)
            arr = series(vals, start="1971-01-01")
            bad_ds = xr.merge([bad_ds, arr])

        # Swap entire variable arrays
        bad_ds["tasmin"].values, bad_ds["tasmax"].values = (
            bad_ds.tasmax.values,
            bad_ds.tasmin.values,
        )
        with pytest.raises(
            df.DataQualityException,
            match="Maximum temperature values found below minimum temperatures.",
        ):
            df.data_flags(bad_ds.tasmax, bad_ds, raise_flags=True)

    def test_era5_ecad_qc_flag(self):
        bad_ds = open_dataset("ERA5/daily_surface_cancities_1990-1993.nc")  # noqa

        # Add some suspicious run values
        bad_ds["tas"].values[0][100:300] = 17 + K2C

        with pytest.raises(
            df.DataQualityException,
            match="Runs of repetitive values for 5 or more days found for tas.",
        ):
            df.ecad_compliant(bad_ds, raise_flags=True)

        df_flagged = df.ecad_compliant(bad_ds)
        np.testing.assert_array_equal(df_flagged.ecad_qc_flag, False)

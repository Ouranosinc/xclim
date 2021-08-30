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
    def test_tas_temperature_flags(self, vars_dropped, flags):
        ds = open_dataset("ERA5/daily_surface_cancities_1990-1993.nc")
        ds = ds.drop_vars(vars_dropped)
        flagged_ds = df.data_flags(ds.tas, ds)

        np.testing.assert_equal(flagged_ds.temperature_extremely_high.values, False)
        np.testing.assert_equal(flagged_ds.temperature_extremely_low.values, False)
        np.testing.assert_equal(
            flagged_ds.values_repeating_for_5_or_more_days.values, False
        )
        np.testing.assert_equal(
            flagged_ds.outside_5_standard_deviations_of_climatology.values, True
        )

        for flag, val in flags.items():
            np.testing.assert_equal(getattr(flagged_ds, flag).values, val)

    def test_pr_precipitation_flags(self):
        ds = open_dataset("ERA5/daily_surface_cancities_1990-1993.nc")
        flagged_ds = df.data_flags(ds.pr, ds)

        np.testing.assert_equal(flagged_ds.negative_accumulation_values.values, False)
        np.testing.assert_equal(
            flagged_ds.very_large_precipitation_events.values, False
        )
        np.testing.assert_equal(
            flagged_ds.values_of_5mm_repeating_for_5_or_more_days.values, False
        )
        np.testing.assert_equal(
            flagged_ds.values_of_1mm_repeating_for_10_or_more_days.values, False
        )

    def test_suspicious_pr_data(self):
        bad_ds = open_dataset("ERA5/daily_surface_cancities_1990-1993.nc")  # noqa
        location = bad_ds.location.values
        time = bad_ds.time.values

        pr = bad_ds.pr.values
        # Add some strangeness
        pr[0][800] = -1e-6  # negative values
        pr[1][1200] = 0.003483796296  # 301mm/day
        pr[2][1200:1300] = 0.000011574074  # 1mm/day
        pr[3][200:300] = 0.00005787037  # 5mm/day
        bad_pr = xr.DataArray(
            pr,
            coords=dict(location=location, time=time),
            dims=["location", "time"],
            attrs=dict(
                units="kg m-2 s-1",
                cell_methods="time: mean within days",
                standard_name="precipitation_flux",
                long_name="Mean daily precipitation flux",
            ),
        )
        bad_ds["pr"] = bad_pr

        flagged = df.data_flags(bad_ds.pr, bad_ds)

        np.testing.assert_equal(flagged.negative_accumulation_values.values, True)
        np.testing.assert_equal(flagged.very_large_precipitation_events.values, True)
        np.testing.assert_equal(
            flagged.values_of_1mm_repeating_for_10_or_more_days.values, True
        )
        np.testing.assert_equal(
            flagged.values_of_5mm_repeating_for_5_or_more_days.values, True
        )

    def test_suspicious_tas_data(self):
        bad_ds = open_dataset("ERA5/daily_surface_cancities_1990-1993.nc")  # noqa
        location = bad_ds.location.values
        time = bad_ds.time.values

        # Swap entire variable arrays
        bad_ds["tasmin"].values, bad_ds["tasmax"].values = (
            bad_ds.tasmax.values,
            bad_ds.tasmin.values,
        )

        tas = bad_ds.tas.values
        # Add some jankiness to tas
        tas[0][100:300] = 17 + K2C
        tas[1][600] = 80 + K2C
        tas[2][950] = -95 + K2C
        bad_tas = xr.DataArray(
            tas,
            coords=dict(location=location, time=time),
            dims=["location", "time"],
            attrs=dict(
                units="K",
                cell_methods="time: mean within days",
                standard_name="air_temperature",
                long_name="Mean daily surface temperature",
            ),
        )
        bad_ds["tas"] = bad_tas

        flagged = df.data_flags(bad_ds.tas, bad_ds)
        np.testing.assert_equal(flagged.temperature_extremely_high.values, True)
        np.testing.assert_equal(flagged.temperature_extremely_low.values, True)
        np.testing.assert_equal(
            flagged.values_repeating_for_5_or_more_days.values, True
        )
        np.testing.assert_equal(
            flagged.outside_5_standard_deviations_of_climatology.values, True
        )
        np.testing.assert_equal(flagged.tas_exceeds_tasmax.values, True)
        np.testing.assert_equal(flagged.tas_below_tasmin.values, True)

    def test_raises(self):
        bad_ds = open_dataset("ERA5/daily_surface_cancities_1990-1993.nc")  # noqa

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

    def test_ecad_qc_flag(self):
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

from __future__ import annotations

import platform
import sys
from pathlib import Path

import numpy as np
import pytest
from xarray import Dataset

from xclim import __version__ as __xclim_version__
from xclim.testing import helpers
from xclim.testing import utils as utilities
from xclim.testing.helpers import test_timeseries as timeseries


class TestFixtures:
    def test_timeseries_made_up_variable(self):
        ds = timeseries(
            np.zeros(31),
            "luminiferous_aether_flux",
            units="W K mol A-1 m-2 s-1",
            as_dataset=True,
        )

        assert isinstance(ds, Dataset)
        assert ds.luminiferous_aether_flux.attrs["units"] == "W K mol A-1 m-2 s-1"
        assert "standard_name" not in ds.luminiferous_aether_flux.attrs


class TestFileRequests:

    @staticmethod
    def file_md5_checksum(f_name):
        import hashlib

        hash_md5 = hashlib.md5()  # noqa: S324
        with open(f_name, "rb") as f:
            hash_md5.update(f.read())
        return hash_md5.hexdigest()

    @pytest.mark.requires_internet
    def test_open_testdata(self):
        ds = helpers.open_dataset(
            Path("cmip5/tas_Amon_CanESM2_rcp85_r1i1p1_200701-200712"), engine="h5netcdf"
        )
        assert ds.lon.size == 128

    def test_md5_sum(self):
        test_data = Path(__file__).parent / "data"
        callendar = test_data / "callendar_1938.txt"
        md5_sum = self.file_md5_checksum(callendar)
        if sys.platform == "win32":
            # Windows has a different line ending (CR-LF) than Unix (LF)
            assert md5_sum == "38083271c2d4c85dea6bd6baf23d34de"  # noqa
        else:
            assert md5_sum == "9a5d9f94d76d4f9d9b7aaadbe8cbf541"  # noqa


class TestReleaseSupportFuncs:
    def test_show_version_file(self, tmp_path):
        temp_filename = tmp_path.joinpath("version_info.txt")
        utilities.show_versions(file=temp_filename)

        with open(temp_filename) as f:
            contents = f.readlines().copy()
            assert "INSTALLED VERSIONS\n" in contents
            assert "------------------\n" in contents
            assert f"python: {platform.python_version()}\n" in contents
            assert f"xclim: {__xclim_version__}\n" in contents
            assert "boltons: installed\n" in contents

    @pytest.mark.requires_docs
    def test_release_notes_file(self, tmp_path):
        temp_filename = tmp_path.joinpath("version_info.txt")
        utilities.publish_release_notes(style="md", file=temp_filename)

        with open(temp_filename) as f:
            assert "# Changelog" in f.readlines()[0]

    @pytest.mark.requires_docs
    def test_release_notes_file_not_implemented(self, tmp_path):
        temp_filename = tmp_path.joinpath("version_info.txt")
        with pytest.raises(NotImplementedError):
            utilities.publish_release_notes(style="qq", file=temp_filename)

from __future__ import annotations

import platform
from pathlib import Path
from urllib.error import HTTPError

import numpy as np
import pytest

import xclim.testing.utils as utilities
from xclim import __version__ as __xclim_version__


class TestFileRequests:
    def test_get_failure(self, tmp_path):
        bad_repo_address = "https://github.com/beard/of/zeus/"
        with pytest.raises(HTTPError):
            utilities._get(
                Path("san_diego", "60_percent_of_the_time_it_works_everytime"),
                bad_repo_address,
                "main",
                ".rudd",
                tmp_path,
            )

    def test_open_dataset_with_bad_file(self, tmp_path):
        cmip3_folder = tmp_path.joinpath("main", "cmip3")
        cmip3_folder.mkdir(parents=True)

        cmip3_file = "tas.sresb1.giss_model_e_r.run1.atm.da.nc"
        Path(cmip3_folder, cmip3_file).write_text("This file definitely isn't right.")

        cmip3_md5 = f"{cmip3_file}.md5"
        bad_cmip3_md5 = "bc51206e6462fc8ed08fd4926181274c"
        Path(cmip3_folder, cmip3_md5).write_text(bad_cmip3_md5)

        # Check for raised warning for local file md5 sum and remote md5 sum
        with pytest.warns(UserWarning):
            new_cmip3_file = utilities._get(
                Path("cmip3", cmip3_file),
                github_url="https://github.com/Ouranosinc/xclim-testdata",
                suffix=".nc",
                branch="main",
                cache_dir=tmp_path,
            )

        # Ensure that the new cmip3 file is in the cache directory
        assert (
            utilities.file_md5_checksum(Path(cmip3_folder, new_cmip3_file))
            != bad_cmip3_md5
        )

        # Ensure that the md5 file was updated at the same time
        assert (
            utilities.file_md5_checksum(Path(cmip3_folder, new_cmip3_file))
            == Path(cmip3_folder, cmip3_md5).read_text()
        )

    def test_open_testdata(self):
        ds = utilities.open_dataset(
            Path("cmip5", "tas_Amon_CanESM2_rcp85_r1i1p1_200701-200712").as_posix()
        )
        assert ds.lon.size == 128

    # Not that this test is super slow, but there is no need in spamming github's API for no reason.
    @pytest.mark.slow
    @pytest.mark.xfail(reason="Test is rate limited by GitHub.")
    def test_list_datasets(self):
        out = utilities.list_datasets()

        assert list(out.columns) == ["size", "url"]
        np.testing.assert_allclose(
            out.loc["cmip6/o3_Amon_GFDL-ESM4_historical_r1i1p1f1_gr1_185001-194912.nc"][
                "size"
            ],
            845.021484375,
        )


class TestFileAssertions:
    def test_md5_sum(self):
        test_data = Path(__file__).parent / "data"
        callendar = test_data / "callendar_1938.txt"
        md5_sum = utilities.file_md5_checksum(callendar)
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


class TestTestingFileAccessors:
    def test_unsafe_urls(self):
        with pytest.raises(
            ValueError, match="GitHub URL not safe: 'ftp://domain.does.not.exist/'."
        ):
            utilities.open_dataset(
                "doesnt_exist.nc", github_url="ftp://domain.does.not.exist/"
            )

        with pytest.raises(
            ValueError, match="OPeNDAP URL not safe: 'ftp://domain.does.not.exist/'."
        ):
            utilities.open_dataset(
                "doesnt_exist.nc", dap_url="ftp://domain.does.not.exist/"
            )

    def test_bad_opendap_url(self):
        with pytest.raises(
            OSError,
            match="OPeNDAP file not read. Verify that the service is available.",
        ):
            utilities.open_dataset(
                "doesnt_exist.nc", dap_url="https://dap.service.does.not.exist/"
            )

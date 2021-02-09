"""Testing and tutorial utilities module."""
# Most of this code copied and adapted from xarray
import logging
from pathlib import Path
from typing import Optional
from urllib.error import HTTPError
from urllib.parse import urljoin
from urllib.request import urlretrieve

from xarray import Dataset
from xarray import open_dataset as _open_dataset
from xarray.tutorial import file_md5_checksum

_default_cache_dir = Path.home() / ".xclim_testing_data"

LOGGER = logging.getLogger("xclim")

__all__ = ["open_dataset"]


def _get(
    fullname: Path,
    github_url: str,
    branch: str,
    suffix: str,
    cache_dir: Path,
) -> Path:
    cache_dir = cache_dir.absolute()
    local_file = cache_dir / branch / fullname
    md5name = fullname.with_suffix("{}.md5".format(suffix))
    md5file = cache_dir / branch / md5name

    if not local_file.is_file():
        # This will always leave this directory on disk.
        # We may want to add an option to remove it.
        local_file.parent.mkdir(parents=True, exist_ok=True)

        url = "/".join((github_url, "raw", branch, fullname.as_posix()))
        LOGGER.info("Fetching remote file: %s" % fullname.as_posix())
        urlretrieve(url, local_file)
        try:
            url = "/".join((github_url, "raw", branch, md5name.as_posix()))
            LOGGER.info("Fetching remote file md5: %s" % md5name.as_posix())
            urlretrieve(url, md5file)
        except HTTPError as e:
            msg = f"{md5name.as_posix()} not found. Aborting file retrieval."
            local_file.unlink()
            raise FileNotFoundError(msg) from e

        localmd5 = file_md5_checksum(local_file)
        try:
            with open(md5file) as f:
                remotemd5 = f.read()
            if localmd5.strip() != remotemd5.strip():
                local_file.unlink()
                msg = """
                    MD5 checksum does not match, try downloading dataset again.
                    """
                raise OSError(msg)
        except OSError as e:
            LOGGER.error(e)
    return local_file


# idea copied from raven that it borrowed from xclim that borrowed it from xarray that was borrowed from Seaborn
def open_dataset(
    name: str,
    suffix: Optional[str] = None,
    dap_url: Optional[str] = None,
    github_url: str = "https://github.com/Ouranosinc/xclim-testdata",
    branch: str = "main",
    cache: bool = True,
    cache_dir: Path = _default_cache_dir,
    **kwds,
) -> Dataset:
    """
    Open a dataset from the online GitHub-like repository.

    If a local copy is found then always use that to avoid network traffic.

    Parameters
    ----------
    name : str
        Name of the file containing the dataset.
    suffix : str, optional
        If no suffix is given, assumed to be netCDF ('.nc' is appended). For no suffix, set "".
    dap_url : str, optional
        URL to OPeNDAP folder where the data is stored. If supplied, supersedes github_url.
    github_url : str
        URL to Github repository where the data is stored.
    branch : str, optional
        For GitHub-hosted files, the branch to download from.
    cache_dir : Path
        The directory in which to search for and write cached data.
    cache : bool
        If True, then cache data locally for use on subsequent calls.
    kwds : dict, optional
        For NetCDF files, **kwds passed to xarray.open_dataset.

    Returns
    -------
    Union[Dataset, Path]

    See Also
    --------
    xarray.open_dataset
    """
    name = Path(name)
    if suffix is None:
        suffix = ".nc"
    fullname = name.with_suffix(suffix)

    if dap_url is not None:
        dap_file = urljoin(dap_url, str(name))
        try:
            ds = _open_dataset(dap_file, **kwds)
            return ds
        except OSError:
            msg = "OPeNDAP file not read. Verify that service is available."
            LOGGER.error(msg)
            raise

    local_file = _get(
        fullname=fullname,
        github_url=github_url,
        branch=branch,
        suffix=suffix,
        cache_dir=cache_dir,
    )

    try:
        ds = _open_dataset(local_file, **kwds)
        if not cache:
            ds = ds.load()
            local_file.unlink()
        return ds
    except OSError:
        raise


def as_tuple(x):  # noqa: D103
    if isinstance(x, (list, tuple)):
        return x
    return (x,)  # noqa


class TestFile:  # noqa: D101
    def __init__(self, name, path=None, url=None):
        """Register a test file.

        Parameters
        ----------
        name : str
          Short identifier for test file.
        path : Path
          Local path.
        url : str
          Remote location to retrieve file if it's not on disk.
        """
        self.name = name
        self.path = path
        self.url = url

    def generate(self):
        """Create the test file from scratch."""
        pass

    def download(self):
        """Download a remote file."""
        for u, p in zip(as_tuple(self.url), as_tuple(self.path)):
            urlretrieve(u, str(p))

    def __call__(self):  # noqa: D102
        """Return the path to the file."""
        if not self.path.exists():
            if self.url is not None:
                self.download()
            else:
                self.generate()

        if not self.path.exists():
            raise FileNotFoundError

        return self.path


class TestDataSet:  # noqa: D101
    def __init__(self, name, path, files=()):
        self.name = name
        self.files = list(files)
        self.path = Path(path)
        if not self.path.exists():
            self.path.mkdir()

    def add(self, name, url, path=None):  # noqa: D102
        if path is None:
            # Create a relative path
            path = self.path / Path(url).name

        elif not Path(path).is_absolute():
            path = self.path / path

        tf = TestFile(name, path, url)
        setattr(self, name, tf)
        self.files.append(tf)

    def __call__(self):  # noqa: D102
        return [f() for f in self.files]

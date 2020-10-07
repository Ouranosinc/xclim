"""Testing and tutorial utilities module."""
# Most of this code copied and adapted from xarray
from pathlib import Path
from urllib.request import urlretrieve

from xarray import open_dataset as _open_dataset
from xarray.tutorial import file_md5_checksum

_default_cache_dir = Path.home() / ".xclim_testing_data"


# idea copied from xarray that borrowed it from Seaborn
def open_dataset(
    name,
    cache: bool = True,
    cache_dir: Path = _default_cache_dir,
    github_url: str = "https://github.com/Ouranosinc/xclim-testdata",
    branch: str = "main",
    **kws,
):
    """
    Open a dataset from the online repository (requires internet).

    If a local copy is found then always use that to avoid network traffic.

    Parameters
    ----------
    name : str
        Name of the file containing the dataset. If no suffix is given, assumed
        to be netCDF ('.nc' is appended). The name may contain
    cache_dir : Path
        The directory in which to search for and write cached data.
    cache : bool
        If True, then cache data locally for use on subsequent calls
    github_url : str
        Github repository where the data is stored
    branch : str
        The git branch to download from
    kws : dict, optional
        Passed to xarray.open_dataset

    See Also
    --------
    xarray.open_dataset

    """
    name = Path(name)
    fullname = name.with_suffix(".nc")
    cache_dir = cache_dir.absolute()
    local_file = cache_dir / fullname
    md5name = fullname.with_suffix(".nc.md5")
    md5file = cache_dir / md5name

    if not local_file.is_file():
        # This will always leave this directory on disk.
        # We may want to add an option to remove it.
        local_file.parent.mkdir(parents=True, exist_ok=True)

        url = "/".join((github_url, "raw", branch, fullname.as_posix()))
        urlretrieve(url, local_file)
        url = "/".join((github_url, "raw", branch, md5name.as_posix()))
        urlretrieve(url, md5file)

        localmd5 = file_md5_checksum(local_file)
        with open(md5file) as f:
            remotemd5 = f.read()
        if localmd5 != remotemd5:
            local_file.unlink()
            msg = """
            MD5 checksum does not match, try downloading dataset again.
            """
            raise OSError(msg)

    ds = _open_dataset(local_file, **kws)

    if not cache:
        ds = ds.load()
        local_file.unlink()

    return ds


def as_tuple(x):  # noqa: D103
    if isinstance(x, (list, tuple)):
        return x
    else:
        return (x,)


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

"""Testing and tutorial utilities module."""
# Most of this code copied and adapted from xarray
import hashlib
import json
import logging
import warnings
from pathlib import Path
from typing import Optional, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import urlopen, urlretrieve

import pandas as pd
from xarray import Dataset
from xarray import open_dataset as _open_dataset
from yaml import safe_dump, safe_load

_default_cache_dir = Path.home() / ".xclim_testing_data"

LOGGER = logging.getLogger("xclim")

__all__ = [
    "open_dataset",
    "list_datasets",
    "list_input_variables",
    "get_all_CMIP6_variables",
    "update_variable_yaml",
]


def file_md5_checksum(fname):
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        hash_md5.update(f.read())
    return hash_md5.hexdigest()


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

    if local_file.is_file():
        localmd5 = file_md5_checksum(local_file)
        try:
            url = "/".join((github_url, "raw", branch, md5name.as_posix()))
            LOGGER.info("Attempting to fetch remote file md5: %s" % md5name.as_posix())
            urlretrieve(url, md5file)
            with open(md5file) as f:
                remote_md5 = f.read()
            if localmd5.strip() != remote_md5.strip():
                local_file.unlink()
                msg = (
                    f"MD5 checksum for {local_file.as_posix()} does not match upstream md5. "
                    "Attempting new download."
                )
                warnings.warn(msg)
        except (HTTPError, URLError):
            msg = f"{md5name.as_posix()} not accessible online. Unable to determine validity with upstream repo."
            warnings.warn(msg)

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
                remote_md5 = f.read()
            if localmd5.strip() != remote_md5.strip():
                local_file.unlink()
                msg = (
                    f"{local_file.as_posix()} and md5 checksum do not match. "
                    "There may be an issue with the upstream origin data."
                )
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


def list_datasets(github_repo="Ouranosinc/xclim-testdata", branch="main"):
    """Return a DataFrame listing all xclim test datasets available on the github repo for the given branch.

    The result includes the filepath, as passed to `open_dataset`, the file size (in KB) and the html url to the file.
    This uses a unauthenticated call to Github's REST API, so it is limited to 60 requests per hour (per IP).
    A single call of this function triggers one request per subdirectory, so use with parsimony.
    """
    res = urlopen(f"https://api.github.com/repos/{github_repo}/contents?ref={branch}")
    base = json.loads(res.read().decode())
    records = []
    for folder in base:
        if folder["path"].startswith(".") or folder["size"] > 0:
            # drop hidden folders and other files.
            continue
        res = urlopen(folder["url"])
        listing = json.loads(res.read().decode())
        for file in listing:
            if file["path"].endswith(".nc"):
                records.append(
                    {
                        "name": file["path"],
                        "size": file["size"] / 2 ** 10,
                        "url": file["html_url"],
                    }
                )
    df = pd.DataFrame.from_records(records).set_index("name")
    print(f"Found {len(df)} datasets.")
    return df


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


def list_input_variables(
    submodules: Sequence[str] = None, realms: Sequence[str] = None
) -> dict:
    """List all possible variables names used in xclim's indicators.

    Made for development purposes. Parses all indicator parameters with the
    :py:attribute:`xclim.core.utils.InputKind.VARIABLE` or `OPTIONAL_VARIABLE` kinds.

    Parameters
    ----------
    realms: Sequence of str, optional
      Restrict the output to indicators of a list of realms only. Default None, which parses all indicators.
    submodules: str, optional
      Restrict the output to indicators of a list of submodules only. Default None, which parses all indicators.

    Returns
    -------
    dict
      A mapping from variable name to indicator class.
    """
    from collections import defaultdict

    from xclim import indicators
    from xclim.core.indicator import registry
    from xclim.core.utils import InputKind

    submodules = submodules or [
        sub for sub in dir(indicators) if not sub.startswith("__")
    ]
    realms = realms or ["atmos", "ocean", "land", "seaIce"]

    variables = defaultdict(list)
    for name, ind in registry.items():
        if "." in name:
            # external submodule, sub module name is prepened to registry key
            if name.split(".")[0] not in submodules:
                continue
        elif ind.realm not in submodules:
            # official indicator : realm == submodule
            continue
        if ind.realm not in realms:
            continue

        # ok we want this one.
        for varname, meta in ind.parameters.items():
            if meta["kind"] in [InputKind.VARIABLE, InputKind.OPTIONAL_VARIABLE]:
                var = meta.get("default") or varname
                variables[var].append(ind)

    return variables


def get_all_CMIP6_variables(get_cell_methods=True):
    data = pd.read_excel(
        "http://proj.badc.rl.ac.uk/svn/exarch/CMIP6dreq/tags/01.00.33/dreqPy/docs/CMIP6_MIP_tables.xlsx",
        sheet_name=None,
    )
    data.pop("Notes")

    variables = {}

    def summarize_cell_methods(rawstr):
        words = str(rawstr).split(" ")
        iskey = [word.endswith(":") for word in words]
        cms = {}
        for i in range(len(words)):
            if iskey[i] and i + 1 < len(words) and not iskey[i + 1]:
                cms[words[i][:-1]] = words[i + 1]
        return cms

    for table, df in data.items():
        for i, row in df.iterrows():
            varname = row["Variable Name"]
            vardata = {
                "standard_name": row["CF Standard Name"],
                "canonical_units": row["units"],
            }
            if get_cell_methods:
                vardata["cell_methods"] = [summarize_cell_methods(row["cell_methods"])]
            if varname in variables and get_cell_methods:
                if vardata["cell_methods"] not in variables[varname]["cell_methods"]:
                    variables[varname]["cell_methods"].append(vardata["cell_methods"])
            else:
                variables[varname] = vardata

    return variables


def update_variable_yaml(filename=None, xclim_needs_only=True):
    print("Downloading CMIP6 variables.")
    allvars = get_all_CMIP6_variables(get_cell_methods=False)

    if xclim_needs_only:
        print("Filtering with xclim-implemented variables.")
        xcvars = list_input_variables()
        allvars = {k: v for k, v in allvars.items() if k in xcvars}

    filepath = Path(
        filename or (Path(__file__).parent.parent / "data" / "variables.yml")
    )

    if filepath.exists():
        with filepath.open() as f:
            stdvars = safe_load(f)

        for var, data in allvars.items():
            if var not in stdvars["variables"]:
                print(f"Added {var}")
                new_data = data.copy()
                new_data.update(description="")
                stdvars["variables"][var] = new_data
    else:
        stdvars = allvars

    with filepath.open("w") as f:
        safe_dump(stdvars, f)

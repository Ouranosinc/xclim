import urllib.request
from pathlib import Path


def as_tuple(x):
    if isinstance(x, (list, tuple)):
        return x
    else:
        return (x,)


class TestFile:
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
            urllib.request.urlretrieve(u, str(p))

    def __call__(self):
        """Return the path to the file."""
        if not self.path.exists():
            if self.url is not None:
                self.download()
            else:
                self.generate()

        if not self.path.exists():
            raise FileNotFoundError

        return self.path


class TestDataSet:
    def __init__(self, name, path, files=()):
        self.name = name
        self.files = list(files)
        self.path = Path(path)
        if not self.path.exists():
            self.path.mkdir()

    def add(self, name, url, path=None):
        if path is None:
            # Create a relative path
            path = self.path / Path(url).name

        elif not Path(path).is_absolute():
            path = self.path / path

        tf = TestFile(name, path, url)
        setattr(self, name, tf)
        self.files.append(tf)

    def __call__(self):
        return [f() for f in self.files]

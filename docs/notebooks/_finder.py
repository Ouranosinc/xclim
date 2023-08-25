from __future__ import annotations

from pathlib import Path


def _find_current_folder():
    """Find the folder containing the notebooks.

    Needed in order to run the notebooks from the docs/notebooks folder.
    """
    return Path(__file__).absolute().parent

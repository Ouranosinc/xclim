# noqa: D100
from __future__ import annotations

import re


def main():  # noqa: D103

    with open("README.rst") as readme_file:
        readme = readme_file.read()

    with open("HISTORY.rst") as history_file:
        history = history_file.read()
        # remove disallowed directives for PyPI publishing
        history = history.replace(".. autolink-skip::", "")

    hyperlink_replacements = {
        r":issue:`([0-9]+)`": r"`GH/\1 <https://github.com/Ouranosinc/xclim/issues/\1>`_",
        r":pull:`([0-9]+)`": r"`PR/\1 <https://github.com/Ouranosinc/xclim/pull/\1>`_",
        r":user:`([a-zA-Z0-9_.-]+)`": r"`@\1 <https://github.com/\1>`_",
    }

    for search, replacement in hyperlink_replacements.items():
        history = re.sub(search, replacement, history)

    long_description = "\n\n".join([readme, history])

    with open("long_description.rst", "w") as f:
        f.write(long_description)


if __name__ == "__main__":
    main()

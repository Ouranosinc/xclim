#!/usr/bin/env python
"""The setup script."""
import re

from setuptools import find_packages, setup

NAME = "xclim"
DESCRIPTION = "Derived climate variables built with xarray."
URL = "https://github.com/Ouranosinc/xclim"
AUTHOR = "Travis Logan"
AUTHOR_EMAIL = "logan.travis@ouranos.ca"
REQUIRES_PYTHON = ">=3.8.0"
VERSION = "0.36.1"
LICENSE = "Apache Software License 2.0"

with open("README.rst") as readme_file:
    readme = readme_file.read()

with open("HISTORY.rst") as history_file:
    history = history_file.read()

hyperlink_replacements = {
    r":issue:`([0-9]+)`": r"`GH/\1 <https://github.com/Ouranosinc/xclim/issues/\1>`_",
    r":pull:`([0-9]+)`": r"`PR/\1 <https://github.com/Ouranosinc/xclim/pull/\1>`_",
    r":user:`([a-zA-Z0-9_.-]+)`": r"`@\1 <https://github.com/\1>`_",
}

for search, replacement in hyperlink_replacements.items():
    history = re.sub(search, replacement, history)

requirements = [
    "boltons>=20.1",
    "bottleneck>=1.3.1",
    "cf-xarray>=0.6.1",
    "cftime>=1.4.1",
    "Click",
    "dask[array]>=2.6",
    "jsonpickle",
    "numba",
    "numpy>=1.16",
    "pandas>=0.23",
    "pint>=0.10",
    "pyyaml",
    "scikit-learn>=0.21.3",
    "scipy>=1.2",
    "statsmodels",
    "xarray>=0.17",
]

dev_requirements = []
with open("requirements_dev.txt") as dev:
    for dependency in dev.readlines():
        dev_requirements.append(dependency)

KEYWORDS = "xclim climate climatology netcdf gridded analysis"

setup(
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Atmospheric Science",
    ],
    description=DESCRIPTION,
    python_requires=REQUIRES_PYTHON,
    install_requires=requirements,
    license=LICENSE,
    long_description=readme + "\n\n" + history,
    long_description_content_type="text/x-rst",
    include_package_data=True,
    keywords=KEYWORDS,
    name=NAME,
    packages=find_packages(),
    entry_points="""
        [console_scripts]
        xclim=xclim.cli:cli
    """,
    extras_require={"dev": dev_requirements},
    url=URL,
    version=VERSION,
    zip_safe=False,
)

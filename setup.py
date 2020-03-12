#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""The setup script."""
import os

from setuptools import find_packages
from setuptools import setup

NAME = "xclim"
DESCRIPTION = "Derived climate variables built with xarray."
URL = "https://github.com/Ouranosinc/xclim"
AUTHOR = "Travis Logan"
AUTHOR_EMAIL = "logan.travis@ouranos.ca"
REQUIRES_PYTHON = ">=3.6.0"
VERSION = "0.15.0"
LICENSE = "Apache Software License 2.0"

with open("README.rst") as readme_file:
    readme = readme_file.read()

with open("HISTORY.rst") as history_file:
    history = history_file.read()

requirements = [
    "numpy>=1.16",
    "xarray>=0.14.1",
    "scipy>=1.2",
    "numba",
    "pandas>=0.23",
    "cftime>=1.0.4",
    "netCDF4>=1.4",
    "fiona>=1.8",
    "geojson>=2.5.0",
    "shapely>=1.6",
    "geopandas>=0.7",
    "dask[complete]>=2.6",
    "pyproj>=2.5",
    "pint>=0.9",
    "bottleneck~=1.3.1",
    "boltons>=19.0",
    "scikit-learn>=0.21.3",
    "packaging>=20.0",
]

if not os.getenv("READTHEDOCS"):
    requirements.append("rtree>=0.9")

setup_requirements = ["pytest-runner"]

test_requirements = ["pytest", "tox"]

docs_requirements = [
    "sphinx",
    "sphinx-rtd-theme",
    "nbsphinx",
    "pandoc",
    "ipython",
    "ipykernel",
    "jupyter_client",
    "matplotlib",
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
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
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
    setup_requires=setup_requirements,
    test_suite="tests",
    tests_require=test_requirements,
    extras_require={"docs": docs_requirements, "dev": dev_requirements},
    url=URL,
    version=VERSION,
    zip_safe=False,
)

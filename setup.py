#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""The setup script."""
from setuptools import find_packages
from setuptools import setup

NAME = "xclim"
DESCRIPTION = "Derived climate variables built with xarray."
URL = "https://github.com/Ouranosinc/xclim"
AUTHOR = "Travis Logan"
AUTHOR_EMAIL = "logan.travis@ouranos.ca"
REQUIRES_PYTHON = ">=3.5.0"
VERSION = "0.10.7-beta"
LICENSE = "Apache Software License 2.0"

with open("README.rst") as readme_file:
    readme = readme_file.read()

with open("HISTORY.rst") as history_file:
    history = history_file.read()

requirements = [
    "scipy>=1.2",
    "numpy>=1.15",
    "pandas>=0.23",
    "cftime>=1.0.3",
    "netCDF4>=1.4",
    "dask[complete]",
    "bottleneck>=1.2.1",
    "xarray>=0.12.0",
    "pyproj>=1.9.5.1",
    "pint>=0.8",
    "boltons>=18.0",
]

setup_requirements = ["pytest-runner"]

test_requirements = ["pytest", "tox"]

docs_requirements = ["sphinx", "guzzle-sphinx-theme", "nbsphinx", "pandoc", "ipython"]

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
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering :: Atmospheric Science",
    ],
    description=DESCRIPTION,
    python_requires=REQUIRES_PYTHON,
    install_requires=requirements,
    license=LICENSE,
    long_description=readme + "\n\n" + history,
    long_description_content_type="text/markdown",
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

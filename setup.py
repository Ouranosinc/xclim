#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [
    'scipy>=1.2',
    'numpy>=1.15',
    'pandas>=0.23',
    'cftime>=1.0.3',
    'netCDF4>=1.4',
    'dask[complete]',
    'bottleneck>=1.2.1',
    'xarray>=0.12.0',
    'pyproj>=1.9.5.1',
    'pint>=0.8',
    'boltons>=18.0',
    'inspect2',
    'unittest2',
    'six',
]

setup_requirements = ['pytest-runner', ]

test_requirements = ['pytest', 'tox', ]

docs_requirements = ['Sphinx', 'guzzle-sphinx-theme', ]

KEYWORDS = "xclim climate climatology netcdf gridded analysis"

setup(
    author="Travis Logan",
    author_email='logan.travis@ouranos.ca',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Natural Language :: English',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering :: Atmospheric Science',
    ],
    description="Derived climate variables built with xarray.",
    install_requires=requirements,
    license="Apache Software License 2.0",
    long_description=readme + '\n\n' + history,
    long_description_content_type='text/markdown',
    include_package_data=True,
    keywords=KEYWORDS,
    name='xclim',
    packages=find_packages(),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    extras_require={'docs': docs_requirements},
    url='https://github.com/Ouranosinc/xclim',
    version='0.9-beta',
    zip_safe=False,
)

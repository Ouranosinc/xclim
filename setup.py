#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

# with open('requirements.txt') as req_file:
#     requirements = req_file.read().split('\n')

requirements = ['numpy>=1.15',
                'scipy>=1.1',
                'pandas>=0.22',
                'cftime>=1.0',
                'netCDF4>=1.4',
                'dask>=0.18',
                'xarray>=0.10',
                'bottleneck>=1.2',
                'pint>=0.8',
                'inspect2>=0.1',
                'unittest2>=1.1',
                'six>=1.11']

setup_requirements = ['pytest-runner', ]

test_requirements = ['pytest', 'tox', ]

docs_requirements = ['Sphinx', 'guzzle-sphinx-theme', ]

KEYWORDS = "xclim climate climatology netcdf gridded analysis"

setup(
    author="Travis Logan",
    author_email='logan.travis@ouranos.ca',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Natural Language :: English',
        "Programming Language :: Python :: 2",
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering :: Atmospheric Science',
    ],
    description="Derived climate variables built with xarray.",
    install_requires=requirements,
    license="Apache Software License 2.0",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords=KEYWORDS,
    name='xclim',
    packages=find_packages(include=['xclim']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    extras_require={'docs': docs_requirements},
    url='https://github.com/Ouranosinc/xclim',
    version='0.6-alpha',
    zip_safe=False,
)

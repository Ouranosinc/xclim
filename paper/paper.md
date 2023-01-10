---
title: 'Xclim: Xarray-based climate data analytics'
tags:
  - climate change
  - climate indices
  - downscaling
  - climatology
  - python
  - xarray
authors:
 - name: Pascal Bourgault
   orcid: 0000-0003-1192-0403
   affiliation: 1
 - name: David Huard
   orcid: 0000-0003-0311-5498
   affiliation: 1
 - name: Trevor James Smith
   orcid: 0000-0001-5393-8359
   affiliation: 1
 - name: Travis Logan
   orcid: 0000-0002-2212-9580
   affiliation: 1
 - name: Abel Aoun
   orcid: 0000-0003-2289-2890
   affiliation: 2
 - name: Juliette Lavoie
   orcid: 0000-0002-4708-5182
   affiliation: 1
 - name: Éric Dupuis
   orcid: 0000-0001-7976-4596
   affiliation: 1
affiliations:
 - name: Ouranos Consortium, Montréal, Québec, Canada.
   index: 1
 - name: Centre européen de recherche et de formation avancée en calcul scientifique (CERFACS)
   index: 2
date: 1 December 2022
bibliography: paper.bib
---

# Summary

`xclim` is a Python library that enables computation of climate indicators over large, heterogeneous data sets. It is built using `xarray` objects and operations, and relies on community conventions for data formatting and metadata attributes. `xclim` is meant as a tool to facilitate both climate science research and the delivery of operational climate services and products. In addition to climate indicator calculations, `xclim` also includes utilities for bias correction and statistical adjustment, ensemble analytics, and model diagnostics.

# Statement of need

Researchers and climate service providers analyze data from large ensembles of Earth System Model (ESM) simulations. These analyses typically include model comparisons with observations, bias-correction and statistical adjustment, computation of various climate indicators and diagnostics, and ensemble statistics. As the number of models contributing to these ensembles grows, so does the complexity of the code required to deal with model idiosyncrasies, outlier detection, unit conversion, etc. In addition, growing ensemble sizes and improvements in the spatiotemporal resolution of ESMs raise the computational costs of running those analyses. `xclim` is designed to meet the operational needs of climate service providers by offering algorithms for over 150 climate indicators, multiple downscaling algorithms, and ensemble statistics.

The development of `xclim` started at Ouranos in 2018 out of the need to deliver data for a pan-Canadian atlas of climate indicators. In-house specialists had different implementations for the same indicators, and there was a desire to adopt a common library that would tie together investments in research and development with operational production capabilities. At the time, the package that was closest to meeting these requirements was `icclim` [@icclim], a library developed in the context of the [European Climate Assessment & Dataset](https://www.ecad.eu/) project, with the purpose of monitoring and analyzing changes in climate extremes. It was not, however, designed to be easily extensible, and we believed the indicators they offered could be written more succinctly and computed more efficiently by relying on objects and primitives from `xarray` [@Hoyer:2017]. `xclim` started as a reimplementation of `icclim` with an `xarray` backend, drawing inspiration from `MetPy` [@metpy], and eventually grew to include other algorithms routinely used in climate data analysis.

`xclim` is meant to be one component in a larger software ecosystem for climate data analysis. Other libraries often used in tandem with `xclim` are `clisops` [@clisops] for spatial subsetting and averaging utilities, and `xESMF` [@xesmf] for spatial regridding.

# Key Features

## Climate indicators calculations

An `Indicator` class is built around a `compute` function defining a climate indicator. It performs health checks on input data (units, time frequency, outlier detection), handles missing values, and assigns attributes to the output, so it complies with the Climate and Forecast (CF) Convention [@Hassel:2017]. Indicators can be customized using a context manager, class inheritance, or through a YAML file - the latter allowing the creation of custom collections of indicators for batch processing.

## Statistical adjustment and bias correction

The `xclim.sdba` subpackage provides different algorithms to adjust the distribution of simulated variables to observed variables. It adopts a train / adjust paradigm, where corrections are first calculated, then applied to the target data or saved for later use. Most methods support additive or multiplicative corrections, different time groupings (seasonal, monthly, or daily with a rolling window). Correction factors can be interpolated between time groupings to avoid discontinuities in the corrected data.

## Ensemble analysis

The `xclim.ensembles` subpackage bundles utilities to facilitate the analysis of results from multiple models. It includes functions to reduce the ensemble size using clustering algorithms, metrics of ensemble robustness, and significance of climate change signals.

## Spatial analogs

The `xclim.analogs` subpackage offers tools to find spatial climate analogs using a selection of distribution comparison algorithms.

## Other utilities
`xclim.core.dataflags` can be used to find abberant values in climate data and  `xclim.cli` implements a command-line interface to most features to facilitate the use of xclim in scripted workflows.

# Projects using `xclim`

`xclim` is core component of Finch [@finch], a server hosting climate analytics services behind a Web Processing Services (WPS) interface. `Finch` itself is part of the computational backend of [climatedata.ca](https://climatedata.ca), an online data portal to access, visualize and analyze climate data over Canada. `xclim` is now also a core component of `icclim` from version 5.0, which itself is used in the `climate4impact` project [@Page:2022]. The statistical adjustment tools of  `xclim` are also being used by the [Climate Impact Lab](https://impactlab.org/) to downscale and adjust CMIP6 simulations on HPCs for climate impact studies.

# Acknowledgements

`xclim` was developed thanks to the financial and strategic contributions of the Canadian Center for Climate Services and the Ouranos Consortium. We acknowledge contributions from Marie-Pier Labonté, David Caron, Jwen-Fai Low, Raquel Alegre, Clair Barnes, Sébastien Biner, Philippe Roy, Carsten Ehbrecht, Tom Keel, Ludwig Lierhammer, Jamie Quinn, Dougie Squire, Ag Stephens, Maliko Tanguy, Jeremy Fyke, Yannick Rousseau, and Chistopher Whelan.

# References

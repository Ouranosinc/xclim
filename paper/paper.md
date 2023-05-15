---
title: 'xclim: xarray-based climate data analytics'
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
   corresponding: true
 - name: Abel Aoun
   orcid: 0000-0003-2289-2890
   affiliation: 2
 - name: Juliette Lavoie
   orcid: 0000-0002-4708-5182
   affiliation: 1
 - name: Éric Dupuis
   orcid: 0000-0001-7976-4596
   affiliation: 1
 - name: Gabriel Rondeau-Genessse
   orcid: 0000-0003-3389-9406
   affiliation: 1
 - name: Raquel Alegre
   orcid: 0000-0002-6081-0721
   affiliation: 3
 - name: Clair Barnes
   orcid: 0000-0002-7806-7913
   affiliation: 3
 - name: Alexis Beaupré Laperrière
   affiliation: 1
 - name: Sébastien Biner
   orcid: 0000-0001-7515-490X
   affiliation: 12
 - name: David Caron
   affiliation: 10
 - name: Carsten Ehbrecht
   affiliation: 4
 - name: Jeremy Fyke
   orcid: 0000-0002-4522-3019
   affiliation: 5
 - name: Tom Keel
   orcid: 0000-0001-9193-5271
   affiliation: 3
 - name: Marie-Pier Labonté
   orcid: 0000-0003-0738-3940
   affiliation: 1
 - name: Ludwig Lierhammer
   orcid: 0000-0002-7207-0003
   affiliation: 6
 - name: Jwen-Fai Low
   affiliation: 13
 - name: Jamie Quinn
   affiliation: 3
 - name: Philippe Roy
   affiliation: 11
 - name: Dougie Squire
   orcid: 0000-0003-3271-6874
   affiliation: 7
 - name: Ag Stephens
   orcid: 0000-0002-1038-7988
   affiliation: 8
 - name: Maliko Tanguy
   orcid: 0000-0002-1516-6834
   affiliation: 9
 - name: Christopher Whelan
   affiliation: 14
affiliations:
 - name: Ouranos Consortium, Montréal, Québec, Canada
   index: 1
 - name: Centre européen de recherche et de formation avancée en calcul scientifique (CERFACS), France
   index: 2
 - name: University College London (UCL), United Kingdom
   index: 3
 - name: Deutsches Klimarechenzentrum (DKRZ), Germany
   index: 4
 - name: Environment and Climate Change Canada (ECCC), Canada
   index: 5
 - name: Helmholtz-Zentrum Hereon, Germany
   index: 6
 - name: Commonwealth Scientific and Industrial Research Organisation (CSIRO), Australia
   index: 7
 - name: Centre for Environmental Data Analysis (CEDA), United Kingdom
   index: 8
 - name: Centre for Ecology & Hydrology (CEH), United Kingdom
   index: 9
 - name: Jakarto, Montréal, Québec, Canada
   index: 10
 - name: Institut de recherche d'Hydro-Québec (IREQ), Québec, Canada
   index: 11
 - name: Hydro-Québec (HQ), Québec, Canada
   index: 12
 - name: Independent Researcher, Canada
   index: 13
 - name: Independent Researcher, United States
   index: 14
date: 8 May 2023
bibliography: paper.bib
---

# Summary

`xclim` is a Python library that enables computation of climate indicators over large, heterogeneous data sets. It is built using `xarray` objects and operations, can seamlessly benefit from the parallelization handling provided by `dask`, and relies on community conventions for data formatting and metadata attributes. `xclim` is meant as a tool to facilitate both climate science research and the delivery of operational climate services and products. In addition to climate indicator calculations, `xclim` also includes utilities for bias correction and statistical adjustment, ensemble analytics, model diagnostics, data quality assurance, and metadata standards compliance.

# Statement of need

Researchers and climate service providers analyse data from large ensembles of Earth System Model (ESM) simulations. These analyses typically include model comparisons with observations, bias-correction and statistical adjustment, computation of various climate indicators and diagnostics, and ensemble statistics. As the number of models contributing to these ensembles grows, so does the complexity of the code required to deal with model idiosyncrasies, outlier detection, unit conversion, etc. In addition, growing ensemble sizes and advancements in the spatiotemporal resolution of ESMs further raises the computational costs of running those analyses. `xclim` is designed to meet the operational needs of climate service providers by offering algorithms for over 150 climate indicators, multiple downscaling algorithms, ensemble statistics, and other associated utilities.

The development of `xclim` started in 2018 at [Ouranos](https://www.ouranos.ca), a consortium on regional climatology and adaptation to climate change based in Montréal, Québec, from the need to deliver data for a pan-Canadian atlas of climate indicators. In-house specialists at Ouranos had different implementations for the same indicators, and there was a desire to adopt a common library that would tie together investments in research and development with operational production capabilities. At the time, the package that was closest to meeting these requirements was `icclim` [@icclim], a library developed within the context of the [European Climate Assessment & Dataset](https://www.ecad.eu/) project, whose purpose was to monitor and analyze changes in climate extremes. It was not, however, designed to be easily extensible, and we believed the indicators they offered could be written more succinctly and computed more efficiently by relying on objects and primitives from `xarray` [@Hoyer:2017], with distributed computation and scheduling via `dask` [@dask:2016]. `xclim` started as a reimplementation of `icclim` with an `xarray` backend, drawing inspiration from projects like `MetPy` [@metpy], and eventually grew to include other algorithms routinely used in climate data analysis, both simple and complex.

`xclim` is intended to be one component in a larger software ecosystem for climate data analysis. Other libraries often used in tandem with `xclim` are `clisops` [@clisops], a spatiotemporal subsetting and averaging library (originally a fork of `xclim`'s subsetting module), and `xESMF` [@xesmf], a PANGEO-developed library for spatial regridding.

# Key Features

## Climate indicators calculations

An `Indicator` class is built around a `compute` function defining a climate indicator. It performs health checks on input data (units, time frequency, outlier detection), handles missing values, and assigns attributes to the output, complying to the Climate and Forecast (CF) metadata Conventions [@Hassel:2017]. Indicators can be customized using a context manager, by class inheritance, or through a YAML file—the latter allowing for the creation of custom collections of indicators for batch processing.

## Statistical adjustment and bias correction

The `xclim.sdba` subpackage provides different algorithms to adjust the distribution of simulated variables to observed variables. It adopts a train / adjust paradigm, where corrections are first calculated, then applied to the target data or saved for later use. Most methods support additive or multiplicative corrections, different time groupings (seasonal, monthly, or daily with a rolling window). Correction factors can be interpolated between time groupings to avoid discontinuities in the corrected data.

## Ensemble analysis

The `xclim.ensembles` subpackage bundles utilities to facilitate the analysis of results from multiple models. It includes functions to reduce the ensemble size using clustering algorithms, metrics of ensemble robustness, and significance of climate change signals.

## Spatial analogs

The `xclim.analogs` subpackage offers tools to find spatial climate analogs using a selection of distribution comparison algorithms.

## Internationalization tools

In order to better support the international community, `xclim` provides methods for building dynamic multilingual metadata translations via the `xclim.core.locales` module. While French is currently the only translation officially supported, other languages can be extended via JSON-based indicator field mappings.

## Other utilities

Among the various modules within `xclim`, a few merit explicit mention;
 - `xclim.cli` implements a command-line interface to most features to enable the use of xclim in shell-scripted workflows;
 - various pseudo-indices provided in `xclim.core.dataflags` can be used to find aberrant values in climate data
 - `xclim.core.datachecks` and `xclim.core.cfchecks` comprise many lower-level functions for evaluating units, dataset consistency, and completeness of metadata;
 - and `xclim.core.calendar` provides numerous tools for standardizing the various calendar systems found in modelled climate datasets.

# Projects using `xclim`

`xclim` is core component of Finch [@finch], a server hosting climate analytics services behind a Web Processing Services (WPS) interface. `Finch` itself is part of the computational backend of [ClimateData.ca](https://climatedata.ca), an online data portal to access, visualize and analyze climate data over Canada. `xclim` is now also a core component of `icclim` from version 5.0, which itself is used in the `climate4impact` project [@Page:2022]. The statistical adjustment tools of `xclim` are also being used by the [Climate Impact Lab](https://impactlab.org/) to downscale and adjust CMIP6 simulations on HPCs for climate impact studies.

# Acknowledgements

`xclim` was developed thanks to the financial and strategic contributions of the [Canadian Center for Climate Services](https://www.canada.ca/en/environment-climate-change/services/climate-change/canadian-centre-climate-services.html) and the [Ouranos Consortium](https://www.ouranos.ca). We also acknowledge the contributions from Marie-Pier Labonté, David Caron, Jwen-Fai Low, Raquel Alegre, Clair Barnes, Sébastien Biner, Philippe Roy, Carsten Ehbrecht, Tom Keel, Ludwig Lierhammer, Jamie Quinn, Dougie Squire, Ag Stephens, Maliko Tanguy, Jeremy Fyke, Yannick Rousseau, Christian Jauvin, and Chistopher Whelan, as well as our user base who regularly provide valuable bug reports and enhancement / support requests.

# References

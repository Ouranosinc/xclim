# -*- coding: utf-8 -*-
# noqa: D205,D400
"""
Indicators module
=================

Indicators are the main tool xclim provides to compute climate indices. In contrast
to the function defined in `xclim.indices`, Indicators add a layer of health checks
and metadata handling. Indicator objects are split into realms : atmos, land and
seaIce.

Virtual modules are also inserted here. A normal installation of xclim comes with
three virtual modules:
 - :py:mod:`xclim.indicators.cf`, Indicators defined in `cf-index-meta`.
 - :py:mod:`xclim.indicators.icclim`, Indicators defined by ECAD, as found in  python package Icclim.
 - :py:mod:`xclim.indicators.anuclim`, Indicators of the Australian National University's Fenner School of Environment and Society.
"""
# The actual code for importing virtual modules is in the top-level __init__.
# This is for import reasons: we need to make sure all normal indicators are created before.

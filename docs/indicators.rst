==================
Climate Indicators
==================

:py:class:`xclim.core.indicator.Indicator` instances essentially perform the same computations as the functions
found in the :mod:`xclim.indices` library, but also run a number of health checks on input data
and assign attributes to the output arrays. So for example, if there are missing values in
a time series, indices won't notice, but indicators will return NaNs for periods with missing
values (depending on the missing values algorithm selected, see :ref:`checks:Missing values identification`). Indicators also check that the input data has the expected frequency (e.g. daily) and that
it is indeed the expected variable (e.g. a precipitation flux). The output is assigned attributes
that conform as much as possible with the `CF-Convention`_.

Indicators are split into realms (atmos, land, seaIce), according to the variables they operate on.
See :ref:`notebooks/extendxclim:Defining new indicators` for instruction on how to create your own indicators. This page
allows a simple free text search of all indicators. Click on the python names to get to the complete docstring of each indicator.

.. raw:: html

    <input type="text" id="queryInput" onkeyup="indFilter()" placeholder="Search for titles, variables or keywords...">
    <input type="checkbox" id="incVirtMod" onchange="indFilter()"><label id="incVirtModLbl" for="virtualModules">Include virtual submodules in results.</label>
    <div id="indTable">
    </div>

..
    Filling of the table and search is done by scripts in _static/indsearch.js which are added through _templates/layout.html
    the data comes from indicators.json which is created by conf.py.

.. _CF-Convention: http://cfconventions.org/

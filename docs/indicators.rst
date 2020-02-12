==================
Climate indicators
==================

:class:`xclim.utils.Indicator` instances essentially perform the same computations as the functions
found in the :mod:`indices` library, but also run a number of health checks on input data
and assign attributes to the output arrays. So for example, if there are missing values in
a time series, indices won't notice, but indicators will return NaNs for periods with missing
values. Indicators also check that the input data has the expected frequency (e.g. daily) and that
it is indeed the expected variable (e.g. a precipitation flux). The output is assigned attributes
that conform as much as possible with the `CF-Convention`_.

Indicators are split into realms (atmos, land, seaIce), according to the variables they operate on.


atmos: Atmosphere
=================

.. raw:: html
    <dl>
   {% for ind in indicators['atmos'] %}
     <dt><b>{{ ind.long_name }}</b>  (<var>atmos.{{ ind.identifier | safe}}</var>) </dt>
     <dd>{{ ind.description }}</dd>
   {% endfor %}
   </dl>

land: Land surface
==================

.. raw:: html
    <dl>
   {% for ind in indicators['land'] %}
     <dt><b>{{ ind.long_name | e }}</b>  (<var>land.{{ ind.identifier | safe}}</var>) </dt>
     <dd>{{ ind.description | e }}</dd>
   {% endfor %}
   </dl>

seaIce: Sea ice
===============

.. raw:: html
    <dl>
   {% for ind in indicators['seaIce'] %}
     <dt><b>{{ ind.long_name }}</b>  (<var>seaIce.{{ ind.identifier | safe}}</var>) </dt>
     <dd>{{ ind.description }}</dd>
   {% endfor %}
   </dl>



.. _CF-Convention: http://cfconventions.org/

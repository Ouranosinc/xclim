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
lists all indicators with a summary description, click on the names to get to the complete docstring of each indicator.

atmos: Atmosphere
=================

.. raw:: html

   <dl>
   {% for indname, ind in indicators['atmos'].items() %}
     <dt><code>atmos.{{ indname | safe}}</code> : <a class="reference_internal" href="api.html#xclim.indicators.atmos.{{ indname }}" title="{{ indname }}"><b>{{ ind.title }}</b></a></dt>
     <dd>
     {% if ind.identifier != indname %}<b>Id: </b> {{ ind.identifier }} <br>{% endif %}
     <b>Description: </b> {{ ind.abstract }} <br>
     <b>Based on </b><a class="reference internal" href="indices.html#{{ ind.function }}" title="{{ ind.function }}"><code class="xref">{{ ind.function }}</code></a> <br>
     <b>Produces: </b> {% for var in ind['outputs'] %} <code>{{ var['var_name'] }}: {{ var['long_name'] }} [{{ var['units'] }}]</code> {% endfor %}
     </dd>
   {% endfor %}
   </dl>


land: Land surface
==================

.. raw:: html

   <dl>
   {% for indname, ind in indicators['land'].items() %}
     <dt><code>land.{{ indname | safe}}</code> : <a class="reference_internal" href="api.html#xclim.indicators.land.{{ indname }}" title="{{ indname }}"><b>{{ ind.title }}</b></a></dt>
     <dd>
     {% if ind.identifier != indname %}<b>Id: </b> {{ ind.identifier }} <br>{% endif %}
     <b>Description: </b> {{ ind.abstract }} <br>
     <b>Based on </b><a class="reference internal" href="indices.html#{{ ind.function }}" title="{{ ind.function }}"><code class="xref">{{ ind.function }}</code></a> <br>
     <b>Produces: </b> {% for var in ind['outputs'] %} <code>{{ var['var_name'] }}: {{ var['long_name'] }} [{{ var['units'] }}]</code> {% endfor %}
     </dd>
   {% endfor %}
   </dl>


seaIce: Sea ice
===============

.. raw:: html

   <dl>
   {% for indname, ind in indicators['seaIce'].items() %}
     <dt><code>seaIce.{{ indname | safe}}</code> : <a class="reference_internal" href="api.html#xclim.indicators.seaIce.{{ indname }}" title="{{ indname }}"><b>{{ ind.title }}</b></a></dt>
     <dd>
     {% if ind.identifier != indname %}<b>Id: </b> {{ ind.identifier }} <br>{% endif %}
     <b>Description: </b> {{ ind.abstract }} <br>
     <b>Based on </b><a class="reference internal" href="indices.html#{{ ind.function }}" title="{{ ind.function }}"><code class="xref">{{ ind.function }}</code></a> <br>
     <b>Produces: </b> {% for var in ind['outputs'] %} <code>{{ var['var_name'] }}: {{ var['long_name'] }} [{{ var['units'] }}]</code> {% endfor %}
     </dd>
   {% endfor %}
   </dl>

.. _CF-Convention: http://cfconventions.org/


Virtual submodules
==================

.. automodule:: xclim.indicators.cf
    :noindex:

.. raw:: html

   <dl>
   {% for indname, ind in indicators['cf'].items() %}
     <dt><code>cf.{{ indname | safe}}</code> : <a class="reference_internal" href="api.html#xclim.indicators.cf.{{ indname }}" title="{{ indname }}"><b>{{ ind.title }}</b></a></dt>
     <dd>
     {% if ind.identifier != indname %}<b>Id: </b> {{ ind.identifier }} <br>{% endif %}
     <b>Description: </b> {{ ind.abstract }} <br>
     <b>Based on </b><a class="reference internal" href="indices.html#{{ ind.function }}" title="{{ ind.function }}"><code class="xref">{{ ind.function }}</code></a> <br>
     <b>Produces: </b> {% for var in ind['outputs'] %} <code>{{ var['var_name'] }}: {{ var['long_name'] }} [{{ var['units'] }}]</code> {% endfor %}
     </dd>
   {% endfor %}
   </dl>

.. automodule:: xclim.indicators.icclim
    :noindex:

.. raw:: html

   <dl>
   {% for indname, ind in indicators['icclim'].items() %}
     <dt><code>icclim.{{ indname | safe}}</code> : <a class="reference_internal" href="api.html#xclim.indicators.icclim.{{ indname }}" title="{{ indname }}"><b>{{ ind.title }}</b></a></dt>
     <dd>
     {% if ind.identifier != indname %}<b>Id: </b> {{ ind.identifier }} <br>{% endif %}
     <b>Description: </b> {{ ind.abstract }} <br>
     <b>Based on </b><a class="reference internal" href="indices.html#{{ ind.function }}" title="{{ ind.function }}"><code class="xref">{{ ind.function }}</code></a> <br>
     <b>Produces: </b> {% for var in ind['outputs'] %} <code>{{ var['var_name'] }}: {{ var['long_name'] }} [{{ var['units'] }}]</code> {% endfor %}
     </dd>
   {% endfor %}
   </dl>

.. automodule:: xclim.indicators.anuclim
    :noindex:

.. raw:: html

   <dl>
   {% for indname, ind in indicators['anuclim'].items() %}
     <dt><code>anuclim.{{ indname | safe}}</code> : <a class="reference_internal" href="api.html#xclim.indicators.anuclim.{{ indname }}" title="{{ indname }}"><b>{{ ind.title }}</b></a></dt>
     <dd>
     {% if ind.identifier != indname %}<b>Id: </b> {{ ind.identifier }} <br>{% endif %}
     <b>Description: </b> {{ ind.abstract }} <br>
     <b>Based on </b><a class="reference internal" href="indices.html#{{ ind.function }}" title="{{ ind.function }}"><code class="xref">{{ ind.function }}</code></a> <br>
     <b>Produces: </b> {% for var in ind['outputs'] %} <code>{{ var['var_name'] }}: {{ var['long_name'] }} [{{ var['units'] }}]</code> {% endfor %}
     </dd>
   {% endfor %}
   </dl>

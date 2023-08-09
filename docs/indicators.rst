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

    <input type="text" id="queryInput" onkeyup="indFilter()" placeholder="Search for titles..">

    <div id="indTable">
    {% for realm, indlist in indicators.items() %}
    {% for indname, ind in indlist.items() %}
        <div class="indElem">
            <div class="indHeader">
                <b class="indTitle">{{ ind.title }}</b>
                <a class="reference_internal indName" href="api.html#xclim.indicators.{{ realm }}.{{ indname }}" title="{{ indname }}">
                    <code>{{ realm }}.{{ indname | safe }}</code>
                </a>
            </div>
            <div class="indVars">Uses: {% for var in ind.vars %}<code class="indVarname">{{ var }}</code> {% endfor %}</div>
        </div>
    {% endfor %}
    {% endfor %}
    </div>

     <script>
    function indFilter() {
      // Declare variables
      var input, filter, table, elems, title, i, txtValue;
      input = document.getElementById("queryInput");
      filter = input.value.toUpperCase();
      table = document.getElementById("indTable");
      elems = table.getElementsByClassName("indElem");

      // Loop through all table rows, and hide those who don't match the search query
      for (i = 0; i < elems.length; i++) {
        title = elems[i].getElementsByClassName("indTitle")[0];
        if (title) {
          txtValue = title.textContent || title.innerText;
          if (txtValue.toUpperCase().indexOf(filter) > -1) {
            elems[i].style.display = "";
          } else {
            elems[i].style.display = "none";
          }
        }
      }
    }
    </script>


.. _CF-Convention: http://cfconventions.org/

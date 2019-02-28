.. _table:

Table of indicators
===================

Click on the indicator name to get a full description of the indicator.

.. raw:: html

   <table class="indices" style="width:100%">
   <tr>
   <th>Name</th>
   <th>Description</th>
   </tr>
   {% for ind in indicators %}
   <tr>
   <td class="name"><a href="xclim.html#xclim.indices.{{ ind.standard_name | trim }}">{{ ind.long_name | safe }}</a></td>
   <td>{{ ind.description | safe }}</td>
   </tr>
   {% endfor %}
   </table>

   <style>
    td, th {
      border: 1px solid #dddddd;
      text-align: left;
      padding: 8px;
    }

    td.name {
        font-weight: 500;
    }

    tr:nth-child(even) {
      background-color: #dddddd;
    }
   </style>




..  {% for ind in indicators %}
    :class:`{{ind.identifier}}` : **{{ ind.long_name | trim }}**

      {{ ind.description | trim }}

    {{ ind.notes }}


    {% endfor %}


.. _table:

Table of indicators
===================

.. raw:: html

   {% for realm in indicators.keys() %}
   <h3>xclim.{{ realm }}</h3>
   <table class="indices" style="width:100%">
   <tr>
   <th>Name</th>
   <th>Description</th>
   </tr>
   {% for indnae, ind in indicators[realm].items() %}
   <tr>
   <td class="name">{{ ind.long_name | safe }}</td>
   <td>{{ ind.description | safe }}</td>
   </tr>
   {% endfor %}
   </table>
   {% endfor %}
   <style>
    td, th {
      border: 1px solid #dddddd;
      text-align: left;
      padding: 8px;
    }

    table {
      margin-bottom: 30px;
    }
    td.name {
        font-weight: 500;
    }

    tr:nth-child(even) {
      background-color: #dddddd;
    }
   </style>

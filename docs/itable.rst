.. _table:

Table of indicators
===================

.. raw:: html

   <table class="indices" style="width:100%">
   <tr>
   <th>Name</th>
   <th>Description</th>
   </tr>
   {% for ind in indicators %}
   <tr>
   <td class="name">{{ ind.long_name | safe }}</td>
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




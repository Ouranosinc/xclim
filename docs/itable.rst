.. _table:

List of indicators
==================

{% for ind in indicators %}
:class:`{{ind.identifier}}` : **{{ ind.long_name | trim }}**

  {{ ind.description | trim }}

{{ ind.notes }}


{% endfor %}


Alias
=====

:doc:`Processor <../../index>` that defines named groups of columns. Once an alias
is registered, later processors can refer to the group name wherever a column
selection is expected, and it expands to the listed columns. This keeps long column
lists out of repeated configuration.

Configuration
-------------

Selected as ``name: alias``.

.. list-table::
   :header-rows: 1
   :widths: 12 65 13 10

   * - Option
     - Description
     - Type
     - Default
   * - ``map``
     - Mapping of group name to the list of columns it expands to.
     - mapping
     - required

.. code-block:: yaml

   - name: alias
     kwargs:
       map:
         position: [ x, y, z ]

Downstream usage example:

.. code-block:: yaml

   - name: copy
     kwargs:
       to: truth
       by: event_id
       cols: position  # equivalent to passing [x, y, z]
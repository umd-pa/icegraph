Map
===

:doc:`Processor <../../index>` that remaps the values of a column according to a
lookup table, optionally writing the result to a new column. It is used to recode
labels, for example collapsing particle types into class indices.

Configuration
-------------

Selected as ``name: map``.

.. list-table::
   :header-rows: 1
   :widths: 12 62 16 10

   * - Option
     - Description
     - Type
     - Default
   * - ``col``
     - Column whose values are remapped.
     - str | int
     - required
   * - ``map``
     - Lookup table of old value to new value.
     - mapping
     - required
   * - ``strict``
     - Fail if a value is missing from the table (otherwise it is left unchanged).
     - bool
     - ``true``
   * - ``out``
     - Destination column; defaults to overwriting ``col``.
     - str | int
     - overwrite

.. code-block:: yaml

   - name: map
     kwargs:
       col: pdg
       map: { 12: 0, 14: 1, 16: 2 }
       out: flavor

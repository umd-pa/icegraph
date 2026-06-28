Pivot
=====

:doc:`Processor <../../index>` that reshapes data from long form to wide form. Rows
sharing an index are collapsed, with the distinct entries of a category column
becoming separate columns. It is used to turn key/value style rows into a flat
per-event table.

Configuration
-------------

Selected as ``name: pivot``.

.. list-table::
   :header-rows: 1
   :widths: 12 65 13 10

   * - Option
     - Description
     - Type
     - Default
   * - ``index``
     - Columns identifying a row in the wide output.
     - column(s)
     - required
   * - ``col``
     - Column whose distinct values become the new columns.
     - str
     - required
   * - ``values``
     - Column supplying the values placed under the new columns.
     - str
     - required

.. code-block:: yaml

   - name: pivot
     kwargs:
       index: event_id
       col: feature_name
       values: feature_value

Compress
========

:doc:`Processor <../../index>` that concatenates columns and stacks the rows of
each group into per-group 2D arrays.

Configuration
-------------

Selected as ``name: compress``.

.. list-table::
   :header-rows: 1
   :widths: 16 58 16 10

   * - Option
     - Description
     - Type
     - Default
   * - ``to``
     - Destination frame for the compressed result.
     - str
     - required
   * - ``by``
     - Columns to group rows by (one output row per group).
     - column(s)
     - required
   * - ``cols``
     - Columns concatenated and stacked into the per-group array.
     - column(s)
     - required
   * - ``out``
     - Output column holding the stacked array.
     - str | int
     - required
   * - ``dtype``
     - Data type of the stacked array.
     - str
     - ``null``
   * - ``record_names``
     - Store the source column names in the attributes.
     - bool
     - ``true``
   * - ``record_offset``
     - Store the per-column offsets in the attributes.
     - bool
     - ``true``

.. code-block:: yaml

   - name: compress
     kwargs:
       to: pulses
       by: event_id
       cols: [ x, y, z, charge, time ]
       out: features

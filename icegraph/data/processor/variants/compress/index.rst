Compress
========

:doc:`Processor <../../index>` that concatenates columns and stacks the rows of
each group into per-group 2D arrays.

Packing many columns into one array gives them all a single dtype, so alongside the
column names it records the dtype each one had, letting a reader put a column back
the way it was written. ``override_dtypes`` records the packed dtype instead, for a
pack where the stored type is the one that matters.

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
     - Columns concatenated and stacked into the per-group array. Use '__all__' to compress all non-``by`` cols.
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
   * - ``override_dtypes``
     - Record every column as the dtype it was packed into, rather than the one it
       came in with. For a pack whose stored type is the one consumers should see.
     - bool
     - ``false``

.. code-block:: yaml

   - name: compress
     kwargs:
       to: pulses
       by: event_id
       cols: [ x, y, z, charge, time ]
       out: features

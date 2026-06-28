Commit
======

:doc:`Processor <../../index>` that writes finished columns from the active frame
into the pipeline's output, which the writer later persists. The committed columns must
form a unique key over the given identifier columns.

Configuration
-------------

Selected as ``name: commit``.

.. list-table::
   :header-rows: 1
   :widths: 12 65 13 10

   * - Option
     - Description
     - Type
     - Default
   * - ``ids``
     - Columns forming the unique key the committed rows are aligned on.
     - column(s)
     - required
   * - ``cols``
     - Columns to write into the output.
     - column(s)
     - required

.. code-block:: yaml

   - name: commit
     kwargs:
       ids: event_id
       cols: [ features, edge_index, edge_attr ]

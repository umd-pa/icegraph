Stats
=====

:doc:`Processor <../../index>` that computes summary statistics over the given
columns and stores them in the dataset attributes.

Configuration
-------------

Selected as ``name: stats``.

.. list-table::
   :header-rows: 1
   :widths: 12 63 15 10

   * - Option
     - Description
     - Type
     - Default
   * - ``cols``
     - Columns to compute statistics over.
     - column(s)
     - required
   * - ``stats``
     - Names of the statistics to compute.
     - list[str]
     - required

.. code-block:: yaml

   - name: stats
     kwargs:
       cols: [ charge, time ]
       stats: [ mean, m2, min, max ]

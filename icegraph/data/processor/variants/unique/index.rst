Unique
======

:doc:`Processor <../../index>` that records the distinct values present in each of
the given columns and stores them in the dataset attributes.

Configuration
-------------

Selected as ``name: unique``.

.. list-table::
   :header-rows: 1
   :widths: 12 65 13 10

   * - Option
     - Description
     - Type
     - Default
   * - ``cols``
     - Columns whose distinct values are recorded.
     - column(s)
     - required

.. code-block:: yaml

   - name: unique
     kwargs:
       cols: [ flavor ]

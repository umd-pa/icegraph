Fill
====

:doc:`Processor <../../index>` that adds a new column, or overwrites an existing
one, with a constant value in every row.

Configuration
-------------

Selected as ``name: fill``.

.. list-table::
   :header-rows: 1
   :widths: 12 60 18 10

   * - Option
     - Description
     - Type
     - Default
   * - ``col``
     - Column to create or overwrite.
     - str
     - required
   * - ``value``
     - Constant value written to every row.
     - str | int | float | bool
     - required
   * - ``dtype``
     - Data type the column is stored as.
     - str
     - required

.. code-block:: yaml

   - name: fill
     kwargs:
       col: bundle
       value: 1
       dtype: int64

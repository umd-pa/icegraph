Filter
======

:doc:`Processor <../../index>` that filters data based on a column value and a conditional.

Configuration
-------------

Selected as ``name: filter``.

.. list-table::
   :header-rows: 1
   :widths: 12 60 18 10

   * - Option
     - Description
     - Type
     - Default
   * - ``col``
     - Column to filter on.
     - str
     - required
   * - ``value``
     - Value used by the conditional.
     - str | int | float | bool
     - required
   * - ``condition``
     - Conditional.
     - Literal["gt", "lt", "ge", "le", "eq", "ne"]
     - required

.. code-block:: yaml

   - name: fill
     kwargs:
       col: sim
       value: 1
       dtype: int64

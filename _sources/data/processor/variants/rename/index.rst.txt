Rename
======

:doc:`Processor <../../index>` that renames columns. It accepts either a direct
mapping of old name to new name, or a parallel ``cols`` and ``out`` pair giving the
source columns and their new names. Exactly one of the two modes must be provided.

Configuration
-------------

Selected as ``name: rename``.

.. list-table::
   :header-rows: 1
   :widths: 12 62 16 10

   * - Option
     - Description
     - Type
     - Default
   * - ``map``
     - Mapping of existing column name to new name (mapping mode).
     - mapping
     - ``null``
   * - ``cols``
     - Columns to rename (paired mode, with ``out``).
     - column(s)
     - ``null``
   * - ``out``
     - New names, positionally matched to ``cols``.
     - column(s)
     - ``null``

.. code-block:: yaml

   - name: rename
     kwargs:
       map: { dom_x: x, dom_y: y, dom_z: z }

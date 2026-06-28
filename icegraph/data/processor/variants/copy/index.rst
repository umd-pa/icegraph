Copy
====

:doc:`Processor <../../index>` that copies columns from the active frame into
another working frame, joining on a shared key.

Configuration
-------------

Selected as ``name: copy``.

.. list-table::
   :header-rows: 1
   :widths: 12 65 13 10

   * - Option
     - Description
     - Type
     - Default
   * - ``to``
     - Name of the destination frame.
     - str
     - required
   * - ``by``
     - Columns to join on.
     - column(s)
     - required
   * - ``cols``
     - Columns to copy across.
     - column(s)
     - required

.. code-block:: yaml

   - name: copy
     kwargs:
       to: truth
       by: event_id
       cols: [ energy ]

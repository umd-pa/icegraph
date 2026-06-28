DOM Processor
=============

:doc:`Processor <../../index>` that converts DOM
identifiers into their Cartesian positions in the detector.

Configuration
-------------

Selected as ``name: domproc``.

.. list-table::
   :header-rows: 1
   :widths: 12 63 15 10

   * - Option
     - Description
     - Type
     - Default
   * - ``cols``
     - Columns holding the DOM identifiers. Expects 3 columns ordered as [string, OM, PMT].
     - column(s)
     - required
   * - ``out``
     - Columns the resulting positions are written to.
     - column(s)
     - required

.. code-block:: yaml

   - name: domproc
     kwargs:
       cols: [ string, om, pmt ]
       out: [ x, y, z ]

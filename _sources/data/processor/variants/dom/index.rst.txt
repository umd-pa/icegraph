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
   * - ``string``
     - Column holding the DOM string IDs.
     - str
     - required
   * - ``om``
     - Column holding the DOM OM IDs.
     - str
     - required
   * - ``pmt``
     - Column holding the DOM PMT IDs.
     - str
     - required
   * - ``out``
     - Columns the resulting positions are written to.
     - str | list[str]
     - ["x", "y", "z"]

.. code-block:: yaml

   - name: domproc
     kwargs:
       string: string
       om: om
       pmt: pmt
       out: [ x, y, z ]

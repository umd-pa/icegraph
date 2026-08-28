Zarr
====

:doc:`Reader <../../index>` variant for datasets stored as Zarr databases.

Configuration
-------------

.. list-table::
   :header-rows: 1
   :widths: 22 50 18 10

   * - Option
     - Description
     - Type
     - Default
   * - ``dense_read_fraction``
     - Density threshold, below which a point-gather load method is used.
     - float
     - 0.5

Selected as ``name: zarr``.

.. code-block:: yaml

   services:
     record:
       reader:
         name: zarr
         kwargs:
           dense_read_fraction: 0.5

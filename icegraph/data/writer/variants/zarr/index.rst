Zarr
====

:doc:`Writer <../../index>` variant that persists the processed dataset as a Zarr group.

Configuration
-------------

Selected as ``name: zarr``.

.. list-table::
   :header-rows: 1
   :widths: 18 60 12 10

   * - Option
     - Description
     - Type
     - Default
   * - ``chunk_size``
     - Approximate writer chunk size in MB.
     - int
     - 8

.. code-block:: yaml

   writer:
     name: zarr
     kwargs:
       chunk_size: 8

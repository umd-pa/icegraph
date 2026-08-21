Zarr
====

:doc:`Reader <../../index>` variant for datasets stored as Zarr databases.

Configuration
-------------

Selected as ``name: zarr``. Takes no options.

.. code-block:: yaml

   services:
     record:
       reader:
         name: zarr
         kwargs: {}

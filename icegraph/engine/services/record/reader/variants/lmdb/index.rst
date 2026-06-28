LMDB
====

:doc:`Reader <../../index>` variant for datasets stored as LMDB databases.

Configuration
-------------

Selected as ``name: lmdb``. Takes no options.

.. code-block:: yaml

   services:
     record:
       reader:
         name: lmdb
         kwargs: {}

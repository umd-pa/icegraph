LMDB
====

:doc:`Writer <../../index>` variant that persists the processed dataset as an LMDB
database.

Configuration
-------------

Selected as ``name: lmdb``.

.. list-table::
   :header-rows: 1
   :widths: 18 60 12 10

   * - Option
     - Description
     - Type
     - Default
   * - ``outdir``
     - Directory the LMDB dataset is written to.
     - path
     - required

.. code-block:: yaml

   writer:
     name: lmdb
     kwargs:
       outdir: /path/to/output

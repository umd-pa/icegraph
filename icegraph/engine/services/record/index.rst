Record
======

The **record service** provides access to the on-disk dataset. It locates the
dataset files, reads their raw records through a configurable reader, and serves
them through a store. Components and other services obtain records and dataset
attributes from this service.

Usage
-----

Configured under ``services.record``. The reader and store are themselves
selectable plugins.

.. code-block:: yaml

   services:
     record:
       source:
         - /path/to/dataset_a
         - /path/to/dataset_b
       reader:
         name: lmdb
         kwargs: {}
       store:
         name: lru-shard
         kwargs:
           cache_size: 32

How it works
------------

The :doc:`reader <reader/index>` defines the on-disk format and knows how to read
records and attributes from a single file. The :doc:`store <store/index>` presents
the collection of files as one indexable sequence and manages how much is held in
memory at once. The storage format and the caching strategy can
vary independently.

Configuration
-------------

.. list-table::
   :header-rows: 1
   :widths: 22 50 18 10

   * - Option
     - Description
     - Type
     - Default
   * - ``source``
     - One or more paths to dataset files or directories.
     - path | list[path]
     - required
   * - ``reader``
     - Reader plugin selection (``name`` / ``kwargs``).
     - mapping
     - required
   * - ``store``
     - Store plugin selection (``name`` / ``kwargs``).
     - mapping
     - required
   * - ``ignore_checksum``
     - Skip dataset checksum verification.
     - bool
     - ``false``

Sub-slots
---------

.. toctree::
   :maxdepth: 2

   reader/index
   store/index

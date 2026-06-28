LRU Shard
=========

:doc:`Store <../../index>` variant that serves the dataset from its files while
keeping only a bounded number of file shards resident in memory, evicting the
least recently used when the limit is reached. This bounds memory use on large
datasets while keeping frequently accessed shards warm.

Configuration
-------------

Selected as ``name: lru-shard``.

.. list-table::
   :header-rows: 1
   :widths: 20 55 15 10

   * - Option
     - Description
     - Type
     - Default
   * - ``cache_size``
     - Number of file shards kept resident in memory at once.
     - int
     - required

.. code-block:: yaml

   services:
     record:
       store:
         name: lru-shard
         kwargs:
           cache_size: 32

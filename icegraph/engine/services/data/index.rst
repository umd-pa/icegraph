Data
====

The **data service** provides the data loaders that feed the training,
validation, and test loops. It turns the dataset into shuffled, batched
graphs delivered to the model, and centralizes the knobs that govern
batching, shuffling, and worker parallelism.

Usage
-----

Configured under ``services.data``.

.. code-block:: yaml

   services:
     data:
       batch_size: 2048
       buffer_size: 16384
       chunk_size: 4096
       num_workers: 8
       prefetch_factor: 8
       mp_context: spawn
       persistent_workers: true

How it works
------------

Samples are read in contiguous chunks, kept columnar, and gathered into groups
of roughly ``buffer_size`` samples. Each group is permuted per record and sliced
into ready-made batches, so batches mix across chunk boundaries without the
whole dataset in memory and without materializing individual samples. A pool of
worker processes reads and assembles batches ahead of the model to reduce I/O
latency. The service builds and caches one loader per requested split.

Configuration
-------------

.. list-table::
   :header-rows: 1
   :widths: 22 50 18 10

   * - Option
     - Description
     - Type
     - Default
   * - ``batch_size``
     - Number of samples per batch.
     - int
     - required
   * - ``chunk_size``
     - Number of samples read as one contiguous block.
     - int
     - required
   * - ``buffer_size``
     - Size of the shuffle window, in samples. Chunks are grouped up to this
       size and records permuted across the group; aim for a buffer-to-chunk
       ratio of at least 8:1.
     - int
     - required
   * - ``shuffle_chunks``
     - Whether to shuffle whole chunks as blocks.
     - bool
     - ``false``
   * - ``num_workers``
     - Number of worker processes loading data.
     - int
     - required
   * - ``prefetch_factor``
     - Number of batches each worker prefetches.
     - int
     - required
   * - ``mp_context``
     - Multiprocessing start method, 'fork' is not supported.
     - ``spawn`` | ``forkserver``
     - required
   * - ``persistent_workers``
     - Keep worker processes alive between epochs.
     - bool
     - required

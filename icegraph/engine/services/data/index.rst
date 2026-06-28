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
       mp_context: fork
       persistent_workers: true

How it works
------------

Samples are read in contiguous chunks and passed through a shuffle buffer so
that batches can mix across chunk boundaries without holding the whole dataset in
memory. A pool of worker processes reads and assembles batches ahead of the model
to reduce I/O latency. The service builds and caches one loader per requested split.

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
     - Size of the shuffle buffer, in samples. Chunks are mixed within it; aim for
       a buffer-to-chunk ratio above 10:1.
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
     - Multiprocessing start method.
     - ``fork`` | ``spawn`` | ``forkserver``
     - required
   * - ``persistent_workers``
     - Keep worker processes alive between epochs.
     - bool
     - required

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
     - Size of the shuffle buffer, in chunks. Chunks are mixed within it; aim for >10 if memory allows.
     - int
     - required
   * - ``shuffle_chunks``
     - Whether to shuffle whole chunks as blocks.
     - bool
     - ``false``
   * - ``buffer_refill_threshold``
     - Fraction of the buffer size at which a refill is triggered.
     - float
     - 0.25
   * - ``max_chunks_per_epoch``
     - Maximum number of chunks drawn per epoch. Chunk order is reshuffled each
       epoch, so the full dataset is still covered after roughly
       :math:`E \approx \frac{C}{K}\left(\ln C + \gamma\right)` epochs, where
       :math:`C = \lfloor N / S \rfloor` is the total chunk count for :math:`N`
       samples of ``chunk_size`` :math:`S`, :math:`K = \lfloor C_{\max} / W
       \rfloor \cdot W` is the per-epoch budget rounded down to a multiple of
       the world size :math:`W`, and :math:`\gamma \approx 0.577`. Requires
       ``shuffle_chunks``, without it the same subset is selected every epoch.
       Pass ``-1`` to use every chunk each epoch.
     - int
     - ``-1``
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
     - Keep worker processes alive between epochs. Highly recommended for performance.
     - bool
     - required

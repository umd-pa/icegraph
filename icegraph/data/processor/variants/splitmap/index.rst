SplitMap
========

:doc:`Processor <../../index>` that assigns each event to a dataset split (for
example train, validation, or test) using a seeded random map. The ``weights`` give
the relative proportion of each split, and the assignment is deterministic for a
given seed so splits are reproducible.

Configuration
-------------

Selected as ``name: splitmap``.

.. list-table::
   :header-rows: 1
   :widths: 12 62 16 10

   * - Option
     - Description
     - Type
     - Default
   * - ``seed``
     - Random seed making the split assignment reproducible.
     - int
     - required
   * - ``range``
     - Number of buckets the weights are spread over (0 to 255).
     - int
     - required
   * - ``weights``
     - Relative proportion of each split. Must have ``range`` entries, be
       non-negative, and sum to a positive value (normalized internally).
     - list[number]
     - required

.. code-block:: yaml

   - name: splitmap
     kwargs:
       seed: 2747
       range: 10
       weights: [ 8, 8, 8, 8, 8, 8, 8, 8, 1, 1 ]

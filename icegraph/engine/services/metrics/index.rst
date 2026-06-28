Metrics
=======

The **metrics service** computes evaluation metrics during validation and testing.
It holds a user-selected list of metrics, updates them as batches are evaluated,
and reports their values. It is one way a run can measure quality beyond the training
loss.

Usage
-----

Configured under ``services.metrics``. The ``select`` list names the metrics to
compute, each a selectable plugin.

.. code-block:: yaml

   services:
     metrics:
       select:
         - name: top-k-acc
           kwargs: { k: 1 }

How it works
------------

Each selected metric accumulates state incrementally as batches arrive and resolves
to a per-head value when computed, so metrics are exact over the whole evaluation
set and combine correctly across processes under distributed execution. The
individual metrics are documented under the :doc:`metric <metric/index>` slot.

Configuration
-------------

.. list-table::
   :header-rows: 1
   :widths: 18 60 12 10

   * - Option
     - Description
     - Type
     - Default
   * - ``select``
     - Ordered list of metric selections, each a ``name`` / ``kwargs`` pair.
     - list[mapping]
     - required

Sub-slots
---------

.. toctree::
   :maxdepth: 2

   metric/index

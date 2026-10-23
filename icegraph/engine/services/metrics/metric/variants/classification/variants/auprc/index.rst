AUPRC
=====

Classification :doc:`metric <../../index>` computing the per-head macro-averaged
area under the precision-recall curve. Each class is scored one-vs-rest:
sweeping a threshold down over that class's predicted probability traces
precision against recall, and the area under that curve -- average precision --
summarizes ranking quality across every operating point rather than at the single
threshold an arg-max implies. The per-class areas are then averaged with equal
weight. Unlike AUROC, the baseline is the class prevalence, which makes AUPRC the
more honest read on a rare signal class.

Classes with no positive samples in the evaluation set are left out of the
average.

Approximation
-------------

The exact curve needs every score held and sorted, which an incremental
accumulator cannot do. Scores instead go into ``bins`` equal-width bins on
``[0, 1]``, kept separately for positives and negatives, and the threshold sweep
runs over bin edges. The area is exact to bin resolution, with ties inside a bin
resolved as if positives and negatives interleaved evenly. Raising ``bins``
tightens the estimate at linear cost in a small histogram, and the bin count is
reported as part of the metric name (ie ``auprc100``).

Configuration
-------------

Selected as ``name: auprc``.

.. list-table::
   :header-rows: 1
   :widths: 15 60 15 10

   * - Option
     - Description
     - Type
     - Default
   * - ``bins``
     - Number of equal-width score bins on ``[0, 1]``; sets the resolution of the
       threshold sweep.
     - int
     - ``100``
   * - ``from_logits``
     - Treat the model output as unnormalized logits. Set to ``false`` when the
       model already emits log-probabilities.
     - bool
     - ``true``

.. code-block:: yaml

   services:
     metrics:
       select:
         - name: auprc
           kwargs: { bins: 200 }

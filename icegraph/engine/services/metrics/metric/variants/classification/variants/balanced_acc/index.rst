Balanced Accuracy
=================

Classification :doc:`metric <../../index>` computing the per-head balanced
accuracy: accuracy as it would read if every class were equally
represented, which is the mean of the per-class recalls. It is immune to the
class imbalance that inflates plain accuracy, and in its default form is the same
quantity as :doc:`macro-recall <../macro_recall/index>`.

With ``adjusted`` set, the score is rescaled so chance sits at ``0`` rather than
``1 / K``, giving ``(score - 1 / K) / (1 - 1 / K)`` over the ``K`` classes that
have support. Perfect prediction is ``1`` either way, but the adjusted form goes
negative for a model doing worse than random.

Configuration
-------------

Selected as ``name: balanced-acc``.

.. list-table::
   :header-rows: 1
   :widths: 15 60 15 10

   * - Option
     - Description
     - Type
     - Default
   * - ``adjusted``
     - Rescale so that chance scores ``0`` instead of ``1 / K``.
     - bool
     - ``false``

.. code-block:: yaml

   services:
     metrics:
       select:
         - name: balanced-acc
           kwargs: { adjusted: true }

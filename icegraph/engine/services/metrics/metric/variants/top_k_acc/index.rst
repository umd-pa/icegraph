Top-K Accuracy
==============

:doc:`Metric <../../index>` computing the per-head top-``k`` accuracy for
classification: the fraction of samples whose true class falls within the ``k``
highest-scoring predictions. With ``k = 1`` this is ordinary top-1 accuracy.

Configuration
-------------

Selected as ``name: top-k-acc``.

.. list-table::
   :header-rows: 1
   :widths: 15 60 15 10

   * - Option
     - Description
     - Type
     - Default
   * - ``k``
     - Number of top-scoring classes considered a correct prediction.
     - int
     - ``1``

.. code-block:: yaml

   services:
     metrics:
       select:
         - name: top-k-acc
           kwargs: { k: 3 }

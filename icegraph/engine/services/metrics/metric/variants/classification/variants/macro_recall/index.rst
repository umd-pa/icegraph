Macro-Recall
============

Classification :doc:`metric <../../index>` computing the per-head macro-averaged
recall: recall is taken per class and then averaged with equal weight per
class. Because every class counts the same regardless of how many samples it
has, this diverges sharply from plain accuracy on an imbalanced evaluation set,
where a dominant class can carry accuracy on its own.

Classes with no support in the evaluation set are left out of the average rather
than counted as zero. This is the same quantity as
:doc:`balanced accuracy <../balanced_acc/index>` in its unadjusted form.

Configuration
-------------

Selected as ``name: macro-recall``. Takes no options.

.. code-block:: yaml

   services:
     metrics:
       select:
         - name: macro-recall
           kwargs: {}

Macro-F1
========

Classification :doc:`metric <../../index>` computing the per-head macro-averaged
F1 score. Per class, F1 is the harmonic mean of precision and recall
(``2 TP / (2 TP + FP + FN)``).The macro average then weights every class
equally, so a rare class counts as much as the bulk of the data. Unlike
:doc:`macro-recall <../macro_recall/index>` it also penalizes over-firing on a
class, since false positives enter the denominator.

Classes that appear neither as a true label nor as a prediction anywhere in the
evaluation set have an undefined F1 and are left out of the average.

Configuration
-------------

Selected as ``name: macro-f1``. Takes no options.

.. code-block:: yaml

   services:
     metrics:
       select:
         - name: macro-f1
           kwargs: {}

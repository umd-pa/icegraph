Per-Class Recall
================

Classification :doc:`metric <../../index>` computing the per-head, per-class
recall: for each class, the fraction of samples truly in that class that
were predicted as it. This metric reports one value per class.

It is the breakdown the macro averages summarize, and the place a collapsed
minority class shows up. A class the evaluation set never contains has undefined
recall and is reported as ``nan``.

Configuration
-------------

Selected as ``name: per-class-recall``. Takes no options.

.. code-block:: yaml

   services:
     metrics:
       select:
         - name: per-class-recall
           kwargs: {}

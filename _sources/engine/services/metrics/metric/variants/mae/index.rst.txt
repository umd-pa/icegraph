MAE
===

:doc:`Metric <../../index>` computing the per-head mean absolute error between the
model output and the targets.

Configuration
-------------

Selected as ``name: mae``. Takes no options.

.. code-block:: yaml

   services:
     metrics:
       select:
         - name: mae
           kwargs: {}

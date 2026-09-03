MAE
===

Regression :doc:`metric <../../index>` computing the per-head mean absolute error
between the model output and the targets.

* **residual**: ``|out - target|``.
* **reported**: the mean residual, in the same units as the target.

Configuration
-------------

Selected as ``name: mae``. Takes no options.

.. code-block:: yaml

   services:
     metrics:
       select:
         - name: mae
           kwargs: {}

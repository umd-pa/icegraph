MSE
===

Regression :doc:`metric <../../index>` computing the per-head mean squared error
between the model output and the targets.

* **residual**: ``(out - target)^2``.
* **reported**: the mean residual, in squared target units.

Configuration
-------------

Selected as ``name: mse``. Takes no options.

.. code-block:: yaml

   services:
     metrics:
       select:
         - name: mse
           kwargs: {}

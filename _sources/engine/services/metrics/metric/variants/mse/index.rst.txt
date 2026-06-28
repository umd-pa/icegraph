MSE
===

:doc:`Metric <../../index>` computing the per-head mean squared error between the
model output and the targets.

Configuration
-------------

Selected as ``name: mse``. Takes no options.

.. code-block:: yaml

   services:
     metrics:
       select:
         - name: mse
           kwargs: {}

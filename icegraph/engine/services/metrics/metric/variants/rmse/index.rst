RMSE
====

:doc:`Metric <../../index>` computing the per-head root mean squared error between
the model output and the targets. It is the square root of the mean squared error,
reported in the same units as the target, and is a common headline metric for
regression tasks.

Configuration
-------------

Selected as ``name: rmse``. Takes no options.

.. code-block:: yaml

   services:
     metrics:
       select:
         - name: rmse
           kwargs: {}

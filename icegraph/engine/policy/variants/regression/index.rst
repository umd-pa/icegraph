Regression
==========

:doc:`Policy <../../index>` variant for regression. It treats each prediction head
as a continuous quantity: the model output layout matches the target layout, the
targets are floating point, and the model predicts target values directly.

Configuration
-------------

Selected as ``name: regression``. Takes no options.

.. code-block:: yaml

   policy:
     name: regression
     kwargs: {}

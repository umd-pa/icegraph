MSE
===

:doc:`Loss <../../index>` variant computing the mean squared error between the
model output and the targets. It penalizes large errors quadratically and is a
standard objective for regression. Compatible with the regression policy.

Configuration
-------------

Selected as ``name: mse``.

=============  ==================================  ===================  =========
Option         Description                         Type                 Default
=============  ==================================  ===================  =========
``reduction``  How per-sample losses are reduced.  ``mean`` | ``sum``   ``mean``
=============  ==================================  ===================  =========

.. code-block:: yaml

   components:
     loss:
       name: mse
       kwargs: {}

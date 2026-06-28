L1
==

:doc:`Loss <../../index>` variant computing the mean absolute error between the
model output and the targets. It penalizes errors linearly, making it less
sensitive to outliers than the squared error. Compatible with the regression
policy.

Configuration
-------------

Selected as ``name: l1``.

=============  ==================================  ===================  =========
Option         Description                         Type                 Default
=============  ==================================  ===================  =========
``reduction``  How per-sample losses are reduced.  ``mean`` | ``sum``   ``mean``
=============  ==================================  ===================  =========

.. code-block:: yaml

   components:
     loss:
       name: l1
       kwargs: {}

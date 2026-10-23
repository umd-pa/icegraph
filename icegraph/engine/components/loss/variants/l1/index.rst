L1
==

:doc:`Loss <../../index>` variant computing the mean absolute error between the
model output and the targets. It penalizes errors linearly, making it less
sensitive to outliers than the squared error. Compatible with the regression
policy.

Configuration
-------------

Selected as ``name: l1``.

.. list-table::
   :header-rows: 1
   :widths: 15 54 21 10

   * - Option
     - Description
     - Type
     - Default
   * - ``reduction``
     - How per-sample losses are reduced.
     - ``mean`` | ``sum``
     - ``mean``
   * - ``ignore_nan``
     - Leave out NaN targets, the events a label has no value for, so they add nothing
       to the loss.
     - bool
     - ``false``

With ``ignore_nan``, each head's mean is taken over the targets kept, and a head
with no target in a batch adds nothing rather than NaN. Without it, a NaN target
makes the loss NaN.

.. code-block:: yaml

   components:
     loss:
       name: l1
       kwargs: {}

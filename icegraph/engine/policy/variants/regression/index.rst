Regression
==========

:doc:`Policy <../../index>` variant for regression. It treats each prediction head
as a continuous quantity: the model output layout matches the target layout, the
targets are floating point, and the model predicts target values directly.

Configuration
-------------

Selected as ``name: regression``.

.. list-table::
   :header-rows: 1
   :widths: 19 62 9 10

   * - Option
     - Description
     - Type
     - Default
   * - ``norm_targets``
     - Transform and normalize the targets as the features are, predicting in that
       space and mapping predictions back to target units.
     - bool
     - ``true``

With ``norm_targets`` on, the loss and metrics are computed on the normalized
targets. A target that is NaN stays NaN. Set
``ignore_nan`` on the :doc:`loss <../../../components/loss/index>` to train on such
a label.

.. code-block:: yaml

   policy:
     name: regression
     kwargs:
       norm_targets: true

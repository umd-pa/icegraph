Expected Calibration Error
==========================

Classification :doc:`metric <../../index>` computing the per-head expected
calibration error. Calibration asks whether a stated confidence means what it says
(ie "of the predictions made at 80% confidence, are 80% of them right?"). Samples are
binned by the probability of their predicted class, and the ECE is the
population-weighted mean gap between each bin's accuracy and its mean confidence.

``0`` is a perfectly calibrated model. A typical over-confident network sits well
above it while its accuracy looks fine, which is exactly the failure this metric
exists to expose.

The estimate is the standard top-label ECE over ``bins`` equal-width bins on
``[0, 1]``. The bin count trades resolution against how many samples land in each
bin so it is reported as part of the metric name (ie ``ece15``, ``ece50``) and two
bin counts can be selected side by side.

Configuration
-------------

Selected as ``name: ece``.

.. list-table::
   :header-rows: 1
   :widths: 15 60 15 10

   * - Option
     - Description
     - Type
     - Default
   * - ``bins``
     - Number of equal-width confidence bins on ``[0, 1]``.
     - int
     - ``15``
   * - ``from_logits``
     - Treat the model output as unnormalized logits. Set to ``false`` when the
       model already emits log-probabilities.
     - bool
     - ``true``

.. code-block:: yaml

   services:
     metrics:
       select:
         - name: ece
           kwargs: { bins: 15 }

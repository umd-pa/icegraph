Cross-Entropy
=============

Classification :doc:`metric <../../index>` computing the per-head cross-entropy,
in nats: the mean negative log-probability the model assigned to
the true class. Unlike the accuracy-style metrics it reads the whole predicted
distribution rather than just its arg-max, so it separates a model that is right
by a hair from one that is right with conviction, and punishes confident mistakes
hard.

Configuration
-------------

Selected as ``name: cross-entropy``.

.. list-table::
   :header-rows: 1
   :widths: 15 60 15 10

   * - Option
     - Description
     - Type
     - Default
   * - ``from_logits``
     - Treat the model output as unnormalized logits. Set to ``false`` when the
       model already emits log-probabilities.
     - bool
     - ``true``

.. code-block:: yaml

   services:
     metrics:
       select:
         - name: cross-entropy
           kwargs: {}

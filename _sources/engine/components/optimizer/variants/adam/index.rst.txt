Adam
====

:doc:`Optimizer <../../index>` variant using the AdamW algorithm: adaptive
per-parameter step sizes with decoupled weight decay. It is the default choice for
most training and is robust to a wide range of learning rates.

Configuration
-------------

Selected as ``name: adam``.

================  ====================================================  ================  ================
Option            Description                                           Type              Default
================  ====================================================  ================  ================
``lr``            Learning rate.                                        float             required
``betas``         Coefficients for the running gradient and squared-    (float, float)    ``(0.9, 0.999)``
                  gradient averages.
``eps``           Term added to the denominator for numerical           float             ``1e-8``
                  stability.
``weight_decay``  Decoupled weight-decay coefficient.                   float             ``1e-2``
``amsgrad``       Whether to use the AMSGrad variant.                   bool              ``False``
``maximize``      Whether to maximize rather than minimize.             bool              ``False``
================  ====================================================  ================  ================

.. code-block:: yaml

   components:
     optimizer:
       name: adam
       kwargs:
         lr: 0.0002
         weight_decay: 0.00005

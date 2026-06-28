SGD
===

:doc:`Optimizer <../../index>` variant implementing stochastic gradient descent,
optionally with momentum and Nesterov acceleration. It offers more direct control
than the adaptive optimizers and is a common choice when a carefully tuned
learning-rate schedule is used.

Configuration
-------------

Selected as ``name: sgd``.

================  ====================================================  ======  =========
Option            Description                                           Type    Default
================  ====================================================  ======  =========
``lr``            Learning rate.                                        float   required
``momentum``      Momentum factor.                                      float   ``0.0``
``dampening``     Dampening for momentum.                               float   ``0.0``
``weight_decay``  Weight-decay (L2 penalty) coefficient.                float   ``0.0``
``nesterov``      Enable Nesterov momentum. Requires ``momentum > 0``.  bool    ``False``
``maximize``      Whether to maximize rather than minimize.             bool    ``False``
================  ====================================================  ======  =========

.. code-block:: yaml

   components:
     optimizer:
       name: sgd
       kwargs:
         lr: 0.01
         momentum: 0.9
         nesterov: true

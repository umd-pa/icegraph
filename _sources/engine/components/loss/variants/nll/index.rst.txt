NLL
===

:doc:`Loss <../../index>` variant computing the negative log-likelihood for
multiclass classification. By default it treats the model output as logits (and is
then equivalent to cross entropy); set ``from_logits`` to ``false`` when the model
already emits log-probabilities. Compatible with the multiclass policy.

Configuration
-------------

Selected as ``name: nll``.

================  ====================================================  ===================  =========
Option            Description                                           Type                 Default
================  ====================================================  ===================  =========
``reduction``     How per-sample losses are reduced.                    ``mean`` | ``sum``   ``mean``
``weight``        Optional per-class weights for class imbalance.       list[float] | null   ``null``
``ignore_index``  Target value that is ignored and contributes no       int                  ``-100``
                  gradient.
``from_logits``   Whether the model output is logits (``true``) or      bool                 ``true``
                  log-probabilities (``false``).
================  ====================================================  ===================  =========

.. code-block:: yaml

   components:
     loss:
       name: nll
       kwargs:
         from_logits: false

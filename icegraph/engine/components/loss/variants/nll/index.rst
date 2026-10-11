NLL
===

:doc:`Loss <../../index>` variant computing the negative log-likelihood for
multiclass classification. By default it treats the model output as logits (and is
then equivalent to cross entropy); set ``from_logits`` to ``false`` when the model
already emits log-probabilities. Compatible with the multiclass policy.

Configuration
-------------

Selected as ``name: nll``.

.. list-table::
   :header-rows: 1
   :widths: 17 54 20 9

   * - Option
     - Description
     - Type
     - Default
   * - ``reduction``
     - How per-sample losses are reduced.
     - ``mean`` | ``sum``
     - ``mean``
   * - ``weight``
     - Optional per-class weights for class imbalance.
     - list[float] | null
     - ``null``
   * - ``ignore_index``
     - Target value that is ignored and contributes no gradient.
     - int
     - ``-100``
   * - ``from_logits``
     - Whether the model output is logits (``true``) or log-probabilities (``false``).
     - bool
     - ``true``

.. code-block:: yaml

   components:
     loss:
       name: nll
       kwargs:
         from_logits: false

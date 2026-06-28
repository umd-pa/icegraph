Cross Entropy
=============

:doc:`Loss <../../index>` variant for multiclass classification. It interprets the
model output as per-class logits and penalizes the negative log-probability of the
correct class. Compatible with the multiclass policy.

Configuration
-------------

Selected as ``name: cross-entropy``.

===================  ====================================================  ===================  =========
Option               Description                                           Type                 Default
===================  ====================================================  ===================  =========
``reduction``        How per-sample losses are reduced.                    ``mean`` | ``sum``   ``mean``
``weight``           Optional per-class weights for class imbalance.       list[float] | null   ``null``
``ignore_index``     Target value that is ignored and contributes no       int                  ``-100``
                     gradient.
``label_smoothing``  Amount of label smoothing in ``[0, 1]``.              float                ``0.0``
===================  ====================================================  ===================  =========

.. code-block:: yaml

   components:
     loss:
       name: cross-entropy
       kwargs: {}

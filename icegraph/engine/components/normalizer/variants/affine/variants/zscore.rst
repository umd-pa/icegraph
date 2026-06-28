ZScore
======

Affine :doc:`normalizer <../index>` that centers each feature column on its
training-split mean and scales it to unit variance, the standard z-score
transform.

* **offset**: the column mean.
* **scale**: the reciprocal of the column standard deviation.

Usage
-----

Selected as ``name: zscore``. Takes no options.

.. code-block:: yaml

   components:
     normalizer:
       name: zscore
       kwargs: {}

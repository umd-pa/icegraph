Unit Variance
=============

Affine :doc:`normalizer <../index>` that scales each feature column to unit
variance without recentering it.

* **offset**: none (zero).
* **scale**: the reciprocal of the column standard deviation.

Usage
-----

Selected as ``name: unit-variance``. Takes no options.

.. code-block:: yaml

   components:
     normalizer:
       name: unit-variance
       kwargs: {}

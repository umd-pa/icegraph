Mean Centering
==============

Affine :doc:`normalizer <../index>` that subtracts each column's mean
without scaling it.

* **offset**: the column mean.
* **scale**: none (one).

Usage
-----

Selected as ``name: mean-centering``. Takes no options.

.. code-block:: yaml

   components:
     normalizer:
       name: mean-centering
       kwargs: {}

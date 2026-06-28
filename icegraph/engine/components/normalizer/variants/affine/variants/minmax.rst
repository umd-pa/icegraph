MinMax
======

Affine :doc:`normalizer <../index>` that rescales each column into the
``[0, 1]`` range.

* **offset**: the column minimum.
* **scale**: the reciprocal of the column range (max - min).

Usage
-----

Selected as ``name: minmax``. Takes no options.

.. code-block:: yaml

   components:
     normalizer:
       name: minmax
       kwargs: {}

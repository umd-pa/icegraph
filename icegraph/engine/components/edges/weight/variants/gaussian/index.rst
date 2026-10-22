Gaussian
========

:doc:`Edge weight <../../index>` that weights each edge using a gaussian.

.. code-block:: text

   d^2    = sum_i (a_i - b_i)^2
   weight = exp( -d^2 / (2 sigma^2) )

Closer neighbours weigh more. The weight lies in ``(0, 1]`` and is 1 at zero
distance.

``sigma`` sets the distance at which neighbours stop mattering, in the units of
``cols``. The weight is about 0.61 at one ``sigma``, 0.14 at two, and 0.01 at
three. Choose it near the typical distance to a node's neighbours.

Configuration
-------------

Selected as ``name: gaussian``.

.. list-table::
   :header-rows: 1
   :widths: 20 55 15 10

   * - Option
     - Description
     - Type
     - Default
   * - ``cols``
     - Logical feature column groups the endpoint distance is taken over.
     - list[str]
     - required
   * - ``sigma``
     - Width of the kernel, in the units of ``cols``. Must be positive.
     - float
     - required

.. code-block:: yaml

   components:
     edges:
       name: knn
       kwargs:
         k: 10
         neighbor_cols: [ dom_x, dom_y, dom_z ]
         weights:
           - name: gaussian
             kwargs:
               cols: [ dom_x, dom_y, dom_z ]
               sigma: 50.0

The attribute produced has shape ``[E, 1]``.

Gaussian
========

KNN :doc:`edge builder <../../index>` that weights each edge using a gaussian.

.. code-block:: text

   d^2    = sum_i (a_i - b_i)^2
   weight = exp( -d^2 / (2 sigma^2) )

Closer neighbours weigh more. The weight lies in ``(0, 1]`` and is 1 at zero
distance.

``sigma`` sets the distance at which neighbours stop mattering, in the units of
``weight_cols``. The weight is about 0.61 at one ``sigma``, 0.14 at two, and 0.01 at
three. Choose it near the typical distance to a node's neighbours.

The columns weighting an edge are configured separately from those spanning the neighbour search.

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
   * - ``k``
     - Number of nearest neighbours connected per node.
     - int
     - required
   * - ``neighbor_cols``
     - Logical feature column groups spanning the space the neighbour search runs
       in.
     - list[str]
     - required
   * - ``weight_cols``
     - Logical feature column groups the endpoint distance is taken over.
     - list[str]
     - required
   * - ``sigma``
     - Width of the kernel, in the units of ``weight_cols``. Must be positive.
     - float
     - required

.. code-block:: yaml

   components:
     edges:
       name: gaussian
       kwargs:
         k: 10
         neighbor_cols: [ dom_x, dom_y, dom_z ]
         weight_cols: [ dom_x, dom_y, dom_z ]
         sigma: 50.0

The attribute produced has shape ``[E, 1]``.

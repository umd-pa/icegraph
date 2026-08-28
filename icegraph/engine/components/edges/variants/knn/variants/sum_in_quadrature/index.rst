Sum in Quadrature
=================

KNN :doc:`edge builder <../../index>` that weights each edge by the quadrature sum of
the differences between its two endpoints, taken over a configurable set of
feature columns. For a position triplet this is the Euclidean distance between the
two nodes.

.. code-block:: text

   weight = sqrt( sum_i (a_i - b_i)^2 )

The columns summed for the weight are configured separately from those spanning
the neighbour search, so an edge may be weighted in a different space from the one
its endpoints were selected in.

Configuration
-------------

Selected as ``name: sum-in-quadrature``.

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
     - Logical feature column groups summed in quadrature to form the edge weight.
     - list[str]
     - required

.. code-block:: yaml

   components:
     edges:
       name: sum-in-quadrature
       kwargs:
         k: 10
         neighbor_cols: [ dom_pos ]
         weight_cols: [ dom_pos ]

The attribute produced has shape ``[E, 1]``.

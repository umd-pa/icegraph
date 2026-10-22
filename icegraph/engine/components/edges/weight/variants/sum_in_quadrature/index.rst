Sum in Quadrature
=================

:doc:`Edge weight <../../index>` that weights each edge by the quadrature sum of
the differences between its two endpoints, taken over a configurable set of
feature columns. For a position triplet this is the Euclidean distance between the
two nodes.

.. code-block:: text

   weight = sqrt( sum_i (a_i - b_i)^2 )

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
   * - ``cols``
     - Logical feature column groups summed in quadrature to form the edge weight.
     - list[str]
     - required

.. code-block:: yaml

   components:
     edges:
       name: knn
       kwargs:
         k: 10
         neighbor_cols: [ dom_x, dom_y, dom_z ]
         weights:
           - name: sum-in-quadrature
             kwargs:
               cols: [ dom_x, dom_y, dom_z ]

The attribute produced has shape ``[E, 1]``.

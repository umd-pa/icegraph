KNN
===

:doc:`Edge builder <../../index>` that connects every node to the ``k`` nodes
closest to it within its own graph.

How it works
------------

The builder resolves ``neighbor_cols`` against the feature block to obtain the
coordinates each node occupies, then runs a batched neighbour search that uses the
batch vector to keep every edge inside a single graph. Where a graph holds fewer
than ``k + 1`` nodes, that graph simply contributes fewer edges.

Edges follow the orientation ``torch_geometric`` uses for message passing:
``edge_index[0]`` is the neighbour a message travels from, and ``edge_index[1]``
is the node it arrives at. Under the default ``source_to_target`` flow, each node
therefore aggregates over its own ``k`` nearest neighbours.

The edge attributes come from the :doc:`edge weights <../../weight/index>` listed
under ``weights``. The columns those weights read are set per weight, separately
from ``neighbor_cols``, so an edge may be weighted in a different space from the
one its endpoints were selected in.

Configuration
-------------

Selected as ``name: knn``.

.. list-table::
   :header-rows: 1
   :widths: 20 55 15 10

   * - Option
     - Description
     - Type
     - Default
   * - ``k``
     - Number of nearest neighbours connected per node. Must be at least 1.
     - int
     - required
   * - ``neighbor_cols``
     - Logical feature column groups spanning the space the neighbour search runs
       in. Must name at least one group.
     - list[str]
     - required
   * - ``weights``
     - Edge weights to run on the edge index, each given as ``name`` and
       ``kwargs``. Their outputs are concatenated in this order.
     - list[EdgeWeightSelection]
     - ``[]``

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
           - name: gaussian
             kwargs:
               cols: [ dom_x, dom_y, dom_z ]
               sigma: 50.0

The attributes produced in this example have shape ``[E, 2]``: column 0 is the
sum-in-quadrature weight and column 1 is the gaussian weight.

KNN
===

:doc:`Processor <../../index>` that builds graph edges by connecting each node to
its ``k`` nearest neighbors in a coordinate space. It produces the edge index and
the corresponding edge weights (neighbor distances).

Configuration
-------------

Selected as ``name: knn``.

.. list-table::
   :header-rows: 1
   :widths: 12 63 15 10

   * - Option
     - Description
     - Type
     - Default
   * - ``by``
     - Columns identifying the per-event grouping the graph is built within.
     - column(s)
     - required
   * - ``col``
     - Column holding the per-node coordinates used for neighbor search.
     - str | int
     - required
   * - ``out``
     - Two columns to write: the edge index and the edge weights.
     - column(s)
     - required
   * - ``k``
     - Number of nearest neighbors connected per node.
     - int
     - required

.. code-block:: yaml

   - name: knn
     kwargs:
       by: event_id
       col: position
       out: [ edge_index, edge_attr ]
       k: 8

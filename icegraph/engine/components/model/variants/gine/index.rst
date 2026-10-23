GINE
====

:doc:`Model <../../index>` variant implementing a graph isomorphism network with
edge features (GINE) for graph-level prediction. Each layer adds a projection of
the edge attributes to the neighbouring node features, sums the result over the
neighbours of each node, adds the node's own features scaled by ``1 + eps``, and
passes the sum through a two layer MLP and an activation. The final node features
are mean-pooled per graph and projected to the output width.

Configuration
-------------

Selected as ``name: gine``.

.. list-table::
   :header-rows: 1
   :widths: 25 50 10 15

   * - Option
     - Description
     - Type
     - Default
   * - ``hidden_layers``
     - Number of GINE convolution layers.
     - int
     - required
   * - ``hidden_channels``
     - Width of each hidden layer.
     - int
     - required
   * - ``edge_channels``
     - Width of the edge attributes produced by the edge builder.
     - int
     - ``1``
   * - ``train_eps``
     - Learn ``eps`` in each layer instead of fixing it at ``0``.
     - bool
     - ``false``
   * - ``checkpoint``
     - Recompute each layer during the backward pass instead of storing its
       activations. Each layer stores a tensor of size edges by channels, so this
       cuts training memory by a large factor at the cost of roughly one extra
       forward pass. Only applies in training.
     - bool
     - ``false``

.. code-block:: yaml

   components:
     model:
       name: gine
       kwargs:
         hidden_layers: 4
         hidden_channels: 256

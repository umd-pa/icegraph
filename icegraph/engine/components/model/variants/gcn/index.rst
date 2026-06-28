GCN
===

:doc:`Model <../../index>` variant implementing a graph convolutional network for
graph-level prediction. Each layer performs a graph convolution that mixes node
features along edges, followed by a linear projection and activation; the final
node features are mean-pooled per graph and projected to the output width.

.. note::

   GCN uses scalar edge weights, so it expects edge attributes of shape ``[E, 1]``.

Configuration
-------------

Selected as ``name: gcn``.

.. list-table::
   :header-rows: 1
   :widths: 25 50 10 15

   * - Option
     - Description
     - Type
     - Default
   * - ``hidden_layers``
     - Number of graph-convolution blocks.
     - int
     - required
   * - ``hidden_channels``
     - Width of each hidden layer.
     - int
     - required

.. code-block:: yaml

   components:
     model:
       name: gcn
       kwargs:
         hidden_layers: 4
         hidden_channels: 256

GravNet
=======

:doc:`Model <../../index>` variant implementing the GravNet architecture for
graph-level prediction. Each block learns a low-dimensional latent space, connects
each node to its nearest neighbors in that space, and aggregates features over
those learned neighbors. Final node features are mean-pooled per
graph and projected to the output width.

See: `Learning representations of irregular particle-detector geometry with distance-weighted graph networks <https://doi.org/10.48550/arXiv.1902.07987>`_

Configuration
-------------

Selected as ``name: gravnet``.

========================  ==============================================================  ======  =========
Option                    Description                                                     Type    Default
========================  ==============================================================  ======  =========
``hidden_layers``         Number of GravNet blocks.                                       int     required
``hidden_channels``       Width of each hidden layer.                                     int     required
``num_neighbors``         Number of nearest neighbors aggregated per node.                int     required
``space_dimensions``      Dimensionality of the learned latent space used for neighbor    int     required
                          search.
``propagate_dimensions``  Dimensionality of the features propagated between neighbors.    int     required
========================  ==============================================================  ======  =========

.. code-block:: yaml

   components:
     model:
       name: gravnet
       kwargs:
         hidden_layers: 4
         hidden_channels: 256
         num_neighbors: 8
         space_dimensions: 4
         propagate_dimensions: 22
